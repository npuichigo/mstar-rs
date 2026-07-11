"""Orpheus on mstar-rs — the first streaming model (T4 streaming tier).

Two concurrent partitions (mstar's `get_partitions`):

- ``LLM`` — ``prefill`` (embed the voice+text prompt, causal, sample the
  first audio token) then a ``decode`` Loop (one token/step, KV append 1,
  sample with temperature/top-p/repetition-penalty, EOS-stop). Every pass
  streams the sampled ``new_token`` to the SNAC partition.
- ``SNAC`` — a self-triggered ``snac_chunk`` walk. The runtime's stream
  buffer collects tokens under a SlidingWindow(28, 7) policy and feeds each
  28-token window to the vendored SNAC decoder, which emits a 24 kHz int16
  audio chunk. Partition-done rides the pass consuming the final window.

Like pi05, this reuses mstar's own nn modules unmodified — `OrpheusForCausalLM`
behind a FlashInfer causal cache handle (llama3 RoPE), mstar's `Sampler`, and
the vendored `SNAC` decoder + its token->code de-interleaving.

NOTE: `canopylabs/orpheus-3b-0.1-ft` (weights) and `-0.1-pretrained`
(tokenizer) are gated on HuggingFace; this runs once that access is granted.
"""

from __future__ import annotations

from typing import Any

import torch

from ..fi import (
    BatchedFlashInferAttention,
    FlashInferAttention,
    FlashInferCacheHandle,
    FlashInferPagedKV,
)
from ..graph import (
    connection,
    edge,
    emit,
    loop,
    node,
    partition,
    sliding_window,
    stream_edge,
    EMPTY_DESTINATION,
)
from ..model import ModelEngine, ModelPolicy, NextWalk

DEFAULT_MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"
KV_LABEL = "main"
PAGE_SIZE = 128
NUM_PAGES = 256  # 32k tokens; prompt is short, decode up to max_output_tokens
# Last page, reserved for CUDA-graph capture warmup writes (a real request's
# ascending page allocation never reaches it: prompt + 2048 decode < 20 pages).
SCRATCH_PAGE = NUM_PAGES - 1
MAX_DECODE_BATCH = 8       # largest batch we'll capture a decode graph for
MAX_PAGES_PER_REQ = 40     # ceil((prompt + max_output_tokens) / PAGE_SIZE) bound


class OrpheusPolicy(ModelPolicy):
    """Control plane (conductor): topology, request ingestion, the LLM->SNAC
    continuation policy, postprocess. Builds mstar's `OrpheusModel` for config
    + tokenizer ONLY (no get_submodule -> no GB model weights). Holds the small
    (~1 MB) batched-sampler buffers, which register/next_forward touch; the GB
    OrpheusForCausalLM + SNAC weights live in `OrpheusEngine`. So the conductor
    is weightless of the model (the point of the split), even though it carries
    the tiny sampler state."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        max_output_tokens: int = 2048,
        greedy: bool = False,
        cuda_graph: bool = True,
    ) -> None:
        from mstar.distributed.communication import TPCommGroup
        from mstar.model.orpheus.orpheus_model import OrpheusModel
        from mstar.utils.sampling import (
            Sampler,
            SamplerBuffers,
            SamplingConfig,
            SeenTokenMask,
        )

        self.device = torch.device(device)
        self.max_output_tokens = max_output_tokens
        self.greedy = greedy
        self.cuda_graph = cuda_graph
        # config + tokenizer only (weights load lazily via get_submodule, which
        # the engine — not this policy — calls).
        self._mstar = OrpheusModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.tokenizer = self._mstar.tokenizer
        self.sampler = Sampler(device=self.device)  # eager path (cuda_graph=False)
        dev = self.device
        V = self.cfg.vocab_size
        # Batched in-graph sampler (mstar's SamplerBuffers): per-request slots +
        # a [max_bs, V] seen-token mask for the repetition penalty (orpheus
        # defaults to rep_penalty=1.3, so this is live). ~1 MB, not the model.
        self._SamplingConfig = SamplingConfig
        self._SeenTokenMask = SeenTokenMask
        self._sbuf = SamplerBuffers.allocate(
            MAX_DECODE_BATCH, dev, tp_group=TPCommGroup.trivial(), vocab_size=V
        )
        self._seen: dict[int, Any] = {}  # request_id -> SeenTokenMask

    # -- graph + topology -------------------------------------------------

    def walks(self) -> dict[str, Any]:
        return {
            "prefill": node(
                "LLM",
                ["text_inputs"],
                [
                    edge(EMPTY_DESTINATION, "new_token", persist=True),
                    stream_edge("snac_decoder", "new_token", "SNAC"),
                ],
            ),
            "decode": loop(
                "decode_loop",
                node(
                    "LLM",
                    ["text_inputs"],
                    [
                        edge("LLM", "text_inputs"),
                        stream_edge("snac_decoder", "new_token", "SNAC"),
                    ],
                ),
                max_iters=self.max_output_tokens,
            ),
            "snac_chunk": node(
                "snac_decoder",
                ["new_token"],
                [emit("audio_chunk", modality="audio")],
            ),
        }

    def kv_config(self):
        return [(KV_LABEL, NUM_PAGES, PAGE_SIZE)], {"LLM": KV_LABEL}

    def unbatchable(self):
        # decode + prefill both batch now (decode: per-N cuda graph + batched
        # sampler; prefill: one eager ragged forward over concatenated prompts).
        # SNAC still replays a single-slot graph, so it stays capped to one
        # request per batch in cuda_graph mode.
        if not self.cuda_graph:
            return []
        return [("snac_decoder", "snac_chunk")]

    def partitions(self):
        return (
            [
                partition("LLM", ["prefill", "decode"]),
                partition("SNAC", ["snac_chunk"]),
            ],
            [
                connection(
                    "LLM", "SNAC", "new_token",
                    sliding_window(
                        self.cfg.snac_window_tokens, self.cfg.snac_stride_tokens
                    ),
                )
            ],
        )

    # -- request ingestion ------------------------------------------------

    def _tokenize(self, prompt: str, voice: str) -> torch.Tensor:
        toks = self.tokenizer(f"{voice}: {prompt}", return_tensors="pt").input_ids[0]
        start = torch.tensor([self.cfg.start_token_id], dtype=torch.long)
        end = torch.tensor(self.cfg.end_token_ids, dtype=torch.long)
        return torch.cat([start, toks, end]).to(self.device)

    def _register_batched(self, rid: int, seed: int) -> None:
        """Register a request with the batched sampler: allocate its slot +
        seed its config (greedy -> temperature=0, which SamplerBuffers encodes
        as top_k=1 argmax), and create its per-request seen-token mask."""
        cfg = self._SamplingConfig(
            vocab_size=self.cfg.vocab_size,
            temperature=0.0 if self.greedy else self.cfg.temperature,
            top_k=1 if self.greedy else 0,
            top_p=1.0 if self.greedy else self.cfg.top_p,
            repetition_penalty=1.0 if self.greedy else self.cfg.repetition_penalty,
            ignore_eos=False,
        )
        cfg.set_seed(seed)
        self._sbuf.register_request(str(rid), cfg)
        self._seen[rid] = self._SeenTokenMask.new(str(rid), self.cfg.vocab_size, self.device)

    def initial_walks(self, request: dict[str, Any]) -> list[NextWalk]:
        rid = request["request_id"]
        text_ids = self._tokenize(request.get("prompt", ""), request.get("voice", "tara"))
        seed = request.get("seed", 0)
        if self.cuda_graph:
            self._register_batched(rid, seed)
        else:
            # Per-request eager sampler state (faithful to mstar's config + seed).
            srid = str(rid)
            self.sampler.add_request(srid)
            self.sampler.set_config(
                srid,
                vocab_size=self.cfg.vocab_size,
                temperature=0.0 if self.greedy else self.cfg.temperature,
                top_p=1.0 if self.greedy else self.cfg.top_p,
                repetition_penalty=1.0 if self.greedy else self.cfg.repetition_penalty,
                ignore_eos=False,
            )
            self.sampler._sampling_config[srid].set_seed(seed)
        # LLM prefill (KV append = prompt length) + self-triggered SNAC walk.
        return [
            ("prefill", [("LLM", "text_inputs", [text_ids])], {KV_LABEL: text_ids.shape[0]}),
            ("snac_chunk", []),
        ]

    # -- policy: LLM prefill -> decode -> done; SNAC re-arm ------------------

    def next_forward(
        self, request_id, partition, walk, fwd_index, persist, stream_done
    ) -> NextWalk | None:
        if partition == "LLM":
            if walk == "prefill":
                # Seed decode's first token with the token sampled at prefill.
                first = persist["new_token"]
                return (
                    "decode",
                    [("LLM", "text_inputs", first)],
                    {KV_LABEL: 1},
                )
            if self.cuda_graph:
                self._sbuf.unregister_request(str(request_id))
                self._seen.pop(request_id, None)
            else:
                self.sampler.remove_request(str(request_id))
            return None  # decode finished -> LLM partition done
        # SNAC: re-arm until the final window is consumed.
        return None if stream_done else ("snac_chunk", [])

    def postprocess(self, name, modality, tensors):
        assert modality == "audio"
        return tensors[0].detach().cpu()  # int16 PCM, 24 kHz


class OrpheusEngine(OrpheusPolicy, ModelEngine):
    """Data plane (worker): loads the GB OrpheusForCausalLM + SNAC weights (via
    get_submodule) + KV cache + CUDA-graph state, and runs `execute`. Being an
    `OrpheusPolicy` too, it is also a full `Model` for the single-process
    Driver."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        max_output_tokens: int = 2048,
        greedy: bool = False,
        cuda_graph: bool = True,
    ) -> None:
        from mstar.distributed.communication import TPCommGroup

        super().__init__(model_id, device, max_output_tokens, greedy, cuda_graph)
        # lm_head is a ColumnParallelLinear; needs a TP group even at ws 1.
        llm_sub = self._mstar.get_submodule(
            "LLM", device=str(device), tp_group=TPCommGroup.trivial(),
            autocast_dtype=torch.bfloat16,
        )
        self.embed_tokens = llm_sub.embed_tokens
        self.language_model = llm_sub.language_model  # OrpheusForCausalLM
        self.lm_head = llm_sub.lm_head
        self.snac_sub = self._mstar.get_submodule("snac_decoder", device=str(device))
        self.kv_cache = FlashInferPagedKV(
            num_layers=self.cfg.num_hidden_layers,
            num_pages=NUM_PAGES,
            page_size=PAGE_SIZE,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._prefill_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim,
            self.device, max_new_tokens=2048, cudagraph=False, causal=True,
        )
        self._decode_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim,
            self.device, max_new_tokens=1, cudagraph=cuda_graph, causal=True,
        )
        self._prefill_battn: dict[int, Any] = {}  # ragged batched prefill, per N
        self._pending_stops: list[tuple[int, str]] = []
        dev = self.device
        # Batched decode CUDA-graph state (request-major rows), per N.
        self._g_tokens = torch.zeros(MAX_DECODE_BATCH, dtype=torch.long, device=dev)
        self._g_sampled = torch.zeros(MAX_DECODE_BATCH, dtype=torch.long, device=dev)
        self._dec_attn: dict[int, Any] = {}
        self._dec_graph: dict[int, Any] = {}
        self._dec_sampler: dict[int, Any] = {}
        # SNAC single-slot graph state.
        self._g_snac_tokens = torch.zeros(
            self.cfg.snac_window_tokens, dtype=torch.long, device=self.device
        )
        n_pcm = self.cfg.snac_audio_slice_end - self.cfg.snac_audio_slice_start
        self._g_snac_pcm = torch.zeros(n_pcm, dtype=torch.int16, device=self.device)
        self._snac_graph: torch.cuda.CUDAGraph | None = None
        self.token_log: dict[int, list[int]] | None = None  # debug: token stream
        self.logit_log: dict[int, list[torch.Tensor]] | None = None  # debug: logits

    # -- node execution ----------------------------------------------------

    def _run_llm(self, text_ids: torch.Tensor, view, attn: FlashInferAttention):
        """Embed -> causal LLM over the paged cache -> logits for the last
        token. `text_ids` is the prompt (prefill) or the single prev token."""
        attn.plan(view["pages"], view["seq_pos"], view["append_len"])
        handle = FlashInferCacheHandle(attn)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = self.embed_tokens(text_ids)
            hidden = self.language_model(emb, cache_handle=handle)
            logits = self.lm_head(hidden[-1:])  # (1, vocab)
        return logits

    def _ensure_decode_graph(self, n: int, batch_plan: list, srids: list, seen_masks: list):
        """Return the capture-time sampler for batch size `n`, capturing its
        decode graph on first use. The graph reads `_g_tokens[:n]`, runs the
        batched paged decode + lm_head -> logits [n, V], and SAMPLES in-graph
        (per-request temperature/top_p/rep_penalty via the gathered
        SamplerBuffers) into `_g_sampled[:n]`. Attention is (re)planned outside
        capture over every request's page table. The sampler offset/seen state
        touched by warmup is restored so the first real decode continues the
        request's stream (mstar's decode dance)."""
        if n in self._dec_graph:
            self._dec_attn[n].plan(batch_plan)
            return self._dec_sampler[n]
        attn = BatchedFlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim, self.device,
            bs=n, max_pages_per_req=MAX_PAGES_PER_REQ, new_len=1, cudagraph=True, causal=True,
        )
        self._dec_attn[n] = attn
        handle = FlashInferCacheHandle(attn)
        attn.plan(batch_plan)
        self._sbuf.stage_seen_token_masks(srids, seen_masks)
        sampler = self._sbuf.gather_for_request_ids(srids, n, gather_seen_tokens=True)
        self._dec_sampler[n] = sampler

        def dec_fn() -> None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = self.embed_tokens(self._g_tokens[:n])
                hidden = self.language_model(emb, cache_handle=handle)
                logits = self.lm_head(hidden)  # [n, V] — each row is a last token
            token = sampler.sample(srids, logits, apply_penalty=True)
            self._g_sampled[:n].copy_(token.view(n))

        saved_off = self._sbuf.offset_buf[:n].clone()
        saved_seen = self._sbuf.seen_tokens.buf[:n].clone()
        torch.cuda.synchronize()
        for _ in range(2):  # warmup (autotune); writes to the current pages
            dec_fn()
        self._sbuf.offset_buf[:n].copy_(saved_off)
        self._sbuf.seen_tokens.buf[:n].copy_(saved_seen)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dec_fn()
        torch.cuda.synchronize()
        self._dec_graph[n] = graph
        return sampler

    def _snac_forward(self) -> None:
        """Capturable SNAC decode over the static `_g_snac_tokens` (always
        exactly snac_window_tokens, padded outside): token->code de-interleave
        -> SNAC decode -> middle slice -> int16, into `_g_snac_pcm`. Reuses
        mstar's own helpers; the pad branch in `_tokens_to_codes` is never
        taken here (input is already a full window), so the graph records the
        no-pad path."""
        codes = self.snac_sub._tokens_to_codes(self._g_snac_tokens)  # (1, 4, 7)
        c0, c1, c2 = self.snac_sub._extract_snac_codes(codes)
        audio = self.snac_sub.snac_model.decode([c0, c1, c2])
        audio = audio[
            :, :, self.cfg.snac_audio_slice_start:self.cfg.snac_audio_slice_end
        ]
        self._g_snac_pcm.copy_((audio.clamp(-1, 1) * 32767).to(torch.int16).view(-1))

    def _capture_snac(self) -> None:
        self._g_snac_tokens.fill_(self.cfg.custom_token_base_id + 10)  # code 0
        torch.cuda.synchronize()
        for _ in range(2):
            self._snac_forward()
        torch.cuda.synchronize()
        self._snac_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._snac_graph):
            self._snac_forward()
        torch.cuda.synchronize()

    def _execute_decode_batched(self, inputs, kv) -> dict:
        """One decode step for a BATCH of requests: load each prev-token into
        its row, plan batched attention over every request's pages, replay the
        per-batch-size graph (batched paged decode + in-graph batched sample),
        split per request. The per-step seen-mask stage/gather (before) and
        sync (after) mirror mstar's cuda_graph_runner decode dance."""
        rids = list(inputs)
        n = len(rids)
        assert n <= MAX_DECODE_BATCH, f"decode batch {n} > {MAX_DECODE_BATCH}"
        srids = [str(r) for r in rids]
        batch_plan = []
        for j, rid in enumerate(rids):
            self._g_tokens[j] = inputs[rid]["text_inputs"][0].view(())
            v = kv[rid]
            batch_plan.append((v["pages"], v["seq_pos"], 1))
        seen_masks = [self._seen[r] for r in rids]
        self._ensure_decode_graph(n, batch_plan, srids, seen_masks)
        if self.logit_log is not None:
            self._record_decode_logits(n, batch_plan, rids)  # extra eager forward
        self._sbuf.stage_seen_token_masks(srids, seen_masks)
        self._sbuf.gather_for_request_ids(srids, n, gather_seen_tokens=True)
        self._dec_graph[n].replay()
        self._dec_sampler[n].sync_seen_token_masks(seen_masks)
        out = {}
        for j, rid in enumerate(rids):
            tok = self._g_sampled[j : j + 1].clone().to(torch.long)
            ti = int(tok.item())
            if ti == self.cfg.stop_token_id:
                self._pending_stops.append((rid, "decode_loop"))
            if self.token_log is not None:
                self.token_log.setdefault(rid, []).append(ti)
            out[rid] = {"new_token": [tok], "text_inputs": [tok]}
        return out

    def _record_decode_logits(self, n: int, batch_plan: list, rids: list) -> None:
        """Verification hook: run one EAGER batched decode forward (no graph)
        to capture per-request pre-sample logits. Writes the same new-token K/V
        the graph replay will (idempotent), so running it just before replay is
        safe. Only used when `logit_log` is set."""
        eattn = BatchedFlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim, self.device,
            bs=n, max_pages_per_req=MAX_PAGES_PER_REQ, new_len=1, cudagraph=False, causal=True,
        )
        eattn.plan(batch_plan)
        eh = FlashInferCacheHandle(eattn)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = self.embed_tokens(self._g_tokens[:n])
            hidden = self.language_model(emb, cache_handle=eh)
            logits = self.lm_head(hidden)  # [n, V]
        for j, rid in enumerate(rids):
            self.logit_log.setdefault(rid, []).append(logits[j].detach().float().cpu())

    def _get_prefill_battn(self, n: int):
        if n not in self._prefill_battn:
            self._prefill_battn[n] = BatchedFlashInferAttention(
                self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim, self.device,
                bs=n, max_pages_per_req=MAX_PAGES_PER_REQ, new_len=None, cudagraph=False,
                causal=True, max_total_q=MAX_DECODE_BATCH * 2048,
            )
        return self._prefill_battn[n]

    def _execute_prefill_batched(self, inputs, kv) -> dict:
        """Batched prefill: one causal forward over the concatenated per-request
        prompts (ragged batched attention), take each request's LAST-token
        logits, and sample its first audio token via the batched sampler."""
        rids = list(inputs)
        n = len(rids)
        srids = [str(r) for r in rids]
        embs, batch_plan, last = [], [], []
        off = 0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for rid in rids:
                view = kv[rid]
                embs.append(self.embed_tokens(inputs[rid]["text_inputs"][0]))
                batch_plan.append((view["pages"], view["seq_pos"], view["append_len"]))
                off += view["append_len"]
                last.append(off - 1)
            attn = self._get_prefill_battn(n)
            attn.plan(batch_plan)
            handle = FlashInferCacheHandle(attn)
            hidden = self.language_model(torch.cat(embs, dim=0), cache_handle=handle)
            logits = self.lm_head(hidden[torch.tensor(last, device=self.device)])  # [n, V]
        self._sbuf.stage_seen_token_masks(srids, [self._seen[r] for r in rids])
        sampler = self._sbuf.gather_for_request_ids(srids, n, gather_seen_tokens=True)
        tokens = sampler.sample(srids, logits, apply_penalty=True)  # [n]
        out = {}
        for j, rid in enumerate(rids):
            tok = tokens[j : j + 1].view(1).to(torch.long)
            self._seen[rid].add_tokens(tok)  # so decode penalises token 0
            if self.token_log is not None:
                self.token_log.setdefault(rid, []).append(int(tok.item()))
            out[rid] = {"new_token": [tok]}
        return out

    @torch.inference_mode()
    def execute(self, node_name, walk, inputs, kv=None):
        if node_name == "LLM" and self.cuda_graph:
            if walk == "decode":
                return self._execute_decode_batched(inputs, kv)
            return self._execute_prefill_batched(inputs, kv)
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        for rid, named in inputs.items():
            if node_name == "LLM":  # eager path (cuda_graph=False)
                attn = self._prefill_attn if walk == "prefill" else self._decode_attn
                logits = self._run_llm(named["text_inputs"][0], kv[rid], attn)
                token = self.sampler.sample([str(rid)], logits, apply_penalty=True).view(1).to(torch.long)
                if walk == "decode" and int(token.item()) == self.cfg.stop_token_id:
                    self._pending_stops.append((rid, "decode_loop"))
                if self.token_log is not None:
                    self.token_log.setdefault(rid, []).append(int(token.item()))
                out = {"new_token": [token]}
                if walk == "decode":
                    out["text_inputs"] = [token]  # loop-back
                outputs[rid] = out
            elif node_name == "snac_decoder":
                # A window of token-id tensors (in order) -> one audio chunk.
                window = torch.stack(named["new_token"]).flatten().to(self.device)
                if self.cuda_graph:
                    w = self.cfg.snac_window_tokens
                    # Pad the (only-ever-short final) window up to a full one,
                    # exactly as mstar's `_tokens_to_codes` does, then replay
                    # the fixed-shape graph.
                    if window.shape[0] < w:
                        pad = window[-1].repeat(w - window.shape[0])
                        window = torch.cat([window, pad])
                    if self._snac_graph is None:
                        self._capture_snac()
                    self._g_snac_tokens.copy_(window)
                    self._snac_graph.replay()
                    pcm = self._g_snac_pcm.clone()
                else:
                    codes = self.snac_sub._tokens_to_codes(window)  # (N, 4, 7)
                    c0, c1, c2 = self.snac_sub._extract_snac_codes(codes)
                    audio = self.snac_sub.snac_model.decode([c0, c1, c2])
                    audio = audio[
                        :, :, self.cfg.snac_audio_slice_start:self.cfg.snac_audio_slice_end
                    ]
                    pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).squeeze(1).squeeze(0)
                outputs[rid] = {"audio_chunk": [pcm]}
            else:
                raise ValueError(f"unknown node: {node_name}")
        return outputs

    def loops_to_finish(self) -> list[tuple[int, str]]:
        stops = self._pending_stops
        self._pending_stops = []
        return stops


Orpheus = OrpheusEngine  # full model (policy + engine) — single-process Driver
# / both roles. Multi-process: conductor gets an OrpheusPolicy (no model
# weights), each worker an OrpheusEngine.
