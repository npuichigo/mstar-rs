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

from ..fi import FlashInferAttention, FlashInferCacheHandle, FlashInferPagedKV
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
from ..model import Model, NextWalk

DEFAULT_MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"
KV_LABEL = "main"
PAGE_SIZE = 128
NUM_PAGES = 256  # 32k tokens; prompt is short, decode up to max_output_tokens
# Last page, reserved for CUDA-graph capture warmup writes (a real request's
# ascending page allocation never reaches it: prompt + 2048 decode < 20 pages).
SCRATCH_PAGE = NUM_PAGES - 1


class Orpheus(Model):
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
        from mstar.utils.sampling import CudaGraphableSampler, Sampler

        self.device = torch.device(device)
        self.max_output_tokens = max_output_tokens
        self.greedy = greedy
        self.cuda_graph = cuda_graph
        self._mstar = OrpheusModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.tokenizer = self._mstar.tokenizer
        # Orpheus' lm_head is a ColumnParallelLinear; it needs a TP comm
        # group even at world_size 1 (mstar's engine passes trivial()).
        llm_sub = self._mstar.get_submodule(
            "LLM",
            device=str(device),
            tp_group=TPCommGroup.trivial(),
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
        # Causal attention for both prefill (multi-token) and decode (1 token).
        self._prefill_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim,
            self.device, max_new_tokens=2048, cudagraph=False, causal=True,
        )
        self._decode_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_attention_heads, self.cfg.head_dim,
            self.device, max_new_tokens=1, cudagraph=cuda_graph, causal=True,
        )
        self.sampler = Sampler(device=self.device)  # eager path (cuda_graph=False)
        # In-graph sampler (bs=1): pre-allocated per-request buffers sampled
        # INSIDE the decode graph, exactly as mstar's decode graph captures its
        # `sampler.sample`. Per-step offset advance + seen-token scatter are
        # both capturable in-place ops. Greedy is encoded as (temp=1, top_k=1).
        dev = self.device
        V = self.cfg.vocab_size
        self._g_sampler = CudaGraphableSampler(
            temperature_buf=torch.ones(1, device=dev),
            top_k_buf=torch.zeros(1, dtype=torch.int32, device=dev),
            top_p_buf=torch.ones(1, device=dev),
            seed_buf=torch.zeros(1, dtype=torch.long, device=dev),
            offset_buf=torch.zeros(1, dtype=torch.long, device=dev),
            rep_penalty_buf=torch.ones(1, device=dev),
            seen_tokens_buf=torch.zeros(1, V, dtype=torch.bool, device=dev),
            tp_group=TPCommGroup.trivial(),
        )
        self._g_sampled = torch.zeros(1, dtype=torch.long, device=dev)
        self._pending_stops: list[tuple[int, str]] = []
        # In cuda_graph mode the sampler/decode/SNAC graphs are single-slot
        # (bs=1 static buffers), so only ONE request may be in flight through
        # the LLM partition at a time — a second would reset `_g_sampler` state
        # mid-decode and corrupt the first. Tracked on the conductor instance.
        self._active_llm_rid: int | None = None
        # CUDA-graph decode state: one graph captures the single-token forward
        # (embed -> 28 layers over the paged cache -> lm_head -> logits) with
        # static token input + logits output. Plan runs outside the graph and
        # updates the wrapper's fixed page/pos buffers; sampling runs outside.
        self._g_token = torch.zeros(1, dtype=torch.long, device=self.device)
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        # CUDA-graph SNAC state: every window is <= snac_window_tokens (the
        # final partial flush is padded up to it), so one fixed-shape graph
        # covers all chunks. Static token input, static PCM output.
        self._g_snac_tokens = torch.zeros(
            self.cfg.snac_window_tokens, dtype=torch.long, device=self.device
        )
        n_pcm = self.cfg.snac_audio_slice_end - self.cfg.snac_audio_slice_start
        self._g_snac_pcm = torch.zeros(n_pcm, dtype=torch.int16, device=self.device)
        self._snac_graph: torch.cuda.CUDAGraph | None = None
        # Debug: when set, `execute` records the LLM token stream per request
        # (for verification against an independent reference decode).
        self.token_log: dict[int, list[int]] | None = None

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
        # In cuda_graph mode every node replays single-slot bs=1 graphs
        # (sampler / decode / SNAC over static buffers), so the scheduler must
        # serialize requests through them. The eager path could batch (the
        # per-request sampler is keyed by rid), so it declares nothing.
        if not self.cuda_graph:
            return []
        return [("LLM", "prefill"), ("LLM", "decode"), ("snac_decoder", "snac_chunk")]

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

    def _config_g_sampler(self, seed: int) -> None:
        """Reset the in-graph sampler for a new request (bs=1): seed, offset,
        seen mask, and the sampling params. Greedy -> (temp=1, top_k=1) argmax,
        as `sample_cuda_graphable_gpu` documents (temp==0 would div-by-zero)."""
        temp = 1.0 if self.greedy else self.cfg.temperature
        top_k = 1 if self.greedy else 0  # 0 -> disabled (full vocab) in-kernel
        top_p = 1.0 if self.greedy else self.cfg.top_p
        rep = 1.0 if self.greedy else self.cfg.repetition_penalty
        self._g_sampler.temperature_buf.fill_(temp)
        self._g_sampler.top_k_buf.fill_(top_k)
        self._g_sampler.top_p_buf.fill_(top_p)
        self._g_sampler.rep_penalty_buf.fill_(rep)
        self._g_sampler.seed_buf.fill_(seed)
        self._g_sampler.offset_buf.zero_()
        self._g_sampler.seen_tokens_buf.zero_()

    def initial_walks(self, request: dict[str, Any]) -> list[NextWalk]:
        rid = request["request_id"]
        text_ids = self._tokenize(request.get("prompt", ""), request.get("voice", "tara"))
        seed = request.get("seed", 0)
        if self.cuda_graph:
            if self._active_llm_rid is not None:
                raise RuntimeError(
                    "orpheus cuda_graph mode is single-in-flight (bs=1 sampler/"
                    f"decode/SNAC graphs); request {self._active_llm_rid} is "
                    f"still active when {rid} was submitted — serialize requests"
                )
            self._active_llm_rid = rid
            self._config_g_sampler(seed)
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

    def _decode_forward(self, handle: FlashInferCacheHandle) -> None:
        """The capturable single-token decode step over static buffers: reads
        `_g_token`, runs the 28-layer paged decode + lm_head, then SAMPLES
        in-graph into `_g_sampled` (matching mstar, whose decode graph captures
        its `sampler.sample`). The 28 per-layer KV reads/writes and the
        sampler's offset-advance + seen-token scatter all bake into the graph;
        KV writes target the wrapper's static token_page/token_slot buffers,
        so replay lands at whatever plan() set them to."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = self.embed_tokens(self._g_token)
            hidden = self.language_model(emb, cache_handle=handle)
            logits = self.lm_head(hidden[-1:])
        token = self._g_sampler.sample(["_"], logits, apply_penalty=True)
        self._g_sampled.copy_(token.view(1))

    def _capture_decode(self) -> None:
        """Capture the decode graph once. Warmup writes go to SCRATCH_PAGE
        (a real request never allocates it), so the committed cache is
        untouched; the graph records writes to the static buffers, which real
        replays repoint at the request's actual pages via plan()."""
        handle = FlashInferCacheHandle(self._decode_attn)
        self._decode_attn.plan([SCRATCH_PAGE], seq_pos=0, new_len=1)
        self._g_token.fill_(self.cfg.start_token_id)
        # The warmup passes run the in-graph sampler, advancing offset_buf and
        # scattering into seen_tokens_buf. Snapshot both and restore after
        # capture so the first real decode continues the request's RNG/seen
        # stream (the prefill already sampled token 0 -> offset=1).
        saved_offset = self._g_sampler.offset_buf.clone()
        saved_seen = self._g_sampler.seen_tokens_buf.clone()
        torch.cuda.synchronize()
        for _ in range(2):  # warmup (writes to scratch page; triggers autotune)
            self._decode_forward(handle)
        torch.cuda.synchronize()
        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph):
            self._decode_forward(handle)
        torch.cuda.synchronize()
        self._g_sampler.offset_buf.copy_(saved_offset)
        self._g_sampler.seen_tokens_buf.copy_(saved_seen)

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

    @torch.inference_mode()
    def execute(self, node_name, walk, inputs, kv=None):
        # cuda_graph mode replays single-slot graphs (sampler/decode/SNAC) over
        # bs=1 static buffers; batching >1 request in one execute call would
        # interleave their state and corrupt every output. Fail loud. (This is
        # the bs=1 path verified bit-exact; per-request graph state is future
        # work — see the [bs,V] buffers mstar uses.)
        if self.cuda_graph and len(inputs) > 1:
            raise RuntimeError(
                f"orpheus cuda_graph execute is single-request; got {len(inputs)}"
            )
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        for rid, named in inputs.items():
            if node_name == "LLM":
                if self.cuda_graph and walk == "decode":
                    view = kv[rid]
                    if self._decode_graph is None:
                        self._capture_decode()  # one-time (uses scratch page)
                    # Plan the real step (append 1) into the fixed buffers,
                    # load the token, replay — the token is sampled IN-GRAPH.
                    self._decode_attn.plan(view["pages"], view["seq_pos"], 1)
                    self._g_token.copy_(named["text_inputs"][0].view(1))
                    self._decode_graph.replay()
                    token = self._g_sampled.clone()
                elif self.cuda_graph:  # prefill: eager forward, in-graph sampler
                    logits = self._run_llm(named["text_inputs"][0], kv[rid], self._prefill_attn)
                    token = self._g_sampler.sample(["_"], logits, apply_penalty=True).view(1)
                else:  # eager path (cuda_graph=False)
                    attn = self._prefill_attn if walk == "prefill" else self._decode_attn
                    logits = self._run_llm(named["text_inputs"][0], kv[rid], attn)
                    token = self.sampler.sample([str(rid)], logits, apply_penalty=True)
                token = token.view(1).to(torch.long)  # (1,)
                # EOS -> stop the decode loop (mstar's check_stop).
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

    # -- policy ------------------------------------------------------------

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
            self.sampler.remove_request(str(request_id))
            if self._active_llm_rid == request_id:
                self._active_llm_rid = None  # sampler free; a new request may start
            return None  # decode finished -> LLM partition done
        # SNAC: re-arm until the final window is consumed.
        return None if stream_done else ("snac_chunk", [])

    def loops_to_finish(self) -> list[tuple[int, str]]:
        stops = self._pending_stops
        self._pending_stops = []
        return stops

    def postprocess(self, name, modality, tensors):
        assert modality == "audio"
        return tensors[0].detach().cpu()  # int16 PCM, 24 kHz
