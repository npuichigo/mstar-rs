"""Qwen3-Omni on mstar-rs — the flagship 3-partition dual-AR streaming model.

Ports `mstar/model/qwen3_omni/` (Qwen/Qwen3-Omni-30B-A3B-Instruct): a Thinker
(30B-A3B MoE, text AR) whose per-token hidden states stream to a Talker (MoE
audio-token AR, with a nested 15-step code-predictor depth loop per frame),
whose codec tokens stream to a Code2Wav vocoder (24 kHz audio). Multimodal in
(text/audio/image/video via audio_encoder + vision_encoder), text + speech out.

This file is being brought up in two stages:

  * **Stage 1 (here): the control plane** — `Qwen3OmniPolicy` declares the 8
    walks, 3 partitions, 3 streaming connections, and 2 KV labels. This
    validates that the Rust runtime can *express and compile* qwen3-omni's
    dual-AR streaming topology (the hardest control shape so far) with NO model
    weights — see `examples/verify_qwen3_graph.py`.
  * **Stage 2 (next): the engine** — `Qwen3OmniEngine` will load the Thinker /
    Talker / Code2Wav + encoders (reusing mstar's nn modules behind `fi.py`
    handles) and run `execute`, verified vs a dense reference on GPU. That
    needs the ~60 GB weights + the dual RoPE regimes, the nested depth loop,
    the cross-partition hidden-state streaming, and the conductor `talker_
    trigger` gating — none of which have an orpheus analog.

Topology (from mstar's get_graph_walk_graphs / get_partitions / get_partition_
topology):

    Thinker ──thinker_states/mask (fixed chunk 1)──▶ Talker
                                                       │
                                          codec_tokens (left-context 25+25)
                                                       ▼
                                                   Code2Wav ──▶ audio
"""

from __future__ import annotations

from typing import Any

from ..graph import (
    EMPTY_DESTINATION,
    connection,
    edge,
    emit,
    fixed_chunk,
    left_context,
    loop,
    node,
    partition,
    sequential,
    stream_edge,
)
from ..model import ModelEngine, ModelPolicy

DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
THINKER_KV = "thinker"
TALKER_KV = "talker"
PAGE_SIZE = 128
THINKER_PAGES = 512   # 64k text-context tokens
THINKER_TEXT_PAGES = 64   # 8k context for the text-only Thinker AR slice
MROPE_SECTION = [24, 20, 20]  # for head_dim 128
TALKER_PAGES = 256

# Multimodal placeholder/marker token ids for THIS checkpoint (verified against
# the processor output; the config.py values 151646/7/8 are stale). The
# processor expands each media placeholder to N tokens; the encoder embeds are
# masked-scattered onto the AUDIO_PAD / IMAGE_PAD positions.
AUDIO_PAD, AUDIO_BOS, AUDIO_EOS = 151675, 151669, 151670
IMAGE_PAD, IMAGE_BOS, IMAGE_EOS = 151655, 151652, 151653


def _thinker_prefill_seeds(mstar, request):
    """Build the Thinker prefill walk name + seed inputs from a request (text +
    media file paths). Feature extraction (HF processor, CPU — no weights) runs
    here in the weightless conductor; the encoder forward + masked-scatter run
    in the Thinker engine. The processor expands each media placeholder to N
    tokens, so `input_ids` already carries the AUDIO_PAD/IMAGE_PAD spans the
    engine scatters onto. Shared by the text-out and audio-out policies.
    Returns (walk_name, seeds, ids). Single media item per modality (v1)."""
    import torch

    fp = request.get("file_paths") or {}
    prompt = request.get("prompt") or request.get("text", "") or ""
    proc = mstar._processor
    if fp.get("audio"):
        import numpy as np
        import soundfile as sf
        wav, sr = sf.read(fp["audio"][0])
        if sr != 16000:  # the feature extractor is 16 kHz
            n = int(len(wav) * 16000 / sr)
            wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                            np.arange(len(wav)), wav)
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": wav},
                                             {"type": "text", "text": prompt}]}]
        out = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")
        ids = out["input_ids"][0]
        m = out["feature_attention_mask"].bool()
        feats = out["input_features"].permute(0, 2, 1)[m].permute(1, 0).contiguous().float()
        seqlens = out["feature_attention_mask"].sum(-1).to(torch.long)
        return ("prefill_audio",
                [("Thinker", "text_inputs", [ids]),
                 ("Thinker", "audio_features", [feats]),
                 ("Thinker", "audio_seqlens", [seqlens])], ids)
    if fp.get("image"):
        from PIL import Image
        img = Image.open(fp["image"][0]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        out = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")
        ids = out["input_ids"][0]
        return ("prefill_vision",
                [("Thinker", "text_inputs", [ids]),
                 ("Thinker", "pixel_values", [out["pixel_values"].contiguous().float()]),
                 ("Thinker", "image_grid_thw", [out["image_grid_thw"].to(torch.long)])], ids)
    text = proc.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
    ids = mstar.tokenizer(text, return_tensors="pt")["input_ids"][0]
    return ("prefill_text", [("Thinker", "text_inputs", [ids])], ids)
MAX_OUTPUT_TOKENS = 2048  # mstar: get_max_output_tokens() (both AR loops)
# Talker→Code2Wav windowing (mstar: codec_chunk_frames / codec_left_context).
CODEC_CHUNK_FRAMES = 25
CODEC_LEFT_CONTEXT_FRAMES = 25


class Qwen3OmniPolicy(ModelPolicy):
    """Control plane: the 3-partition dual-AR streaming graph. Weightless —
    the Thinker/Talker/Code2Wav weights live in `Qwen3OmniEngine` (Stage 2)."""

    def __init__(self, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> None:
        self.max_output_tokens = max_output_tokens

    # -- walks: 8 (4 Thinker, 3 Talker, 1 Code2Wav) -----------------------

    def _thinker_out(self) -> list[dict[str, Any]]:
        """Thinker's outward edges: emit the text token to the client, and
        stream the (reduced) hidden states + mask to the Talker, one per step
        (FixedChunkPolicy chunk=1)."""
        return [
            emit("new_token", modality="text", persist=True),
            stream_edge("Talker", "thinker_states", "Talker"),
            stream_edge("Talker", "thinker_mask", "Talker"),
        ]

    def _talker_out(self) -> list[dict[str, Any]]:
        """Talker's outward edge: stream the 16 codec tokens/frame to Code2Wav
        (LeftContextChunkPolicy 25+25)."""
        return [stream_edge("Code2Wav", "codec_tokens", "Code2Wav")]

    def walks(self) -> dict[str, Any]:
        return {
            # --- Thinker partition ---
            "prefill_text": node("Thinker", ["text_inputs"], self._thinker_out()),
            "prefill_audio": sequential(
                node("audio_encoder", ["audio_features", "audio_seqlens"],
                     [edge("Thinker", "audio_embeds")]),
                node("Thinker", ["audio_embeds"], self._thinker_out()),
            ),
            "prefill_vision": sequential(
                node("vision_encoder", ["pixel_values", "image_grid_thw"],
                     [edge("Thinker", "vision_embeds"), edge("Thinker", "deepstack")]),
                node("Thinker",
                     ["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
                     self._thinker_out()),
            ),
            "thinker_decode": loop(
                "thinker_decode_loop",
                node("Thinker", ["text_inputs"],
                     [edge("Thinker", "text_inputs")] + self._thinker_out()),  # self-feedback + emit/stream
                max_iters=self.max_output_tokens,
            ),
            # --- Talker partition (fed streaming thinker_states + a conductor
            # talker_trigger; the last prefill step starts emitting codec) ---
            "talker_prefill": node(
                "Talker", ["thinker_states", "thinker_mask", "talker_trigger"], []
            ),  # KV-extend only, no outputs
            "talker_last_prefill": sequential(
                node("Talker", ["thinker_states", "thinker_mask", "talker_trigger"],
                     [edge(EMPTY_DESTINATION, "talker_input_embeds", persist=True)]
                     + self._talker_out()),
            ),
            "talker_decode": loop(
                "talker_decode_loop",
                node("Talker", ["thinker_states", "thinker_mask", "talker_input_embeds"],
                     [edge("Talker", "talker_input_embeds")] + self._talker_out()),  # self-feedback + stream
                max_iters=self.max_output_tokens,
            ),
            # --- Code2Wav partition ---
            "code2wav_chunk": node("Code2Wav", ["codec_tokens"],
                                   [emit("audio_chunk", modality="audio")]),
        }

    # -- partitions + streaming topology ----------------------------------

    def partitions(self):
        return (
            [
                partition("Thinker", ["prefill_text", "prefill_audio",
                                      "prefill_vision", "thinker_decode"]),
                partition("Talker", ["talker_prefill", "talker_last_prefill",
                                     "talker_decode"]),
                partition("Code2Wav", ["code2wav_chunk"]),
            ],
            [
                # Thinker feeds the Talker one hidden-state token at a time;
                # continue_after_done keeps the Talker's stream alive past the
                # Thinker's EOS so its own decode can drain.
                connection("Thinker", "Talker", "thinker_states",
                           fixed_chunk(1, continue_after_done=True)),
                connection("Thinker", "Talker", "thinker_mask",
                           fixed_chunk(1, continue_after_done=True)),
                # Talker feeds Code2Wav in 25-frame chunks with 25-frame
                # left-context overlap (warms the causal ConvNet; trimmed from
                # emitted audio in the vocoder).
                connection("Talker", "Code2Wav", "codec_tokens",
                           left_context(CODEC_CHUNK_FRAMES, CODEC_LEFT_CONTEXT_FRAMES)),
            ],
        )

    def kv_config(self):
        # Two separate paged caches — one per AR partition (mstar keeps a
        # Thinker cache and a Talker cache, both label "main"; flattened here to
        # distinct labels). The Code Predictor's dense depth-loop KV is a plain
        # tensor inside the engine, not an engine KV label.
        return (
            [(THINKER_KV, THINKER_PAGES, PAGE_SIZE), (TALKER_KV, TALKER_PAGES, PAGE_SIZE)],
            {"Thinker": THINKER_KV, "Talker": TALKER_KV},
        )


# =========================================================================
# Stage 2, slice 1: the Thinker text-only AR engine (multi-GPU / TP).
#
# The Thinker (~39B MoE, ~78 GB) does not fit one 80 GB card, so it MUST run
# tensor-parallel — the engine builds it SHARDED across `tp_world` ranks (NCCL
# SPMD via mstar's WorkerTPGroups; ~31 GB/rank shard-on-load). This is mstar's
# `output_modalities='text'` mode: the Thinker generates text with no Talker /
# Code2Wav, a single-partition prefill -> decode loop (the same control shape
# as orpheus). It is driven through the TP-aware conductor/worker in dist.py
# (a node maps to the LIST of its ranks; only rank 0's output is routed) —
# see examples/verify_thinker_tp_dist.py. The full 3-partition audio path
# (Qwen3OmniPolicy above) layers on top later.
# =========================================================================


class Qwen3OmniThinkerPolicy(ModelPolicy):
    """Conductor-side (weightless) control plane for text-only Thinker AR:
    config + tokenizer only, the prefill->decode graph, and the AR loop
    policy. Qwen3OmniThinkerEngine adds the sharded weights + execute."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 max_output_tokens: int = 256, greedy: bool = True) -> None:
        from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

        self._mstar = Qwen3OmniModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.tokenizer = self._mstar.tokenizer
        self.max_output_tokens = max_output_tokens
        self.greedy = greedy

    def walks(self) -> dict[str, Any]:
        prefill_emit = [emit("new_token", modality="text", persist=True)]
        return {
            "prefill_text": node("Thinker", ["text_inputs"], prefill_emit),
            # Media prefills: the encoder embeds are scattered onto the placeholder
            # positions in the Thinker engine, so each declares its extra seeds.
            "prefill_audio": node(
                "Thinker", ["text_inputs", "audio_features", "audio_seqlens"], prefill_emit),
            "prefill_vision": node(
                "Thinker", ["text_inputs", "pixel_values", "image_grid_thw"], prefill_emit),
            "thinker_decode": loop(
                "thinker_decode_loop",
                node("Thinker", ["text_inputs"],
                     [edge("Thinker", "text_inputs"),
                      emit("new_token", modality="text", persist=True)]),
                max_iters=self.max_output_tokens,
            ),
        }

    def kv_config(self):
        return [(THINKER_KV, THINKER_TEXT_PAGES, PAGE_SIZE)], {"Thinker": THINKER_KV}

    def _tokenize(self, prompt: str):
        text = self._mstar._processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0]

    def initial_walks(self, request: dict[str, Any]):
        walk, seeds, ids = _thinker_prefill_seeds(self._mstar, request)
        return [(walk, seeds, {THINKER_KV: int(ids.shape[0])})]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        # prefill sampled the first token (persisted); seed the decode loop with
        # it (KV append = 1). decode returning None finishes the request. Every
        # prefill variant (text/audio/vision) transitions to the same decode.
        if walk.startswith("prefill"):
            return ("thinker_decode", [("Thinker", "text_inputs", persist["new_token"])],
                    {THINKER_KV: 1})
        return None

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].reshape(-1)[0].item())  # emitted token id


class Qwen3OmniThinkerEngine(Qwen3OmniThinkerPolicy, ModelEngine):
    """Worker-side data plane: builds the Thinker SHARDED across `tp_world`
    ranks (get_submodule with a real TPCommGroup, NCCL-initialized via
    WorkerTPGroups in this __init__ — the same pattern the toy engine uses),
    a per-rank head-sharded fi KV pool, and runs execute. MRoPE is applied by
    the model, so the same FlashInferCacheHandle used for pi05/orpheus drives
    the Thinker unchanged."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 tp_rank: int = 0, tp_world: int = 1, max_output_tokens: int = 256,
                 greedy: bool = True, tp_port: int = 29700,
                 audio_output: bool = False, cuda_graph: bool = True) -> None:
        import torch

        from mstar.distributed.communication import TPCommGroup, WorkerTPGroups
        from mstar.model.qwen3_omni.components.rope import (
            compute_3d_cos_sin,
            compute_rope_freqs,
            get_rope_index_audio,
            get_rope_index_text,
            get_rope_index_vision,
        )

        from ..fi import (
            FlashInferAttention,
            FlashInferCacheHandle,
            FlashInferPagedKV,
        )

        super().__init__(model_id, max_output_tokens, greedy)
        self._torch = torch
        self._Handle = FlashInferCacheHandle
        self._cos_sin = compute_3d_cos_sin
        self._rope_index = get_rope_index_text
        self._rope_text = get_rope_index_text
        self._rope_audio = get_rope_index_audio
        self._rope_vision = get_rope_index_vision
        # Encoders load lazily on the first media request (replicated per rank,
        # not sharded). Text-only serving never pays for them.
        self._audio_encoder = None
        self._vision_encoder = None
        # Per-request RoPE-vs-KV position offset: for a vision prefill the grid
        # span exceeds the token count, so decode tokens advance RoPE from the
        # grid span while the KV cache advances by token count (verify_vision's
        # dual counters). 0 for text/audio.
        self._rope_offset: dict[int, int] = {}

        # comm group + NCCL init (mstar's WorkerTPGroups); init_dist sets cuda:rank.
        members = list(range(tp_world))
        cg = TPCommGroup(my_global_rank=tp_rank, my_group_rank=tp_rank, group_members=members)
        tp = WorkerTPGroups(num_workers=tp_world, global_rank=tp_rank,
                            any_tp=(tp_world > 1), world_tp_groups=[tuple(members)])
        tp.add("Thinker", cg)
        tp.init_dist(init_method=f"tcp://127.0.0.1:{tp_port}")
        self.device = torch.device(f"cuda:{tp_rank}")

        thinker = self._mstar.get_submodule(
            "Thinker", device=str(self.device), tp_group=cg, autocast_dtype=torch.bfloat16,
        )
        self._thinker_sub = thinker                        # for _get_talker_text_mask
        self.audio_output = audio_output
        self._tmodel = thinker.model                       # Qwen3OmniThinkerModel
        self._embed = thinker.model.model.embed_tokens
        self._lm_head = thinker.model.lm_head

        tc = self.cfg.thinker_text
        head_dim = getattr(self.cfg, "thinker_head_dim", None) or getattr(tc, "head_dim", 128)
        qo = tc.num_attention_heads // tp_world            # per-rank query heads
        kv = max(1, tc.num_key_value_heads // tp_world)    # per-rank kv heads
        self._inv_freq = compute_rope_freqs(head_dim, tc.rope_theta, self.device)
        self._cache = FlashInferPagedKV(
            tc.num_hidden_layers, THINKER_TEXT_PAGES, PAGE_SIZE, kv, head_dim,
            self.device, torch.bfloat16,
        )
        self._attn = FlashInferAttention(
            self._cache, qo, head_dim, self.device, max_new_tokens=2048,
            cudagraph=False, causal=True, dtype=torch.bfloat16,
        )
        self._eos = self.tokenizer.eos_token_id
        self._generated: dict[int, int] = {}
        self._pending_stops: list[tuple[int, str]] = []

        # --- per-request sampling (mstar's model_kwargs-at-ingest) ---------
        # Default is greedy (temperature=0 -> the sampler's one-hot argmax
        # path), so behavior without kwargs is unchanged. `thinker_temperature`
        # / `thinker_top_k` / `thinker_top_p` / `seed` arrive via
        # register_request. The eager Sampler covers prefill + the eager decode
        # fallback; the CUDA-graph path uses SamplerBuffers (below), whose
        # tp_group broadcast keeps the sampled token identical across the TP
        # ranks (without it, ranks drift on the first tied-logit draw).
        from mstar.utils.sampling import Sampler, SamplerBuffers, SamplingConfig

        self._SamplingConfig = SamplingConfig
        self.sampler = Sampler(device=self.device, tp_group=cg)
        self._req_cfg: dict[int, Any] = {}   # rid -> SamplingConfig

        # --- CUDA-graph capture of the per-token decode (TP-safe) ---------
        # Capture embed -> Thinker forward (incl. the 2 NCCL all-reduces/layer,
        # graph-capturable) -> lm_head -> in-graph CudaGraphableSampler (greedy
        # rows one-hot argmax), over static buffers. My fi decode wrapper is in
        # cudagraph mode (plan() outside); MRoPE cos/sin are recomputed per
        # token and copied in (position advances each step). Per-request
        # sampling params re-stage into the SamplerBuffers OUTSIDE the graph
        # (the graph reads the per-step buffer views), so no recapture per
        # request.
        self.cuda_graph = cuda_graph
        self._head_dim = head_dim
        self._g_tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._g_cos = torch.zeros(1, head_dim, dtype=torch.bfloat16, device=self.device)
        self._g_sin = torch.zeros(1, head_dim, dtype=torch.bfloat16, device=self.device)
        self._g_out_tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._g_layer0 = torch.zeros(1, tc.hidden_size, dtype=torch.bfloat16, device=self.device)
        self._decode_attn = FlashInferAttention(
            self._cache, qo, head_dim, self.device, max_new_tokens=1,
            cudagraph=cuda_graph, causal=True, dtype=torch.bfloat16,
        )
        self._sbuf = SamplerBuffers.allocate(1, self.device, tp_group=cg)
        self._decode_sampler = None
        self._sampler_rid = None   # whose config is staged in the per-step bufs
        self._decode_graph = None

    # -- per-request sampling config (mstar's model_kwargs-at-ingest) -------

    def _cfg_for(self, mk: dict):
        """thinker_* kwargs -> SamplingConfig; absent => greedy (temp=0)."""
        cfg = self._SamplingConfig(
            temperature=float(mk.get("thinker_temperature", 0.0)),
            top_k=int(mk.get("thinker_top_k", 0)),
            top_p=float(mk.get("thinker_top_p", 1.0)),
        )
        if (seed := mk.get("seed")) is not None:
            cfg.set_seed(int(seed))
        return cfg

    def register_request(self, request_id, model_kwargs: dict) -> None:
        cfg = self._cfg_for(model_kwargs or {})
        self._req_cfg[request_id] = cfg
        self.sampler.add_request(request_id)
        self.sampler.set_config(
            request_id, temperature=cfg.temperature, top_k=cfg.top_k,
            top_p=cfg.top_p, ignore_eos=False)
        if cfg.seed:
            self.sampler._sampling_config[request_id].set_seed(cfg.seed)

    def release_request(self, request_id) -> None:
        self._req_cfg.pop(request_id, None)
        self.sampler.remove_request(request_id)
        self._sbuf.unregister_request(request_id)
        if self._sampler_rid == request_id:
            self._sampler_rid = None
        self._generated.pop(request_id, None)
        self._rope_offset.pop(request_id, None)

    def _ensure_registered(self, rid) -> None:
        """Requests that arrived without a register (no model_kwargs on the
        wire) get the greedy default lazily."""
        if rid not in self._req_cfg:
            self.register_request(rid, {})

    def _sample_eager(self, rid, logits):
        """Per-request sampling for prefill + the eager decode fallback
        (mstar's eager Sampler; greedy rows take its argmax path)."""
        self._ensure_registered(rid)
        return int(self.sampler.sample([rid], logits.float())[0].item())

    def _stage_sampler(self, rid) -> None:
        """(Re)stage `rid`'s sampling params into the per-step SamplerBuffers
        the captured graph reads — runs OUTSIDE the graph, no recapture. The
        RNG offset resets per request so a fixed seed reproduces."""
        self._ensure_registered(rid)
        if self._sampler_rid == rid:
            return
        self._sbuf.register_request(rid, self._req_cfg[rid])
        self._sbuf.offset_buf.zero_()
        self._decode_sampler = self._sbuf.gather_for_request_ids(
            [rid], 1, gather_seen_tokens=False)
        self._sampler_rid = rid

    def _mrope(self, seq_pos: int):
        torch = self._torch
        pos3d = torch.full((3, 1), float(seq_pos), dtype=torch.float, device=self.device)
        return self._cos_sin(pos3d, self._inv_freq, MROPE_SECTION, target_dtype=torch.bfloat16)

    def _capture_decode(self, rid, pages, seq_pos: int) -> None:
        torch = self._torch
        handle = self._Handle(self._decode_attn)
        self._decode_attn.plan(pages, seq_pos, 1)
        self._stage_sampler(rid)
        sampler = self._decode_sampler

        def dec_fn() -> None:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden, layer_0, _ = self._tmodel(
                    input_embeds=self._embed(self._g_tok).to(torch.bfloat16),
                    cache_handle=handle, cos_sin_3d=(self._g_cos, self._g_sin),
                    mrope_section=MROPE_SECTION,
                )
                logits = self._lm_head(hidden[-1:])
                self._g_out_tok.copy_(sampler.sample([rid], logits).reshape(1))
                if self.audio_output:
                    self._g_layer0.copy_(layer_0)

        torch.cuda.synchronize()
        for _ in range(2):
            dec_fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            dec_fn()
        torch.cuda.synchronize()
        self._decode_graph = g

    def _decode_graphed(self, rid, tok_id: int, pages, seq_pos: int,
                        rope_pos: int | None = None):
        """Replay the captured Thinker decode for one token. Returns
        (sampled_token:int, layer_0_hidden or None). `rope_pos` is the MRoPE
        position (defaults to seq_pos); it differs from the KV `seq_pos` after a
        vision prefill, where RoPE advances by the image grid span. `rid`'s
        sampling params are re-staged into the graph's buffers on request
        switch (no recapture)."""
        torch = self._torch
        cos, sin = self._mrope(seq_pos if rope_pos is None else rope_pos)
        if self._decode_graph is None:
            self._g_cos.copy_(cos); self._g_sin.copy_(sin)
            self._capture_decode(rid, pages, seq_pos)
            # capture warmups advanced the RNG offset; start the request at 0
            self._sbuf.offset_buf.zero_()
        self._stage_sampler(rid)
        self._decode_attn.plan(pages, seq_pos, 1)   # growing KV, outside the graph
        self._g_tok.copy_(torch.as_tensor([tok_id], device=self.device))
        self._g_cos.copy_(cos); self._g_sin.copy_(sin)
        self._decode_graph.replay()
        tok = int(self._g_out_tok.item())   # .item() syncs; no explicit sync needed
        layer0 = self._g_layer0.clone() if self.audio_output else None
        return tok, layer0

    def _move_stray_cpu_tensors(self, mod) -> None:
        # The encoders' sinusoidal PE (and similar) are plain tensor attributes,
        # not registered buffers, so .to(device) skips them — move them here.
        torch = self._torch
        for m in mod.modules():
            for name, val in list(vars(m).items()):
                if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                    setattr(m, name, val.to(self.device))

    def _ensure_audio_encoder(self):
        if self._audio_encoder is None:
            enc = self._mstar.get_submodule("audio_encoder", device=str(self.device)).audio_encoder
            enc = enc.to(self.device)
            self._move_stray_cpu_tensors(enc)
            self._audio_encoder = enc
            self._audio_enc_dtype = next(enc.parameters()).dtype
        return self._audio_encoder

    def _ensure_vision_encoder(self):
        if self._vision_encoder is None:
            enc = self._mstar.get_submodule("vision_encoder", device=str(self.device)).vision_encoder
            enc = enc.to(self.device)
            self._move_stray_cpu_tensors(enc)
            self._vision_encoder = enc
            self._vision_enc_dtype = next(enc.parameters()).dtype
        return self._vision_encoder

    def _embed_with_audio(self, ids, named):
        """Embed the prompt with the audio encoder's embeds scattered onto the
        AUDIO_PAD positions, plus audio-aware 3D-MRoPE (temporal-only across the
        audio span). Mirrors verify_audio_input.py, inside the engine."""
        torch = self._torch
        enc = self._ensure_audio_encoder()
        feats = named["audio_features"][0].to(self.device, self._audio_enc_dtype)
        seqlens = named["audio_seqlens"][0].to(self.device).to(torch.long)
        with torch.no_grad():
            embeds = enc(feats, feature_lens=seqlens, return_dict=True).last_hidden_state
        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)
        inp = self._embed(ids).to(torch.bfloat16)
        mask = ids == AUDIO_PAD
        inp = inp.masked_scatter(mask.unsqueeze(-1), embeds.to(inp.dtype))
        a0 = int(mask.nonzero()[0])
        seq, n_aud = int(ids.shape[0]), int(mask.sum())
        pips = self.cfg.thinker.position_id_per_seconds
        pos = torch.empty(3, seq, dtype=torch.float, device=self.device)
        pos[:, :a0] = self._rope_text(a0, 0.0, self.device)
        pos[:, a0:a0 + n_aud] = self._rope_audio(n_aud, float(a0), self.device, pips)
        if a0 + n_aud < seq:
            pos[:, a0 + n_aud:] = self._rope_text(seq - a0 - n_aud, float(a0 + n_aud), self.device)
        return inp, pos, None, 0  # audio span occupies n_aud units -> no offset

    def _embed_with_vision(self, ids, named):
        """Embed the prompt with the vision encoder's (spatial-merged) embeds
        scattered onto the IMAGE_PAD positions, deepstack features placed at the
        same positions (added into the Thinker's first layers), and grid-based
        3D-MRoPE. Returns the RoPE-vs-KV offset since the grid span exceeds the
        token count. Mirrors verify_vision_input.py, inside the engine."""
        torch = self._torch
        enc = self._ensure_vision_encoder()
        pixel_values = named["pixel_values"][0].to(self.device, self._vision_enc_dtype)
        grid_thw = named["image_grid_thw"][0].to(self.device).to(torch.long)
        with torch.no_grad():
            enc_out = enc(pixel_values, grid_thw=grid_thw)
        if isinstance(enc_out, tuple):
            vision_embeds, deepstack = enc_out
        else:
            vision_embeds = enc_out.pooler_output      # spatial-merged, not raw patches
            deepstack = enc_out.deepstack_features
        if isinstance(deepstack, torch.Tensor):
            deepstack = [deepstack]
        inp = self._embed(ids).to(torch.bfloat16)
        mask = ids == IMAGE_PAD
        inp = inp.masked_scatter(mask.unsqueeze(-1), vision_embeds.to(inp.dtype))
        hidden_size = inp.shape[-1]
        ds_full = []
        for ds in deepstack:
            f = torch.zeros(ids.shape[0], hidden_size, dtype=inp.dtype, device=self.device)
            f[mask] = ds.to(inp.dtype)
            ds_full.append(f)
        v0 = int(mask.nonzero()[0])
        seq, n_vis = int(ids.shape[0]), int(mask.sum())
        pips = self.cfg.thinker.position_id_per_seconds
        sms = self.cfg.vision.spatial_merge_size
        pos = torch.empty(3, seq, dtype=torch.float, device=self.device)
        pos[:, :v0] = self._rope_text(v0, 0.0, self.device)
        pos[:, v0:v0 + n_vis] = self._rope_vision(
            grid_thw, float(v0), position_id_per_seconds=pips, device=self.device,
            spatial_merge_size=sms, seconds_per_grid=None)
        nxt = float(pos[:, v0:v0 + n_vis].max().item()) + 1   # grid span > token count
        if v0 + n_vis < seq:
            pos[:, v0 + n_vis:] = self._rope_text(seq - v0 - n_vis, nxt, self.device)
        rope_pos = int(pos.max().item()) + 1
        return inp, pos, ds_full, rope_pos - seq   # decode continues from grid span

    def execute(self, node_name, walk, inputs, kv=None):
        torch = self._torch
        out: dict[int, dict[str, list]] = {}
        for rid, named in inputs.items():
            ids = named["text_inputs"][0].to(self.device).reshape(-1)
            view = kv[rid]
            seq_pos, n = view["seq_pos"], view["append_len"]
            # per-token decode: the CUDA-graph path (embed + Thinker forward
            # incl. NCCL all-reduces + lm_head + argmax, captured over static
            # buffers; MRoPE cos/sin copied in per token).
            if walk == "thinker_decode" and self.cuda_graph:
                rp = seq_pos + self._rope_offset.get(rid, 0)
                tok, layer0 = self._decode_graphed(rid, int(ids[0].item()),
                                                   view["pages"], seq_pos, rp)
                o = {"new_token": [torch.tensor([tok], dtype=torch.long)],
                     "text_inputs": [torch.tensor([tok], dtype=torch.long)]}
                if self.audio_output:
                    o["thinker_states"] = [layer0]
                    o["thinker_mask"] = [torch.zeros(1, dtype=torch.bool, device=self.device)]
                cnt = self._generated.get(rid, 0) + 1
                self._generated[rid] = cnt
                if tok == self._eos or cnt >= self.max_output_tokens:
                    self._pending_stops.append((rid, "thinker_decode_loop"))
                out[rid] = o
                continue
            deepstack = None
            if walk == "prefill_audio":
                input_embeds, pos3d, deepstack, off = self._embed_with_audio(ids, named)
                self._rope_offset[rid] = off
            elif walk == "prefill_vision":
                input_embeds, pos3d, deepstack, off = self._embed_with_vision(ids, named)
                self._rope_offset[rid] = off
            elif n > 1:  # text prefill
                pos3d = self._rope_index(n, seq_pos, self.device)
                input_embeds = self._embed(ids).to(torch.bfloat16)
                self._rope_offset[rid] = 0
            else:  # eager decode fallback (cuda_graph off); RoPE may be offset from KV
                rp = seq_pos + self._rope_offset.get(rid, 0)
                pos3d = torch.full((3, 1), float(rp), dtype=torch.float, device=self.device)
                input_embeds = self._embed(ids).to(torch.bfloat16)
            cos, sin = self._cos_sin(pos3d, self._inv_freq, MROPE_SECTION, target_dtype=torch.bfloat16)
            self._attn.plan(view["pages"], seq_pos, n)
            handle = self._Handle(self._attn)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden, layer_0, _ = self._tmodel(
                    input_embeds=input_embeds,
                    cache_handle=handle, cos_sin_3d=(cos, sin), mrope_section=MROPE_SECTION,
                    deepstack_visual_embeds=deepstack,
                )
                logits = self._lm_head(hidden[-1:])
            # per-request sampling (greedy default = argmax path in the sampler)
            tok = self._sample_eager(rid, logits)
            o = {"new_token": [torch.tensor([tok], dtype=torch.long)]}
            if self.audio_output:
                # Stream the selected Thinker hidden + mask to the Talker. For a
                # text-only prompt: multimodal mask all-False, so the selected
                # hidden is layer_0_embed; prefill drops system/prior-assistant
                # tokens via _get_talker_text_mask (mstar postprocess L1140-1163).
                if walk.startswith("prefill"):
                    incl = self._thinker_sub._get_talker_text_mask(ids)
                    o["thinker_states"] = [layer_0[incl].contiguous()]
                    o["thinker_mask"] = [torch.zeros(int(incl.sum()), dtype=torch.bool,
                                                     device=self.device)]
                else:  # thinker_decode: one token's layer_0_embed
                    o["thinker_states"] = [layer_0.contiguous()]
                    o["thinker_mask"] = [torch.zeros(1, dtype=torch.bool, device=self.device)]
            if walk == "thinker_decode":
                o["text_inputs"] = [torch.tensor([tok], dtype=torch.long)]  # loop-back
                cnt = self._generated.get(rid, 0) + 1
                self._generated[rid] = cnt
                if tok == self._eos or cnt >= self.max_output_tokens:
                    self._pending_stops.append((rid, "thinker_decode_loop"))
            out[rid] = o
        return out

    def loops_to_finish(self):
        stops = self._pending_stops
        self._pending_stops = []
        return stops


# =========================================================================
# Stage 2, slice 2: the audio path — Talker (backbone + code-predictor depth
# loop) + Code2Wav vocoder. Both run TP=1 on a single GPU (small models; only
# the Thinker needed sharding). The Talker backbone uses the same handle
# contract as orpheus (apply_rope theta=1e6 + run_attention), so the
# FlashInferCacheHandle drives it unchanged; the code predictor is a
# self-contained dense-KV fp-loop that never touches the paged handle. This
# reuses mstar's OWN `_forward_decode_like` / `_forward_prefill` (they take a
# cache_handle + injected_sampler, no engine coupling) for maximal fidelity —
# only the orchestration around them lives here. Streaming this onto rank 2
# behind the conductor (cross-partition thinker_states / codec_tokens) is the
# next slice; here it is drivable standalone (verify_talker_code2wav.py).
# =========================================================================


class Qwen3OmniAudioPolicy(ModelPolicy):
    """Conductor-side control plane for the FULL audio pipeline: Thinker text
    AR (prefill_text -> thinker_decode) streaming thinker_states to the Talker
    (talker_prefill -> talker_last_prefill -> talker_decode) streaming
    codec_tokens to Code2Wav. Weightless. `talker_trigger` is dropped — in this
    runtime the streamed-chunk arrival already gates the Talker walks. The
    Thinker runs TP (ranks 0,1); Talker + Code2Wav run on rank 2."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 max_output_tokens: int = 2048, voice: str = "chelsie") -> None:
        from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

        self._mstar = Qwen3OmniModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.tokenizer = self._mstar.tokenizer
        self.max_output_tokens = max_output_tokens
        self.voice = voice

    # -- graph -------------------------------------------------------------

    def _thinker_out(self):
        return [
            emit("new_token", modality="text", persist=True),
            stream_edge("Talker", "thinker_states", "Talker"),
            stream_edge("Talker", "thinker_mask", "Talker"),
        ]

    def walks(self):
        return {
            "prefill_text": node("Thinker", ["text_inputs"], self._thinker_out()),
            # Media prefills: encoder embeds scattered onto the placeholders in
            # the Thinker engine; same thinker_states/mask stream to the Talker.
            "prefill_audio": node(
                "Thinker", ["text_inputs", "audio_features", "audio_seqlens"],
                self._thinker_out()),
            "prefill_vision": node(
                "Thinker", ["text_inputs", "pixel_values", "image_grid_thw"],
                self._thinker_out()),
            "thinker_decode": loop(
                "thinker_decode_loop",
                node("Thinker", ["text_inputs"],
                     [edge("Thinker", "text_inputs")] + self._thinker_out()),
                max_iters=self.max_output_tokens,
            ),
            "talker_prefill": node("Talker", ["thinker_states", "thinker_mask"], []),
            "talker_last_prefill": node(
                "Talker", ["thinker_states", "thinker_mask"],
                [edge(EMPTY_DESTINATION, "talker_input_embeds", persist=True),
                 stream_edge("Code2Wav", "codec_tokens", "Code2Wav")],
            ),
            "talker_decode": loop(
                "talker_decode_loop",
                node("Talker", ["thinker_states", "thinker_mask", "talker_input_embeds"],
                     [edge("Talker", "talker_input_embeds"),
                      stream_edge("Code2Wav", "codec_tokens", "Code2Wav")]),
                max_iters=self.max_output_tokens,
            ),
            "code2wav_chunk": node("Code2Wav", ["codec_tokens"],
                                   [emit("audio_chunk", modality="audio")]),
        }

    def partitions(self):
        return (
            [
                partition("Thinker", ["prefill_text", "prefill_audio", "prefill_vision",
                                      "thinker_decode"]),
                partition("Talker", ["talker_prefill", "talker_last_prefill", "talker_decode"]),
                partition("Code2Wav", ["code2wav_chunk"]),
            ],
            [
                connection("Thinker", "Talker", "thinker_states",
                           fixed_chunk(1, continue_after_done=True)),
                connection("Thinker", "Talker", "thinker_mask",
                           fixed_chunk(1, continue_after_done=True)),
                connection("Talker", "Code2Wav", "codec_tokens",
                           left_context(CODEC_CHUNK_FRAMES, CODEC_LEFT_CONTEXT_FRAMES)),
            ],
        )

    def kv_config(self):
        return (
            [(THINKER_KV, THINKER_TEXT_PAGES, PAGE_SIZE), (TALKER_KV, TALKER_PAGES, PAGE_SIZE)],
            {"Thinker": THINKER_KV, "Talker": TALKER_KV},
        )

    # -- ingestion + continuation -----------------------------------------

    def _talker_text_mask(self, ids):
        """Replicate ThinkerSubmodule._get_talker_text_mask (config-only, no
        weights): drop system / prior-assistant segments; keep user + the
        trailing assistant-generation header. Gives num_kept = the Talker's
        prefill KV append."""
        import torch

        c = self.cfg
        starts = (ids == c.im_start_token_id).nonzero(as_tuple=True)[0]
        mask = torch.ones(ids.shape, dtype=torch.bool)
        for i in range(len(starts) - 1):
            role = ids[starts[i] + 1]
            if role in (c.system_token_id, c.assistant_token_id):
                mask[starts[i]: starts[i + 1]] = False
        return mask

    def _tokenize(self, prompt):
        text = self._mstar._processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0]

    def initial_walks(self, request):
        # Text or media prefill (feature extraction in the conductor); the
        # placeholder-expanded ids drive the Talker's text-conditioning mask too.
        walk, seeds, ids = _thinker_prefill_seeds(self._mstar, request)
        num_kept = int(self._talker_text_mask(ids).sum())
        # seed all three partitions (mstar kicks off every partition at ingest);
        # Talker/Code2Wav have no seed inputs — they wait on streamed chunks.
        return [
            (walk, seeds, {THINKER_KV: int(ids.shape[0])}),
            ("talker_prefill", [], {TALKER_KV: num_kept}),
            ("code2wav_chunk", []),
        ]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        if partition == "Thinker":
            if walk.startswith("prefill"):
                return ("thinker_decode", [("Thinker", "text_inputs", persist["new_token"])],
                        {THINKER_KV: 1})
            return None  # thinker_decode finished
        if partition == "Talker":
            if walk == "talker_prefill":   # text-only: one prefill -> last_prefill
                return ("talker_last_prefill", [], {TALKER_KV: 6})
            if walk == "talker_last_prefill":
                return ("talker_decode",
                        [("Talker", "talker_input_embeds", persist["talker_input_embeds"])],
                        {TALKER_KV: 1})
            return None  # talker_decode finished (codec_eos / max)
        # Code2Wav: re-arm until the codec stream is drained.
        return None if stream_done else ("code2wav_chunk", [])

    def postprocess(self, name, modality, tensors):
        if modality == "audio":
            return tensors[0].detach().cpu()      # int16 PCM, 24 kHz
        return int(tensors[0].reshape(-1)[0].item())  # text token id


class Qwen3OmniAudioEngine(ModelEngine):
    """Talker + Code2Wav on one GPU (rank 2). Handles both the "Talker" and
    "Code2Wav" nodes via `execute`; the standalone step methods (used by
    verify_talker_code2wav.py) mirror the same walks. Each frame is 1 semantic
    code (Talker codec_head) + 15 residual codes (code predictor) =
    num_code_groups. Consumes streamed thinker_states from the Thinker and
    streams codec_tokens to Code2Wav."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                 voice: str = "chelsie", max_output_tokens: int = 2048,
                 cuda_graph: bool = True) -> None:
        import torch

        from mstar.distributed.communication import TPCommGroup
        from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
        from mstar.utils.sampling import (
            Sampler,
            SamplerBuffers,
            SamplingConfig,
            SeenTokenMask,
        )

        from ..fi import FlashInferAttention, FlashInferCacheHandle, FlashInferPagedKV

        self._torch = torch
        self._Handle = FlashInferCacheHandle
        self.device = torch.device(device)
        # Make this the current device so Triton/CUDA kernels (e.g. the sampler)
        # launch on the same device as their operands — otherwise a kernel
        # launched on the default cuda:0 over cuda:2 tensors faults. Only when
        # an explicit index is given ("cuda" with no index -> leave current).
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        self._mstar = Qwen3OmniModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.sr = 24000

        self.talker = self._mstar.get_submodule(
            "Talker", device=str(self.device), tp_group=TPCommGroup.trivial(),
            autocast_dtype=torch.bfloat16,
        )
        self.code2wav = self._mstar.get_submodule("Code2Wav", device=str(self.device))
        # The weight loader moves params but not the non-persistent `code_offset`
        # buffer (created on CPU in __init__); move the whole vocoder to device.
        self.code2wav.code2wav.to(self.device)
        self.num_codes = self.talker.num_codes                       # 16
        self._csum_dtype = self.talker.talker_code_emb.weight.dtype

        tt = self.cfg.talker_text
        head_dim = getattr(tt, "head_dim", tt.hidden_size // tt.num_attention_heads)
        self._cache = FlashInferPagedKV(
            tt.num_hidden_layers, THINKER_TEXT_PAGES, PAGE_SIZE,
            tt.num_key_value_heads, head_dim, self.device, torch.bfloat16,
        )
        self._attn = FlashInferAttention(
            self._cache, tt.num_attention_heads, head_dim, self.device,
            max_new_tokens=512, cudagraph=False, causal=True, dtype=torch.bfloat16,
        )
        self._pages = list(range(THINKER_TEXT_PAGES))
        self.sampler = Sampler(device=self.device)
        self.voice = voice
        self.max_output_tokens = max_output_tokens
        self._registered: set[str] = set()
        # per-request sampling + voice (mstar's model_kwargs-at-ingest):
        # talker_temperature / talker_top_k / talker_top_p /
        # talker_repetition_penalty / voice / seed. Defaults match the previous
        # hard-coded values (temp=1.0, top_p=1.0, rep=1.1, ctor voice).
        self._req_cfg: dict[Any, Any] = {}
        self._req_voice: dict[Any, str] = {}
        # per-request streaming state (for execute)
        self._eos_sent: set[str] = set()      # tts_eos injected once after Thinker EOS
        self._first_chunk: set[str] = set()    # Code2Wav left-context trim gate
        self._codec_eos_hit: set[int] = set()
        self._pending_stops: list[tuple[int, str]] = []

        # --- CUDA-graph decode capture (RTF lever) -----------------------
        # Capture the WHOLE per-frame decode (mstar's _forward_decode_like:
        # backbone + codec_head + layer-0 sample + the unrolled 16-step depth
        # loop) as one graph, reusing mstar's graph-safe code verbatim. My fi
        # attention runs in cudagraph mode (plan() outside), and an in-graph
        # CudaGraphableSampler replaces the eager Sampler. mstar's own runner
        # can't be reused here (it's bound to mstar's paged cache manager, which
        # the Rust runtime + fi handle replace) — so I own only the torch.cuda
        # .graph harness; every op inside is mstar's.
        self.cuda_graph = cuda_graph
        self._SamplingConfig = SamplingConfig
        self._SeenTokenMask = SeenTokenMask
        self._g_ie = torch.zeros(1, tt.hidden_size, dtype=torch.bfloat16, device=self.device)
        self._g_all_codes = torch.zeros(1, self.num_codes, dtype=torch.long, device=self.device)
        self._g_csum = torch.zeros(1, self.cfg.talker_hidden_size,
                                   dtype=self._csum_dtype, device=self.device)
        self._g_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._decode_attn = FlashInferAttention(
            self._cache, tt.num_attention_heads, head_dim, self.device,
            max_new_tokens=1, cudagraph=cuda_graph, causal=True, dtype=torch.bfloat16,
        )
        self._sbuf = SamplerBuffers.allocate(
            1, self.device, tp_group=TPCommGroup.trivial(), vocab_size=tt.vocab_size,
        )
        self._decode_graph = None
        self._decode_sampler = None
        self._sampler_rid = None   # whose config/seen-mask is staged for the graph

    def _handle(self, pages, seq_pos: int, n: int):
        self._attn.plan(pages, seq_pos, n)
        return self._Handle(self._attn)

    def _talker_cfg(self, mk: dict):
        """talker_* kwargs -> SamplingConfig; defaults = the previous
        hard-coded values (temp=1.0, top_k=0, top_p=1.0, rep_penalty=1.1)."""
        cfg = self._SamplingConfig(
            vocab_size=self.cfg.talker_text.vocab_size,
            temperature=float(mk.get("talker_temperature", 1.0)),
            top_k=int(mk.get("talker_top_k", 0)),
            top_p=float(mk.get("talker_top_p", 1.0)),
            repetition_penalty=float(mk.get("talker_repetition_penalty", 1.1)),
        )
        if (seed := mk.get("seed")) is not None:
            cfg.set_seed(int(seed))
        return cfg

    def register_request(self, request_id, model_kwargs: dict) -> None:
        # execute keys its per-request state by str(rid) (`srid`) — normalize.
        rid = str(request_id)
        mk = model_kwargs or {}
        self._req_cfg[rid] = self._talker_cfg(mk)
        if voice := mk.get("voice"):
            self._req_voice[rid] = str(voice)

    def release_request(self, request_id) -> None:
        rid = str(request_id)
        self._req_cfg.pop(rid, None)
        self._req_voice.pop(rid, None)
        self._sbuf.unregister_request(rid)
        if self._sampler_rid == rid:
            self._sampler_rid = None
        if rid in self._registered:
            self._registered.discard(rid)
            self.sampler.remove_request(rid)
        self._eos_sent.discard(rid)
        self._first_chunk.discard(rid)
        self._codec_eos_hit.discard(request_id)  # keyed by the int rid

    def _cfg_for(self, rid):
        cfg = self._req_cfg.get(rid)
        return cfg if cfg is not None else self._talker_cfg({})

    def _voice_for(self, rid) -> str:
        return self._req_voice.get(rid, self.voice)

    def _register(self, rid: str) -> None:
        if rid not in self._registered:
            cfg = self._cfg_for(rid)
            self.sampler.add_request(rid)
            self.sampler.set_config(
                rid, vocab_size=cfg.vocab_size, temperature=cfg.temperature,
                top_k=cfg.top_k, top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty, ignore_eos=False,
            )
            if cfg.seed:
                self.sampler._sampling_config[rid].set_seed(cfg.seed)
            self._registered.add(rid)

    def _depth(self, rid: str, handle, input_embeds):
        """Run the Talker backbone over `input_embeds` then the code-predictor
        depth loop (mstar's _forward_decode_like). Returns its output dict:
        codec_tokens (1,16), talker_input_embeds (1,1024), new_token (layer0)."""
        torch = self._torch
        bs = 1
        return self.talker._forward_decode_like(
            request_ids=[rid],
            cache_handle=handle,
            injected_sampler=self.sampler,
            suppress_mask=self.talker._get_suppress_mask(),
            all_codes=torch.zeros(bs, self.num_codes, dtype=torch.long, device=self.device),
            codec_emb_sum=torch.zeros(bs, self.cfg.talker_hidden_size,
                                      dtype=self._csum_dtype, device=self.device),
            pos_buf=torch.zeros(bs, 1, dtype=torch.long, device=self.device),
            is_batched_decode=False,
            input_embeds=input_embeds,
        )

    def _capture_decode(self, rid: str, pages, seq_pos: int) -> None:
        """Capture the whole per-frame decode (mstar's _forward_decode_like:
        backbone + codec_head + layer-0 sample + 16-step depth loop) as ONE
        CUDA graph over static buffers, with the fi wrapper in cudagraph mode
        (planned outside) and an in-graph CudaGraphableSampler. All compute is
        mstar's code; only this harness is mine."""
        torch = self._torch
        self._stage_decode_sampler(rid)
        sampler = self._decode_sampler
        handle = self._Handle(self._decode_attn)
        self._decode_attn.plan(pages, seq_pos, 1)

        def dec_fn() -> None:
            self._g_pos.zero_()
            self._g_csum.zero_()               # _forward_decode_like accumulates via add_
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                self.talker._forward_decode_like(
                    request_ids=[rid], cache_handle=handle, injected_sampler=sampler,
                    suppress_mask=self.talker._get_suppress_mask(),
                    all_codes=self._g_all_codes, codec_emb_sum=self._g_csum,
                    pos_buf=self._g_pos, is_batched_decode=False, input_embeds=self._g_ie,
                )

        saved_off = self._sbuf.offset_buf[:1].clone()
        saved_seen = self._sbuf.seen_tokens.buf[:1].clone()
        torch.cuda.synchronize()
        for _ in range(2):  # warmup / autotune (writes garbage KV at this slot)
            dec_fn()
        self._sbuf.offset_buf[:1].copy_(saved_off)
        self._sbuf.seen_tokens.buf[:1].copy_(saved_seen)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            dec_fn()
        torch.cuda.synchronize()
        self._decode_graph = g

    def _stage_decode_sampler(self, rid) -> None:
        """(Re)stage `rid`'s sampling params + a FRESH seen-token mask into the
        per-step SamplerBuffers the captured graph reads — outside the graph, no
        recapture. Guards on request switch: without the fresh mask, request
        N+1 would inherit request N's seen tokens and the repetition penalty
        would bleed across requests. Resets the RNG offset so a fixed seed
        reproduces per request."""
        if self._sampler_rid == rid:
            return
        self._sbuf.register_request(rid, self._cfg_for(rid))
        seen = self._SeenTokenMask.new(rid, self.cfg.talker_text.vocab_size, self.device)
        self._sbuf.stage_seen_token_masks([rid], [seen])
        self._sbuf.offset_buf.zero_()
        self._decode_sampler = self._sbuf.gather_for_request_ids(
            [rid], 1, gather_seen_tokens=True)
        self._sampler_rid = rid

    def _decode_graphed(self, rid: str, input_embeds, pages, seq_pos: int):
        """Replay the captured decode graph for one frame. Re-plans the fi
        wrapper outside the graph (growing KV), copies the input embed in, and
        reads the static output buffers. Per-request sampling params re-stage
        on request switch (the graph reads the buffers; no recapture)."""
        torch = self._torch
        if self._decode_graph is None:
            self._capture_decode(rid, pages, seq_pos)
            # capture warmups advanced RNG offset + seen mask; restage fresh
            self._sampler_rid = None
        self._stage_decode_sampler(rid)
        self._decode_attn.plan(pages, seq_pos, 1)   # growing KV, outside the graph
        self._g_ie.copy_(input_embeds)
        self._decode_graph.replay()
        # clones are stream-ordered after replay; the downstream .item()/SHM
        # staging syncs — no explicit per-step device sync needed.
        return (self._g_all_codes.clone(), self._g_csum.clone(),
                self._g_all_codes[:, 0].clone())

    def talker_prefill(self, thinker_states, mm_mask, seq_pos: int) -> int:
        """KV-fill from a chunk of streamed Thinker hiddens; returns new seq_pos."""
        torch = self._torch
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = self.talker._get_talker_embeds(thinker_states.to(self.device), mm_mask.to(self.device))
            self.talker._forward_prefill(self._handle(self._pages, seq_pos, emb.shape[0]), emb)
        return seq_pos + emb.shape[0]

    def last_prefill(self, rid: str, last_thinker_hidden, voice: str, seq_pos: int):
        """The 6-token assistant prefix -> first codec frame. Returns
        (codec_tokens[16], talker_input_embeds[1024], layer0, new_seq_pos)."""
        torch = self._torch
        self._register(rid)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            codec6 = self.talker._get_last_prefill_codec_hidden(voice)
            text6 = self.talker._get_last_prefill_talker_hidden(last_thinker_hidden.to(self.device))
            out = self._depth(rid, self._handle(self._pages, seq_pos, 6), codec6 + text6)
        return out["codec_tokens"][0], out["talker_input_embeds"][0], out["new_token"][0], seq_pos + 6

    def decode_step(self, rid: str, talker_input_embeds, text_hidden, seq_pos: int):
        """One decode frame: self-fed codec ⊕ text conditioning -> depth loop.
        Returns (codec_tokens[1,16], talker_input_embeds[1,1024], layer0[1], new_seq_pos)."""
        torch = self._torch
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = (talker_input_embeds + text_hidden).to(torch.bfloat16)
        if self.cuda_graph:
            codes, csum, layer0 = self._decode_graphed(rid, emb, self._pages, seq_pos)
            return codes, csum, layer0, seq_pos + 1
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self._depth(rid, self._handle(self._pages, seq_pos, 1), emb)
        return out["codec_tokens"][0], out["talker_input_embeds"][0], out["new_token"][0], seq_pos + 1

    def tts_pad(self):
        return self.talker._tts_pad_embed_cached.unsqueeze(0)

    def tts_eos(self):
        return self.talker._tts_eos_embed_cached.unsqueeze(0)

    def code2wav_chunk(self, codec_frames, first_chunk: bool):
        """codec_frames: (F, 16) codec tokens -> 24 kHz int16 PCM. Pads to the
        constant full_seqlen (left-context + chunk) so the vocoder graph shape
        is fixed, then trims the left-context samples off non-first chunks."""
        torch = self._torch
        sub = self.code2wav
        full = sub.full_seqlen
        up = sub.total_upsample
        ctx = self.cfg.code2wav.codec_left_context_frames
        F = codec_frames.shape[0]
        codes = codec_frames[:, : self.cfg.code2wav.num_quantizers].to(self.device)
        if F < full:  # pad up to the fixed vocoder length (repeat last frame)
            codes = torch.cat([codes, codes[-1:].expand(full - F, -1)], dim=0)
        codes_t = codes.t().contiguous()                       # (Q, full)
        pos = torch.arange(full, device=self.device)
        with torch.no_grad():
            wav = sub.code2wav(codes_t.unsqueeze(0), pos.unsqueeze(0))  # (1,1,full*up)
        wav = wav.reshape(-1)
        trim = 0 if first_chunk else ctx * up
        pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16)
        return pcm[trim: F * up]

    # -- conductor-driven execution ---------------------------------------

    def _check_eos(self, rid: int, layer0) -> None:
        if int(layer0.reshape(-1)[0].item()) == self.cfg.talker.codec_eos_token_id:
            self._codec_eos_hit.add(rid)
            self._pending_stops.append((rid, "talker_decode_loop"))

    def execute(self, node_name, walk, inputs, kv=None):
        torch = self._torch
        out: dict[int, dict[str, list]] = {}
        for rid, named in inputs.items():
            srid = str(rid)
            if node_name == "Talker":
                view = kv[rid]
                if walk == "talker_decode":
                    # per-frame hot loop — the CUDA-graph path (mstar's
                    # _forward_decode_like captured over static buffers).
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        states = named.get("thinker_states", [])
                        if states and states[0].numel() > 0:
                            text_h = self.talker.model.text_projection(states[0].to(self.device))
                        elif srid not in self._eos_sent:  # Thinker EOS -> tts_eos once
                            text_h = self.tts_eos()
                            self._eos_sent.add(srid)
                        else:
                            text_h = self.tts_pad()
                        emb = (named["talker_input_embeds"][0].to(self.device) + text_h).to(torch.bfloat16)
                    if self.cuda_graph:
                        codes, csum, layer0 = self._decode_graphed(
                            srid, emb, view["pages"], view["seq_pos"])
                    else:
                        o = self._depth(srid, self._handle(view["pages"], view["seq_pos"], 1), emb)
                        codes, csum, layer0 = (o["codec_tokens"][0], o["talker_input_embeds"][0],
                                               o["new_token"][0])
                    out[rid] = {"codec_tokens": [codes], "talker_input_embeds": [csum]}
                    self._check_eos(rid, layer0)
                    continue
                # prefill walks (one-off, eager)
                handle = self._handle(view["pages"], view["seq_pos"], view["append_len"])
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if walk == "talker_prefill":
                        states = named["thinker_states"][0].to(self.device)
                        mask = named["thinker_mask"][0].to(self.device)
                        self.talker._forward_prefill(handle, self.talker._get_talker_embeds(states, mask))
                        out[rid] = {}  # KV-fill only, no outputs
                        continue
                    # talker_last_prefill
                    self._register(srid)
                    states = named["thinker_states"][0].to(self.device)
                    codec6 = self.talker._get_last_prefill_codec_hidden(self._voice_for(srid))
                    text6 = self.talker._get_last_prefill_talker_hidden(states)
                    o = self._depth(srid, handle, codec6 + text6)
                out[rid] = {
                    "codec_tokens": [o["codec_tokens"][0]],
                    "talker_input_embeds": [o["talker_input_embeds"][0]],
                }
                self._check_eos(rid, o["new_token"][0])
            elif node_name == "Code2Wav":
                frames = named["codec_tokens"]           # window of (1,16) frames
                codec = torch.cat([f.reshape(1, -1) for f in frames], dim=0)
                first = rid not in self._first_chunk
                self._first_chunk.add(rid)
                out[rid] = {"audio_chunk": [self.code2wav_chunk(codec, first_chunk=first)]}
            else:
                raise ValueError(f"unknown node: {node_name}")
        return out

    def loops_to_finish(self):
        stops = self._pending_stops
        self._pending_stops = []
        return stops
