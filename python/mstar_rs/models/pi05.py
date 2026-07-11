"""Pi0.5 on mstar-rs — the first KV-cache model (T4 tier).

Walks (ported from ``mstar/model/pi05/pi05_model.py``):

- ``prefill``:    image_inputs -> [vit_encoder] -> img_emb;
                  img_emb + text_inputs -> [LLM]  (writes prefix KV, no outputs)
- ``action_gen``: Loop x num_flow_steps over [LLM]: (noisy_actions,
                  timestep_index) -> euler flow step -> loop-back; final
                  noisy_actions -> EMIT_TO_CLIENT (action).

The Rust runtime owns page tables and sequence positions (prefill declares
``kv_appends = prefix_len``; action_gen declares append 0 + scratch pages
for the transient suffix). The data plane matches mstar's engine exactly —
bf16 autocast, FlashInfer paged attention, and a CUDA-graph-captured euler
step replayed ``num_flow_steps`` times (`mstar_rs.fi`) — and drives
**mstar's own pi05 nn modules unmodified**: `Pi05SiglipEncoder` (fp32, as in
mstar), `Pi05PaliGemmaExpert`, `Pi05ActionExpert`, weight remapping, even
`_euler_step`, through the same cache-manager interface mstar's engine
implements.
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
from ..graph import edge, emit, loop, node, sequential
from ..model import ModelEngine, ModelPolicy, NextWalk

DEFAULT_MODEL_ID = "lerobot/pi05_base"
KV_LABEL = "main"
PAGE_SIZE = 128  # mstar kv_store default
NUM_PAGES = 64   # 8192 tokens — plenty for prefix (768 img + <=220 text) + suffix
MAX_PREFILL_TOKENS = 2048
MAX_ACTION_BATCH = 8       # largest batch we'll capture an action_gen graph for
MAX_PAGES_PER_REQ = 32     # ceil((prefix + scratch) / PAGE_SIZE) upper bound
MAX_PREFILL_BATCH_TOKENS = MAX_ACTION_BATCH * MAX_PREFILL_TOKENS  # ragged prefill cap


class Pi05Policy(ModelPolicy):
    """Control plane (conductor): graph, KV declaration, request ingestion,
    the flow-loop policy, postprocess. Constructs mstar's `Pi05Model` for its
    config + tokenizer ONLY — no `get_submodule`, so no GPU weights load here
    (weights live in `Pi05Engine`, mirroring mstar's lazy submodule build)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> None:
        from mstar.model.pi05.pi05_model import Pi05Model

        self.device = torch.device(device)
        # config + tokenizer only (Pi05Model defers weight loading to
        # get_submodule, which the engine — not this policy — calls).
        self._mstar = Pi05Model(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self._seeds: dict[int, int] = {}  # request_id -> action-noise seed

    # -- graph + KV declaration -------------------------------------------

    def walks(self) -> dict[str, Any]:
        return {
            "prefill": sequential(
                node("vit_encoder", ["image_inputs"], [edge("LLM", "img_emb")]),
                node("LLM", ["img_emb", "text_inputs"], []),
            ),
            "action_gen": loop(
                "flow",
                node(
                    "LLM",
                    ["noisy_actions", "timestep_index"],
                    [edge("LLM", "noisy_actions"), edge("LLM", "timestep_index")],
                ),
                max_iters=self.cfg.num_flow_steps,
                outputs=[emit("noisy_actions", modality="action", persist=True)],
            ),
        }

    def kv_config(self):
        return [(KV_LABEL, NUM_PAGES, PAGE_SIZE)], {"LLM": KV_LABEL}

    def unbatchable(self):
        # action_gen now batches: it replays a per-batch-size CUDA graph over a
        # BatchedFlashInferAttention (verified bit-exact vs per-request), so the
        # scheduler may hand it up to MAX_ACTION_BATCH requests. Nothing to cap.
        return []

    # -- request ingestion (mirrors Pi05Model.process_prompt) --------------

    def tokenize(self, prompt: str, robot_state) -> torch.Tensor:
        from mstar.model.pi05.components.flow_matching import discretize_state

        cleaned = (prompt or "").strip().replace("_", " ").replace("\n", " ")
        if robot_state is not None:
            if not isinstance(robot_state, torch.Tensor):
                robot_state = torch.tensor(robot_state, dtype=torch.float32)
            bins = discretize_state(
                robot_state.to(torch.float32), num_bins=self.cfg.state_token_bins
            ).tolist()
            state_str = " ".join(str(b) for b in bins)
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        else:
            full_prompt = cleaned
        return self._mstar.tokenizer.encode_prompt(full_prompt)

    def initial_inputs(self, request: dict[str, Any]) -> NextWalk:
        images = request["images"].to(self.device)  # (num_cameras, 3, H, W)
        text_ids = self.tokenize(request.get("prompt", ""), request.get("robot_state"))
        text_ids = text_ids.to(self.device)
        prefix_len = self.cfg.num_cameras * self.cfg.tokens_per_image + text_ids.shape[0]
        self._seeds[request["request_id"]] = request.get("seed", 0)
        return (
            "prefill",
            [
                ("vit_encoder", "image_inputs", [images]),
                ("LLM", "text_inputs", [text_ids]),
            ],
            {KV_LABEL: prefix_len},
        )

    # -- policy: prefill -> action_gen -> done ------------------------------

    def next_forward(
        self, request_id, partition, walk, fwd_index, persist, stream_done
    ) -> NextWalk | None:
        if walk == "prefill":
            seed = self._seeds.pop(request_id)  # consumed once; don't leak
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noisy = torch.randn(
                self.cfg.action_horizon,
                self.cfg.action_dim,
                device=self.device,
                generator=generator,
            )
            ts = torch.zeros(1, device=self.device, dtype=torch.long)
            return (
                "action_gen",
                [
                    ("LLM", "noisy_actions", [noisy]),
                    ("LLM", "timestep_index", [ts]),
                ],
                {KV_LABEL: 0},
                # Scratch pages for the transient suffix K/V each euler step
                # (mstar plans prefix+suffix pages with write_store=False).
                {KV_LABEL: self.cfg.action_horizon},
            )
        return None  # action_gen done -> request done

    def postprocess(self, name, modality, tensors):
        assert modality == "action"
        return tensors[0].detach().to(torch.float32).cpu()


class Pi05Engine(Pi05Policy, ModelEngine):
    """Data plane (worker): loads the GPU weights (via get_submodule) + KV
    cache + CUDA-graph state, and runs `execute`. Being a `Pi05Policy` too, it
    is also a full `Model` for the single-process Driver."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> None:
        super().__init__(model_id, device)  # cheap config/tokenizer/policy state
        dev = str(device)
        self.vit = self._mstar.get_submodule("vit_encoder", device=dev)
        self.llm = self._mstar.get_submodule("LLM", device=dev)
        self.kv_cache = FlashInferPagedKV(
            num_layers=self.cfg.num_layers,
            num_pages=NUM_PAGES,
            page_size=PAGE_SIZE,
            num_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._prefill_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_qo_heads, self.cfg.head_dim,
            self.device, max_new_tokens=MAX_PREFILL_TOKENS, cudagraph=False,
        )
        # Ragged batched prefill attention (eager), cached per batch size N.
        self._prefill_battn: dict[int, Any] = {}
        # Batched CUDA-graph euler state (request-major rows), one graph per N.
        H = self.cfg.action_horizon
        self._g_noisy = torch.zeros(
            MAX_ACTION_BATCH * H, self.cfg.action_dim, device=self.device
        )
        self._g_ts = torch.zeros(MAX_ACTION_BATCH, device=self.device, dtype=torch.long)
        self._agraphs: dict[int, tuple] = {}
        half = self.cfg.action_hidden_size // 2
        self._fraction = torch.linspace(
            0.0, 1.0, half, device=self.device, dtype=torch.float64
        )
        self._time_emb_buffer = torch.empty(
            1, self.cfg.action_hidden_size, device=self.device, dtype=torch.float32
        )
        self._g_time_emb = torch.empty(
            MAX_ACTION_BATCH, self.cfg.action_hidden_size, device=self.device, dtype=torch.float32
        )

    # -- batched CUDA-graph euler step --------------------------------------

    def _euler_capturable(self, n: int, handle: FlashInferCacheHandle) -> None:
        """One euler step over the first `n` requests' static buffers, in-place.
        Everything here is captured: timestep embed, time_mlp, action expert
        with batched paged attention over each request's frozen prefix, euler
        update — mstar captures the same region (`_forward_action_gen_batched`),
        and `_euler_step` is mstar's own batched-capable method (noisy
        [n*horizon, dim], timestep [n])."""
        h = self.cfg.action_horizon
        nv, tv, tev = self._g_noisy[: n * h], self._g_ts[:n], self._g_time_emb[:n]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            next_actions, next_index = self.llm._euler_step(
                nv, tv, self._fraction, tev, handle
            )
        nv.copy_(next_actions)
        tv.copy_(next_index)

    def _ensure_agraph(self, n: int, batch_plan: list) -> FlashInferCacheHandle:
        """Return the handle for batch size `n`, capturing its graph on first
        use. `batch_plan` is `[(pages, seq_pos, horizon)]` per request; the
        attention is planned here (outside capture) before recording. The
        current `_g_noisy`/`_g_ts` rows are preserved across warmup so step 0
        comes from the first replay (mstar-style)."""
        h = self.cfg.action_horizon
        if n in self._agraphs:
            attn, handle, _ = self._agraphs[n]
            attn.plan(batch_plan)
            return handle
        attn = BatchedFlashInferAttention(
            self.kv_cache, self.cfg.num_qo_heads, self.cfg.head_dim, self.device,
            bs=n, max_pages_per_req=MAX_PAGES_PER_REQ, new_len=h, cudagraph=True,
        )
        handle = FlashInferCacheHandle(attn)
        attn.plan(batch_plan)
        saved_noisy = self._g_noisy[: n * h].clone()
        saved_ts = self._g_ts[:n].clone()
        torch.cuda.synchronize()
        for _ in range(2):  # warmup, mstar-style
            self._euler_capturable(n, handle)
        self._g_noisy[: n * h].copy_(saved_noisy)
        self._g_ts[:n].copy_(saved_ts)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._euler_capturable(n, handle)
        torch.cuda.synchronize()
        # Capture records without executing: buffers still hold the loaded
        # state; step 0 comes from the first replay.
        self._agraphs[n] = (attn, handle, graph)
        return handle

    # -- node execution ------------------------------------------------------

    def _get_prefill_battn(self, n: int):
        if n not in self._prefill_battn:
            self._prefill_battn[n] = BatchedFlashInferAttention(
                self.kv_cache, self.cfg.num_qo_heads, self.cfg.head_dim, self.device,
                bs=n, max_pages_per_req=MAX_PAGES_PER_REQ, new_len=None, cudagraph=False,
                causal=False, max_total_q=MAX_PREFILL_BATCH_TOKENS,
            )
        return self._prefill_battn[n]

    def _execute_prefill_batched(self, inputs, kv) -> dict:
        """Batched prefill: one PaliGemma forward over the concatenated
        per-request prefixes (ragged batched attention), writing each request's
        KV to its own pages. Mirrors mstar's _forward_prefill_batched."""
        rids = list(inputs)
        n = len(rids)
        prefix_embs = []
        batch_plan = []
        for rid in rids:
            named, view = inputs[rid], kv[rid]
            img_emb = named["img_emb"][0] * self.llm._image_embed_scale
            text_emb = self.llm._embed_tokens_scaled(named["text_inputs"][0])
            pe = torch.cat([img_emb, text_emb], dim=0)
            assert pe.shape[0] == view["append_len"], (pe.shape, view)
            prefix_embs.append(pe)
            batch_plan.append((view["pages"], view["seq_pos"], view["append_len"]))
        attn = self._get_prefill_battn(n)
        attn.plan(batch_plan)
        handle = FlashInferCacheHandle(attn)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.llm.paligemma(
                query_sequence=torch.cat(prefix_embs, dim=0),
                cache_handle=handle,
                write_cache=True,
            )
        return {rid: {} for rid in rids}  # KV side effect only

    @torch.inference_mode()
    def execute(self, node_name, walk, inputs, kv=None):
        if node_name == "LLM" and walk == "action_gen":
            return self._execute_action_gen(inputs, kv)
        if node_name == "LLM" and walk == "prefill":
            return self._execute_prefill_batched(inputs, kv)
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        for rid, named in inputs.items():
            if node_name == "vit_encoder":
                pv = self.vit._prepare_one(named["image_inputs"][0])
                features = self.vit.encoder(pv.float())  # (cams, 256, 2048)
                outputs[rid] = {"img_emb": [features.reshape(-1, features.shape[-1])]}
            else:
                raise ValueError(f"unknown node/walk: {node_name}/{walk}")
        return outputs

    def _execute_action_gen(self, inputs, kv) -> dict:
        """One euler step for a BATCH of requests. Loads each request's loop
        state into its rows (request-major), plans the batched attention over
        every request's page table, replays the per-batch-size CUDA graph, and
        splits the result back per request. Mirrors mstar's
        `_forward_action_gen_batched`."""
        h = self.cfg.action_horizon
        rids = list(inputs)
        n = len(rids)
        assert n <= MAX_ACTION_BATCH, f"action_gen batch {n} > {MAX_ACTION_BATCH}"
        batch_plan = []
        for j, rid in enumerate(rids):
            view = kv[rid]
            assert view["scratch_len"] == h, view
            self._g_noisy[j * h : (j + 1) * h].copy_(inputs[rid]["noisy_actions"][0])
            self._g_ts[j : j + 1].copy_(inputs[rid]["timestep_index"][0])
            batch_plan.append((view["pages"], view["seq_pos"], h))
        self._ensure_agraph(n, batch_plan)  # capture on first use, else re-plan
        self._agraphs[n][2].replay()
        return {
            rid: {
                "noisy_actions": [self._g_noisy[j * h : (j + 1) * h]],
                "timestep_index": [self._g_ts[j : j + 1]],
            }
            for j, rid in enumerate(rids)
        }


PI05 = Pi05Engine  # the full model (policy + engine) — for the single-process
# Driver and for handing one instance to both roles. Multi-process: give the
# conductor a Pi05Policy (weightless) and each worker a Pi05Engine.
