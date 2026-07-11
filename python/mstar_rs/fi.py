"""FlashInfer-backed paged KV data plane — mstar-engine-aligned fast path.

Mirrors mstar's `utils/flashinfer_utils.py` + `engine/cache_manager.py`
recipe exactly:

- KV layout `[num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]`
  (axis 2: 0=K, 1=V); the whole per-layer slice goes to `wrapper.run()`.
- `BatchPrefillWithPagedKVCacheWrapper("NHD")` for both prefill and the
  action_gen suffix (multi-token queries), planned with CPU int32 index
  tensors, `causal=False`, `q_data_type=bf16`.
- K/V written by vectorized fancy indexing (token -> page/slot maps), then
  attention runs over prefix + just-written tokens.
- RoPE via `flashinfer.rope.apply_rope_pos_ids_inplace` (non-interleaved),
  absolute position ids per query token.

The handle implements the cache-manager interface mstar's transformer
modules call (`set_active_label / set_layer_idx / apply_rope /
run_attention / advance_seq_lens` — the subset mstar's own parity tests
mock), so those modules run unmodified on mstar-rs.

`FlashInferAttention` is the single-request (bs=1) path used today.
`BatchedFlashInferAttention` is the ragged-batch generalization (mstar's
`BatchedCacheManager`: one plan/run over many requests) for compute-level
batching — verified bit-exact vs per-request on GPU by
`examples/verify_batched_capture.py`; what's left is model-side wiring.
"""

from __future__ import annotations

import torch

WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferPagedKV:
    """KV pool in mstar/FlashInfer layout, one label."""

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.page_size = page_size
        self.num_pages = num_pages
        self.kv = torch.zeros(
            num_layers, num_pages, 2, page_size, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        ).contiguous()


class FlashInferAttention:
    """One wrapper + its static plan-dependent buffers for single-request
    (bs=1) execution. `cudagraph=True` pre-allocates fixed index buffers so
    `run()` is capturable; `plan()` is always called outside capture."""

    def __init__(
        self,
        cache: FlashInferPagedKV,
        num_qo_heads: int,
        head_dim: int,
        device: torch.device,
        max_new_tokens: int,
        cudagraph: bool,
        causal: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        import flashinfer

        self.cache = cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = cache.kv.shape[4]
        self.head_dim = head_dim
        self.causal = causal
        self.dtype = dtype
        self.device = device
        self._workspace = torch.empty(
            WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        # Static token->page/slot maps and rope positions (updated at plan).
        self.token_page = torch.zeros(max_new_tokens, dtype=torch.long, device=device)
        self.token_slot = torch.zeros(max_new_tokens, dtype=torch.long, device=device)
        self.pos_ids = torch.zeros(max_new_tokens, dtype=torch.long, device=device)
        self.new_len = 0
        self.total_q = 0  # total query tokens planned (== new_len for bs=1)
        if cudagraph:
            self._qo_indptr_buf = torch.zeros(2, dtype=torch.int32, device=device)
            self._kv_indptr_buf = torch.zeros(2, dtype=torch.int32, device=device)
            self._kv_indices_buf = torch.zeros(
                cache.num_pages, dtype=torch.int32, device=device
            )
            self._last_page_len_buf = torch.ones(1, dtype=torch.int32, device=device)
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                paged_kv_indptr_buf=self._kv_indptr_buf,
                paged_kv_indices_buf=self._kv_indices_buf,
                paged_kv_last_page_len_buf=self._last_page_len_buf,
            )
        else:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._workspace, "NHD"
            )

    def plan(self, pages: list[int], seq_pos: int, new_len: int) -> None:
        """Plan attention of `new_len` query tokens over `seq_pos + new_len`
        cached tokens laid out in `pages`. CPU int32 tensors, per mstar."""
        ps = self.cache.page_size
        total = seq_pos + new_len
        n_pages = (total + ps - 1) // ps
        assert n_pages <= len(pages), (n_pages, len(pages))
        qo_indptr = torch.tensor([0, new_len], dtype=torch.int32)
        kv_indptr = torch.tensor([0, n_pages], dtype=torch.int32)
        kv_indices = torch.tensor(pages[:n_pages], dtype=torch.int32)
        last_page_len = total % ps or ps
        self.wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=torch.tensor([last_page_len], dtype=torch.int32),
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=ps,
            causal=self.causal,
            q_data_type=self.dtype,
        )
        # Token -> (page, slot) for the new tokens; rope positions.
        g = torch.arange(seq_pos, total)
        page_tbl = torch.tensor(pages[:n_pages], dtype=torch.long)
        self.token_page[:new_len].copy_(
            page_tbl[torch.div(g, ps, rounding_mode="floor")], non_blocking=True
        )
        self.token_slot[:new_len].copy_(g % ps, non_blocking=True)
        self.pos_ids[:new_len].copy_(g, non_blocking=True)
        self.new_len = new_len
        self.total_q = new_len


class BatchedFlashInferAttention:
    """Batched paged attention over a RAGGED batch of requests — one
    FlashInfer plan + one run for the whole batch (mstar's
    `BatchedCacheManager`, `engine/cache_manager.py`). The ragged batch's
    per-request page tables are concatenated into one set of index tensors;
    `wrapper.run()` issues a single kernel over all requests.

    Sized for a FIXED `bs` (a capture bucket) with `new_len` query tokens per
    request (decode: 1). All buffers are static so `run()` is CUDA-graph
    capturable at that `bs`; `plan()` is always called OUTSIDE capture and
    refills the fixed index buffers (the FlashInfer `use_cuda_graph` pattern —
    identical to the single-request path, just wider).

    VERIFIED bit-exact vs per-request attention on GPU
    (`examples/verify_batched_capture.py`: batched == per-request AND
    graph-replay == eager, max_abs_diff 0.0). What remains before a model can
    use it is model-side wiring: the model's decode step must call this with a
    stacked batch (+ a [max_bs, V] batched sampler for orpheus), after which
    drop that (node, walk) from the model's `unbatchable()`.
    """

    def __init__(
        self,
        cache: FlashInferPagedKV,
        num_qo_heads: int,
        head_dim: int,
        device: torch.device,
        bs: int,
        max_pages_per_req: int,
        new_len: int | None = 1,
        cudagraph: bool = True,
        causal: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        max_total_q: int | None = None,
    ) -> None:
        import flashinfer

        self.cache = cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = cache.kv.shape[4]
        self.head_dim = head_dim
        self.causal = causal
        self.dtype = dtype
        self.device = device
        self.bs = bs
        # `new_len` fixed (decode: 1 token/request, graph-capturable) OR None
        # for a RAGGED batch — variable query length per request (prefill over
        # concatenated prefixes). Ragged is eager only (cudagraph=False); its
        # total query count varies, so `max_total_q` sizes the token buffers.
        self.new_len = new_len
        if new_len is None:
            assert not cudagraph, "ragged batch (new_len=None) is eager-only"
            assert max_total_q is not None, "ragged batch needs max_total_q"
            self.max_q = max_total_q
        else:
            self.max_q = bs * new_len  # total query tokens across the batch
        self._workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        # Token -> (page, slot) and rope positions, concatenated over the batch.
        self.token_page = torch.zeros(self.max_q, dtype=torch.long, device=device)
        self.token_slot = torch.zeros(self.max_q, dtype=torch.long, device=device)
        self.pos_ids = torch.zeros(self.max_q, dtype=torch.long, device=device)
        # Ragged batch index buffers, sized for the widest possible plan.
        self._qo_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        self._kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        self._kv_indices_buf = torch.zeros(
            bs * max_pages_per_req, dtype=torch.int32, device=device
        )
        self._last_page_len_buf = torch.ones(bs, dtype=torch.int32, device=device)
        if cudagraph:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                paged_kv_indptr_buf=self._kv_indptr_buf,
                paged_kv_indices_buf=self._kv_indices_buf,
                paged_kv_last_page_len_buf=self._last_page_len_buf,
            )
        else:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._workspace, "NHD"
            )

    def plan(self, batch: list[tuple[list[int], int, int]]) -> None:
        """Plan attention for a batch. `batch[i] = (pages, seq_pos, new_len)`
        for request i. `len(batch)` must equal the fixed `bs` (pad with dummy
        requests to the bucket before calling). Builds ragged CPU int32 index
        tensors and writes the static buffers, exactly like the single-request
        `FlashInferAttention.plan`, concatenated across the batch."""
        assert len(batch) == self.bs, (len(batch), self.bs)
        ps = self.cache.page_size
        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices: list[int] = []
        last_page_len: list[int] = []
        tok_page: list[int] = []
        tok_slot: list[int] = []
        pos: list[int] = []
        for pages, seq_pos, new_len in batch:
            assert self.new_len is None or new_len == self.new_len, (new_len, self.new_len)
            total = seq_pos + new_len
            n_pages = (total + ps - 1) // ps
            assert n_pages <= len(pages) and n_pages <= (len(self._kv_indices_buf) // self.bs), (
                n_pages, len(pages),
            )
            qo_indptr.append(qo_indptr[-1] + new_len)
            kv_indptr.append(kv_indptr[-1] + n_pages)
            kv_indices.extend(pages[:n_pages])
            last_page_len.append(total % ps or ps)
            for g in range(seq_pos, total):
                tok_page.append(pages[g // ps])
                tok_slot.append(g % ps)
                pos.append(g)

        n_idx = len(kv_indices)
        self._qo_indptr_buf.copy_(torch.tensor(qo_indptr, dtype=torch.int32, device=self.device))
        self._kv_indptr_buf.copy_(torch.tensor(kv_indptr, dtype=torch.int32, device=self.device))
        self._kv_indices_buf[:n_idx].copy_(
            torch.tensor(kv_indices, dtype=torch.int32, device=self.device)
        )
        self._last_page_len_buf.copy_(
            torch.tensor(last_page_len, dtype=torch.int32, device=self.device)
        )
        nq = len(pos)
        self.token_page[:nq].copy_(torch.tensor(tok_page, dtype=torch.long), non_blocking=True)
        self.token_slot[:nq].copy_(torch.tensor(tok_slot, dtype=torch.long), non_blocking=True)
        self.pos_ids[:nq].copy_(torch.tensor(pos, dtype=torch.long), non_blocking=True)
        self.total_q = nq
        self.wrapper.plan(
            qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
            paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
            paged_kv_indices=torch.tensor(kv_indices, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=ps,
            causal=self.causal,
            q_data_type=self.dtype,
        )


class FlashInferCacheHandle:
    """mstar cache-manager interface over a planned FlashInferAttention."""

    def __init__(self, attn: FlashInferAttention) -> None:
        self.attn = attn
        self.layer_idx = 0

    def set_active_label(self, label: str) -> None:
        pass

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-interleaved RoPE via FlashInfer. Routes to the llama3.1
        variant when the model passes scaling params (orpheus), exactly as
        mstar's `cache_manager.apply_rope` does."""
        import flashinfer

        n = self.attn.total_q
        q, k = q.to(self.attn.dtype).contiguous(), k.to(self.attn.dtype).contiguous()
        pos_ids = self.attn.pos_ids[:n]
        llama31 = {
            key: kwargs[key]
            for key in ("low_freq_factor", "high_freq_factor", "old_context_len")
            if key in kwargs and kwargs[key] is not None
        }
        if llama31:
            flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
                q, k, pos_ids,
                interleave=False,
                rope_scale=rope_scale or 1.0,
                rope_theta=rope_theta,
                **llama31,
            )
        else:
            flashinfer.rope.apply_rope_pos_ids_inplace(
                q, k, pos_ids,
                interleave=False,
                rope_scale=rope_scale or 1.0,
                rope_theta=rope_theta,
            )
        return q, k

    def run_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        attn = self.attn
        n = attn.total_q
        layer = attn.cache.kv[self.layer_idx]
        layer[attn.token_page[:n], 0, attn.token_slot[:n]] = k.to(attn.dtype)
        layer[attn.token_page[:n], 1, attn.token_slot[:n]] = v.to(attn.dtype)
        return attn.wrapper.run(q.to(attn.dtype), layer)

    def advance_seq_lens(self, *a: object, **kw: object) -> None:
        pass  # the Rust runtime owns sequence positions
