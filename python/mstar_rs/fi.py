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
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        import flashinfer

        self.cache = cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = cache.kv.shape[4]
        self.head_dim = head_dim
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
            causal=False,
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
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import flashinfer

        n = self.attn.new_len
        q, k = q.to(self.attn.dtype).contiguous(), k.to(self.attn.dtype).contiguous()
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q, k, self.attn.pos_ids[:n],
            interleave=False,
            rope_scale=rope_scale or 1.0,
            rope_theta=rope_theta,
        )
        return q, k

    def run_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        attn = self.attn
        n = attn.new_len
        layer = attn.cache.kv[self.layer_idx]
        layer[attn.token_page[:n], 0, attn.token_slot[:n]] = k.to(attn.dtype)
        layer[attn.token_page[:n], 1, attn.token_slot[:n]] = v.to(attn.dtype)
        return attn.wrapper.run(q.to(attn.dtype), layer)

    def advance_seq_lens(self, *a: object, **kw: object) -> None:
        pass  # the Rust runtime owns sequence positions
