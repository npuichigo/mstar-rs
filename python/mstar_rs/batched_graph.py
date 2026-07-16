"""Batched CUDA-graph capture with pad-to-bucket replay.

Mirrors mstar's `engine/cuda_graph_runner.py`: capture a decode step at a set
of batch-size **buckets** (`[1, 2, 4, 8, 16, 32, 64]`) over `[max_bs, ...]`
static buffers, then at replay time pad a real batch of `N` up to the nearest
captured bucket, replay that graph, and discard the padding rows. This is what
lets mstar batch compute across requests while still using CUDA graphs; the
bs=1 single-slot graphs in pi05/orpheus are the degenerate case.

Verification:
  * `padded_bucket` and the pad/discard index math are **pure Python** and are
    unit-tested on CPU (examples/verify_batched_graph_buckets.py).
  * `BucketedCudaGraph` capture/replay is **verified bit-exact on GPU** by
    `examples/verify_batched_capture.py` (graph-replay == eager batched,
    max_abs_diff 0.0). What remains is wiring a model's decode step to it.
"""

from __future__ import annotations

from typing import Callable

import torch

# mstar's DEFAULT_AR_CAPTURE_BATCH_SIZES (cuda_graph_runner.py:45).
DEFAULT_CAPTURE_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def padded_bucket(n: int, buckets: tuple[int, ...] = DEFAULT_CAPTURE_BATCH_SIZES) -> int:
    """Smallest captured bucket >= n (mstar's `_get_padded_batch_size`,
    `bisect_left`). Pure function — CPU-testable."""
    if n <= 0:
        raise ValueError(f"batch size must be positive, got {n}")
    import bisect

    i = bisect.bisect_left(buckets, n)
    if i == len(buckets):
        raise ValueError(f"batch {n} exceeds max captured bucket {buckets[-1]}")
    return buckets[i]


class BucketedCudaGraph:
    """Capture `step_fn` at each bucket size and replay with pad-to-bucket.

    `step_fn(bs)` must run one decode step reading/writing the SAME static
    buffers every call (so the captured graph replays deterministically). The
    caller loads the real batch's rows `[0:real_bs]` of those buffers before
    `replay`, and reads outputs from rows `[0:real_bs]` after — the padding
    rows `[real_bs:bucket]` carry dummy data and are ignored.

    GPU-only; capture/replay verified bit-exact by
    `examples/verify_batched_capture.py`.
    """

    def __init__(
        self,
        step_fn: Callable[[int], None],
        buckets: tuple[int, ...] = DEFAULT_CAPTURE_BATCH_SIZES,
        warmups: int = 3,
    ) -> None:
        self.buckets = buckets
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        for bs in buckets:
            # Warm up (allocates workspaces, JITs kernels) OUTSIDE capture, as
            # mstar does — capture records without executing, so a cold kernel
            # captured would replay uninitialized.
            for _ in range(warmups):
                step_fn(bs)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step_fn(bs)
            self._graphs[bs] = g

    def replay(self, real_bs: int) -> int:
        """Replay the graph for the bucket covering `real_bs`. Returns the
        bucket size (so the caller knows the valid row range is `[0:real_bs]`
        and `[real_bs:bucket]` are padding to discard)."""
        bucket = padded_bucket(real_bs, self.buckets)
        self._graphs[bucket].replay()
        return bucket
