"""The Step-2 pinning question, measured: does `cudaHostRegister`ing a
`/dev/shm` segment make D2H/H2D copies through it truly async on a side
stream (overlapping compute), where the unregistered (pageable) mmap
synchronizes?

Three configurations, same copy size, compute kernel running on the default
stream while a side stream copies GPU->host:

  A. pageable  — copy into the raw (unregistered) segment view
  B. registered — same view after cudaHostRegister(ptr, len)
  C. torch pinned — torch's own pin_memory tensor (the reference ceiling)

For each: run compute alone (T_c), copy alone (T_x), then both concurrently
(T_b). Overlap ratio = (T_c + T_x - T_b) / min(T_c, T_x): ~1 = full overlap,
~0 = serialized. Expectation: A serializes (async_ copy into pageable memory
degrades to sync), B ≈ C overlap.

    python examples/bench_shm_pinning.py    # 1 GPU, ~1 minute
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs._core import SegmentedShmArena  # noqa: E402

MB = 1 << 20
COPY_BYTES = 256 * MB
REPS = 20


def cuda_host_register(ptr: int, nbytes: int) -> None:
    cudart = ctypes.CDLL("libcudart.so")
    # cudaHostRegisterPortable = 1: pinned for all CUDA contexts.
    rc = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), 1)
    if rc != 0:
        raise RuntimeError(f"cudaHostRegister failed: rc={rc}")


def compute_ms(a, b, iters=60) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a
    for _ in range(iters):
        c = c @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, c


def copy_ms(src_gpu, dst_view, stream) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(REPS):
            dst_view.copy_(src_gpu, non_blocking=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def both_ms(a, b, src_gpu, dst_view, stream, iters=60) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(REPS):
            dst_view.copy_(src_gpu, non_blocking=True)
    c = a
    for _ in range(iters):
        c = c @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, c


def bench(tag: str, dst_view, src_gpu, a, b, stream) -> float:
    t_c, _ = compute_ms(a, b)
    t_x = copy_ms(src_gpu, dst_view, stream)
    t_b, _ = both_ms(a, b, src_gpu, dst_view, stream)
    overlap = (t_c + t_x - t_b) / max(min(t_c, t_x), 1e-9)
    print(f"  {tag:12s} compute={t_c:7.1f}ms copy={t_x:7.1f}ms both={t_b:7.1f}ms "
          f"-> overlap {overlap*100:5.1f}%")
    return overlap


def main() -> int:
    dev = torch.device("cuda")
    src_gpu = torch.randn(COPY_BYTES // 4, device=dev)  # fp32
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    stream = torch.cuda.Stream()

    # A/B share one arena: seg0 raw, then registered in place for run B —
    # same physical pages, isolating the registration effect.
    arena = SegmentedShmArena.create("mstar_rs_pinbench", COPY_BYTES + (1 << 12), 2)
    seg = arena.segment(0)
    view = torch.frombuffer(memoryview(seg)[:COPY_BYTES], dtype=torch.float32)

    print(f"[bench] {COPY_BYTES // MB} MiB x{REPS} D2H on a side stream vs matmul compute")
    ov_pageable = bench("pageable", view, src_gpu, a, b, stream)

    ptr, nbytes = seg.ptr_len()
    t0 = time.perf_counter()
    cuda_host_register(ptr, nbytes)
    print(f"  [register] cudaHostRegister({nbytes // MB} MiB) took "
          f"{(time.perf_counter()-t0)*1e3:.0f}ms (one-time per segment)")
    ov_registered = bench("registered", view, src_gpu, a, b, stream)

    pinned = torch.empty(COPY_BYTES // 4, dtype=torch.float32, pin_memory=True)
    ov_pinned = bench("torch-pinned", pinned, src_gpu, a, b, stream)

    # Registered segment must behave like real pinned memory, and clearly
    # better than the pageable mmap.
    ok = ov_registered > 0.5 and ov_registered > ov_pageable + 0.2 \
        and abs(ov_registered - ov_pinned) < 0.35
    print(f"\nSHM PINNING {'OK' if ok else 'INCONCLUSIVE'} "
          f"(registered /dev/shm segment overlaps like torch-pinned; pageable does not)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
