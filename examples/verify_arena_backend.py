"""Interop proof for the RFC #130 Step-2 backend draft: mstar's tensor
transport primitives (store / load / uuid-reclaim) over the segmented arena —
the object `SharedMemoryCommunicationManager` swaps in for the per-tensor
file dance. Cross-process load, grouped+idempotent uuid reclaim, dedicated
segment for oversized tensors, and the registration hook firing per segment.

Run with an environment that has BOTH mstar (the draft in its tree) and
mstar_rs:   python examples/verify_arena_backend.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO.parent / "mstar"))

from mstar.communication.arena_backend import ArenaTensorBackend  # noqa: E402


def consumer(q, desc) -> None:
    c = ArenaTensorBackend("ab_consumer", segment_size=4096, max_segments=2)
    q.put(c.load(desc).sum().item())


def main() -> int:
    regs = []
    b = ArenaTensorBackend("ab_prod", segment_size=4096, max_segments=4,
                           on_new_segment=lambda n, s: regs.append(n))
    t = torch.arange(900, dtype=torch.int64)   # 7200 B -> dedicated segment
    b.store(1, torch.randn(8))
    d2 = b.store(1, t)                          # same uuid holds two tensors

    ok = torch.equal(b.load(d2), t)
    print(f"  same-process load: {'OK' if ok else 'FAIL'}")

    q = mp.get_context("spawn").Queue()
    p = mp.get_context("spawn").Process(target=consumer, args=(q, d2))
    p.start()
    got = q.get(timeout=20)
    p.join(timeout=5)
    ok_x = got == t.sum().item()
    print(f"  cross-process load: {'OK' if ok_x else 'FAIL'}")

    ok_free = b.free_uuid(1) == 2 and b.free_uuid(1) == 0
    print(f"  uuid reclaim (grouped, idempotent): {'OK' if ok_free else 'FAIL'}")
    ok_seg = len(regs) == 2 and d2[0].endswith(".seg1")
    print(f"  segments: {regs} (oversized got a dedicated one) "
          f"{'OK' if ok_seg else 'FAIL'}")

    all_ok = ok and ok_x and ok_free and ok_seg
    print(f"\nARENA BACKEND {'OK' if all_ok else 'FAILED'} "
          f"(store/load cross-process, uuid reclaim, registration hook)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
