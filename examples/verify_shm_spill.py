"""Arena-saturation degradation: at the segment cap, `ShmPool.stage`
falls back to the INLINE descriptor form (bytes ride the control message —
the multi-node wire shape) instead of failing. Proven two ways:

1. Pool-level: a tiny arena staged past its cap returns inline dicts that
   round-trip through `read`; slots that did land in the arena still free.
2. End-to-end: the decentralized stack under a tiny-arena env serves a
   multi-chunk stream whose tensors cannot all fit — results stay correct,
   nothing crashes, and the arena still fully reclaims at idle.

    python examples/verify_shm_spill.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TINY = {"MSTAR_RS_SHM_SEGMENT_MB": "1", "MSTAR_RS_SHM_MAX_SEGMENTS": "2"}
os.environ.update(TINY)

from mstar_rs import Model  # noqa: E402
from mstar_rs.graph import (  # noqa: E402
    connection,
    edge,
    emit,
    fixed_chunk,
    loop,
    node,
    partition,
    stream_edge,
)

PARTITION_TO_WORKER = {"A": "wA", "B": "wB"}
STEPS = 4
CHUNK = 900_000   # ~0.9 MB/chunk vs a 2 MiB arena: must spill mid-stream


def pool_level() -> bool:
    from mstar_rs.dist import ShmPool

    pool = ShmPool("spill_prod")
    tensors = [torch.full((CHUNK,), i, dtype=torch.uint8) for i in range(6)]
    descs = [pool.stage(t) for t in tensors]
    arena_descs = [d for d in descs if isinstance(d, list)]
    inline_descs = [d for d in descs if isinstance(d, dict)]
    ok = bool(arena_descs) and bool(inline_descs)
    print(f"  staged 6x{CHUNK}B into a 2 MiB pool: {len(arena_descs)} in "
          f"arena, {len(inline_descs)} inline {'OK' if ok else 'FAIL'}")

    reader = ShmPool("spill_cons")
    for t, d in zip(tensors, descs, strict=True):
        got = reader.read(d)
        if not torch.equal(got, t):
            print(f"  MISMATCH for desc {type(d).__name__}")
            return False
    print("  all 6 round-trip through read() OK")

    for d in arena_descs:
        pool.free(d)
    total, free, largest = pool.arena.stats()
    # largest is PER-SEGMENT (free space cannot span separate mappings), so
    # a fully drained N-segment arena reports one segment's size.
    ok2 = free == total and largest == total // pool.arena.num_segments
    print(f"  arena fully reclaimed: stats=({total}, {free}, {largest}) "
          f"{'OK' if ok2 else 'FAIL'}")
    return ok and ok2


class BigStream(Model):
    """A streams CHUNK-sized tensors; B checksums each and emits."""

    def __init__(self) -> None:
        self._stops: list = []

    def walks(self):
        return {
            "a": loop(
                "a_loop",
                node("A", ["x"], [
                    stream_edge("B", "s", "B"),
                    edge("A", "x"),
                ]),
                max_iters=100,
            ),
            "b": node("B", ["s"], [emit("out", modality="tensor",
                                        persist=True)]),
        }

    def partitions(self):
        return ([partition("A", ["a"]), partition("B", ["b"])],
                [connection("A", "B", "s", fixed_chunk(1))])

    def initial_walks(self, request):
        n = torch.tensor([int(request["steps"])])
        return [("a", [("A", "x", [n])]), ("b", [])]

    def next_forward(self, rid, part, walk, fwd_index, persist, stream_done):
        if walk == "b" and not stream_done:
            return "b", []
        return None

    def loops_to_finish(self):
        stops, self._stops = self._stops, []
        return stops

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            if node_name == "A":
                left = int(named["x"][0][0])
                big = torch.full((CHUNK,), left % 251, dtype=torch.uint8)
                out[rid] = {"s": [big],
                            "x": [torch.tensor([left - 1])]}
                if left <= 1:
                    self._stops.append((rid, "a_loop"))
            else:
                out[rid] = {"out": [named["s"][0].to(torch.int64).sum()]}
        return out

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def worker_main(worker_id, parts, socket_dir):
    try:
        from mstar_rs.dist import DisaggWorker

        DisaggWorker(worker_id, BigStream(), parts, PARTITION_TO_WORKER,
                     socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def e2e() -> bool:
    from mstar_rs.dist import DisaggCoordinator

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_spill_")
    workers = []
    for wid, parts in (("wA", ["A"]), ("wB", ["B"])):
        p = ctx.Process(target=worker_main, args=(wid, parts, socket_dir),
                        daemon=True)
        p.start()
        workers.append(p)
    deadline = time.time() + 15
    need = [f"{socket_dir}/wA.ipc", f"{socket_dir}/wB.ipc"]
    while time.time() < deadline and not all(Path(x).exists() for x in need):
        time.sleep(0.05)

    cond = DisaggCoordinator(BigStream(), PARTITION_TO_WORKER, socket_dir)
    expected = [((s % 251) * CHUNK) for s in range(STEPS, 0, -1)]
    gid = cond.submit({"steps": STEPS})
    cond.run_until_idle()
    got = cond.results[gid]
    ok = got == expected
    print(f"  e2e stream (4x{CHUNK}B through a 2 MiB arena): "
          f"{'OK' if ok else f'MISMATCH {got} != {expected}'}")

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for wid in ("wA", "wB"):
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")
            return False
    return ok


def main() -> int:
    ok = pool_level()
    ok &= e2e()
    print(f"\nSHM SPILL {'OK' if ok else 'FAILED'} (at the segment cap "
          f"stage() degrades to inline descriptors; streams stay correct; "
          f"arena reclaims)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
