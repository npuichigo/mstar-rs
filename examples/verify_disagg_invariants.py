"""Regression invariants for the decentralized (DisaggCoordinator/DisaggWorker)
path — the decentralized counterpart of the old centralized checks:

  1. SHM + bookkeeping reclaim — after requests finish, both shared-memory
     arenas return to fully free and the coordinator's per-request maps are
     drained (no leak across a serving run).
  2. Loop-stop bridging — the engine's `loops_to_finish()` (EOS/check_stop),
     which fires in the worker, terminates the graph loop early instead of
     running to `max_iters`.

(Worker fault isolation is covered by verify_disagg_worker_crash.py.)
CPU-only, no weights/GPU — runs anywhere.

    python examples/verify_disagg_invariants.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from mstar_rs import Model  # noqa: E402
from mstar_rs.dist import DisaggCoordinator, DisaggWorker  # noqa: E402
from mstar_rs.graph import edge, emit, loop, node  # noqa: E402

PARTS = [{"name": "P", "walks": ["gen"]}]


class LoopEcho(Model):
    """AR echo as a graph loop; the engine stops each request's loop when its
    tokens run out (mstar's check_stop -> STOP_LOOPS)."""

    def __init__(self) -> None:
        self._pending_stops: list = []

    def partitions(self):
        return (PARTS, [])

    def walks(self):
        return {
            "gen": loop(
                "gen_loop",
                node("step", ["state"], [
                    edge("step", "state"),
                    emit("token", modality="text"),
                ]),
                max_iters=10_000,
            )
        }

    def initial_inputs(self, request):
        return "gen", [("step", "state",
                        [torch.tensor(list(request["tokens"]), dtype=torch.int64)])]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        return None

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())

    def loops_to_finish(self):
        stops, self._pending_stops = self._pending_stops, []
        return stops

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            out[rid] = {"token": [state[:1]], "state": [state[1:]]}
            if state.numel() <= 1:
                self._pending_stops.append((rid, "gen_loop"))
        return out


def main() -> int:
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_disagg_inv_")
    worker = DisaggWorker("w0", LoopEcho(), ["P"], {"P": "w0"}, socket_dir,
                          device="cpu")
    threading.Thread(target=worker.run, daemon=True).start()
    deadline = time.time() + 5
    while time.time() < deadline and not Path(f"{socket_dir}/w0.ipc").exists():
        time.sleep(0.05)

    cond = DisaggCoordinator(LoopEcho(), {"P": "w0"}, socket_dir)
    lengths = [3, 5, 2, 7, 4, 6]
    gids = [cond.submit({"tokens": [100 + i for i in range(n)]})
            for n in lengths]

    deadline = time.time() + 15
    while time.time() < deadline and len(cond.finished) < len(gids):
        cond.poll(timeout_ms=20)
    assert len(cond.finished) == len(gids), (
        f"only {len(cond.finished)}/{len(gids)} finished; errors={cond.errors}")

    # 2. loop-stop bridging: emitted exactly len(tokens) tokens, not 10k.
    ok_stop = all(len(cond.results[g]) == n for g, n in zip(gids, lengths))
    for g, n in zip(gids, lengths):
        assert len(cond.results[g]) == n, (
            f"request {g}: {len(cond.results[g])} tokens, expected {n}")
    print(f"1. LOOP-STOP OK — {lengths} tokens emitted exactly "
          f"(max_iters=10000 never reached)")

    # 1. reclaim: both arenas fully free, per-request maps drained.
    time.sleep(0.5)  # let frees propagate
    cond.poll(timeout_ms=20)
    def fully_free(pool):
        return all(
            pool.arena.segment(i).bytes_free == pool.arena.segment(i).size
            for i in range(pool.arena.num_segments))

    assert fully_free(cond.shm), "coordinator arena leak"
    assert fully_free(worker.shm), "worker arena leak"
    assert not cond._pending, f"pending leak: {cond._pending}"
    print("2. RECLAIM OK — both arenas fully free, coordinator maps drained "
          f"after {len(gids)} requests")

    cond.shutdown_workers()
    print(f"\nDISAGG INVARIANTS {'OK' if ok_stop else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
