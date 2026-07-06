"""Regression checks for the conductor/worker (multi-process) control path.

Runs the worker in a thread (same process) so BOTH shared-memory arenas are
inspectable, and exercises three invariants that are easy to regress:

  1. SHM + descriptor reclaim — after requests finish, both arenas return to
     fully free and the conductor's `desc`/`_req_uuids` maps are empty (no
     leak across a serving run).
  2. Worker fault isolation — an exception in `execute` fails the request fast
     (recorded in `errors`) instead of killing the worker and hanging the
     conductor on a stuck in-flight count.
  3. Loop-stop bridging — a model's `loops_to_finish()` (EOS / check_stop),
     which runs in the worker, is carried to the conductor and terminates the
     loop early rather than running to `max_iters`.

CPU-only, no weights/GPU — runs anywhere.

    python examples/verify_dist.py
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

from mstar_rs.dist import Conductor, Worker  # noqa: E402
from mstar_rs.graph import edge, emit, loop, node  # noqa: E402
from mstar_rs.model import Model  # noqa: E402
from mstar_rs.models.echo import EchoAR  # noqa: E402


def _start_worker(model, socket_dir, wid="worker_0"):
    w = Worker(wid, model, socket_dir, device="cpu")
    threading.Thread(target=w.run, daemon=True).start()
    deadline = time.time() + 5
    while time.time() < deadline and not Path(f"{socket_dir}/{wid}.sock").exists():
        time.sleep(0.05)
    return w


def _drive(cond, budget=8.0):
    t0 = time.time()
    while time.time() - t0 < budget:
        if not cond.poll(timeout_ms=50) and cond._inflight == 0:
            return True
    return False


def test_reclaim() -> None:
    d = tempfile.mkdtemp(prefix="verify_reclaim_")
    w = _start_worker(EchoAR(), d)
    cond = Conductor(EchoAR(), {"step": "worker_0"}, d)
    for i in range(6):
        cond.submit({"tokens": list(range(3 + i)), "max_tokens": 100})
    assert _drive(cond), "did not reach idle"
    time.sleep(0.4)  # let the worker process the free messages
    assert not cond.desc, f"conductor desc leak: {len(cond.desc)} entries"
    assert not cond._req_uuids, f"req_uuids leak: {cond._req_uuids}"
    assert cond.shm.arena.bytes_free == cond.shm.arena.size, "conductor arena leak"
    assert w.shm.arena.bytes_free == w.shm.arena.size, "worker arena leak"
    cond.shutdown_workers()
    print("1. RECLAIM OK — both arenas fully free, no desc/uuid leak after 6 requests")


class _Boom(EchoAR):
    def execute(self, *a, **k):
        raise ValueError("boom in execute")


def test_error_no_hang() -> None:
    d = tempfile.mkdtemp(prefix="verify_err_")
    _start_worker(_Boom(), d)
    cond = Conductor(_Boom(), {"step": "worker_0"}, d)
    rid = cond.submit({"tokens": [1, 2, 3], "max_tokens": 10})
    assert _drive(cond, budget=5.0), "conductor hung after worker error"
    assert rid in cond.errors, "worker error not recorded"
    assert not cond._req_uuids.get(rid), "failed request's uuids not reclaimed"
    cond.shutdown_workers()
    print(f"2. ERROR-PATH OK — request failed fast ({cond.errors[rid]}), no hang")


class _LoopStop(Model):
    """Emits a counter each iteration; asks to stop after `stop_at` (like an
    EOS detected in execute), which must bridge worker->conductor."""

    def __init__(self, stop_at: int = 5) -> None:
        self.stop_at = stop_at
        self._pending: list = []
        self._count: dict = {}

    def walks(self):
        return {
            "gen": loop(
                "g",
                node("dec", ["state"], [emit("tok", modality="text"), edge("dec", "state")]),
                max_iters=1000,
            )
        }

    def initial_inputs(self, request):
        return "gen", [("dec", "state", [torch.tensor([0], dtype=torch.int64)])]

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            s = named["state"][0]
            out[rid] = {"tok": [s], "state": [s + 1]}
            self._count[rid] = self._count.get(rid, 0) + 1
            if self._count[rid] >= self.stop_at:
                self._pending.append((rid, "g"))
        return out

    def loops_to_finish(self):
        p, self._pending = self._pending, []
        return p

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def test_loop_stop() -> None:
    d = tempfile.mkdtemp(prefix="verify_loop_")
    _start_worker(_LoopStop(stop_at=5), d)
    cond = Conductor(_LoopStop(stop_at=5), {"dec": "worker_0"}, d)
    rid = cond.submit({})
    assert _drive(cond), "loop never stopped"
    toks = cond.results.get(rid, [])
    assert toks == [0, 1, 2, 3, 4], f"expected [0..4], got {toks}"
    cond.shutdown_workers()
    print(f"3. LOOP-STOP OK — EOS bridged; stopped at {len(toks)} tokens (max_iters=1000)")


def main() -> int:
    test_reclaim()
    test_error_no_hang()
    test_loop_stop()
    print("\nALL DIST CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
