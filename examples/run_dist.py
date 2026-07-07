"""Multi-process demo: a conductor + two worker processes run a disaggregated
2-node model, proving cross-worker tensor transport over shared memory.

Topology: node `a` (y = 2*x) on worker_0, node `b` (z = y + 1, emitted) on
worker_1. So `a`'s output crosses from worker_0's SHM arena to worker_1 —
the real multi-process path — with the conductor routing only descriptors
over the UDS control mesh.

    python examples/run_dist.py

CPU-only + a toy model, so it runs anywhere with no weights/GPU: the point is
the runtime split (conductor drives, workers execute, tensors via /dev/shm),
not the compute.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Model, edge, emit, node, sequential  # noqa: E402


class ToyDisagg(Model):
    def walks(self):
        return {
            "fwd": sequential(
                node("a", ["x"], [edge("b", "y")]),
                node("b", ["y"], [emit("z", modality="tensor", persist=True)]),
            )
        }

    def initial_inputs(self, request):
        return "fwd", [("a", "x", [request["x"]])]

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            if node_name == "a":
                out[rid] = {"y": [named["x"][0] * 2]}
            else:  # b
                out[rid] = {"z": [named["y"][0] + 1]}
        return out

    def postprocess(self, name, modality, tensors):
        return tensors[0]


def worker_main(worker_id: str, socket_dir: str) -> None:
    # Log any startup crash to a file (spawned daemons swallow stderr).
    try:
        from mstar_rs.dist import Worker

        Worker(worker_id, ToyDisagg(), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_dist_")

    workers = []
    for wid in ("worker_0", "worker_1"):
        p = ctx.Process(target=worker_main, args=(wid, socket_dir), daemon=True)
        p.start()
        workers.append(p)
    # Wait for both worker inboxes to appear (their bind creates the socket).
    deadline = time.time() + 15
    need = {f"{socket_dir}/worker_0.ipc", f"{socket_dir}/worker_1.ipc"}
    while time.time() < deadline:
        if all(Path(p).exists() for p in need):
            break
        time.sleep(0.1)
    bound = [w for w in ("worker_0", "worker_1") if Path(f"{socket_dir}/{w}.ipc").exists()]
    print(f"workers bound: {bound}")

    from mstar_rs.dist import Conductor

    cond = Conductor(
        ToyDisagg(),
        node_to_worker={"a": "worker_0", "b": "worker_1"},
        socket_dir=socket_dir,
    )

    ok = True
    for i in range(3):
        x = torch.tensor([float(i), float(i + 1), float(i + 2)])
        rid = cond.submit({"x": x})
        results = cond.run_until_idle()
        z = results[rid][0]
        expected = x * 2 + 1  # a doubles, b adds one
        match = torch.equal(z, expected)
        ok &= match
        print(f"request {rid}: x={x.tolist()} -> z={z.tolist()} "
              f"(expected {expected.tolist()}) {'OK' if match else 'MISMATCH'}")
        cond.results.clear()

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=3)
    print("\nCROSS-WORKER SHM TRANSPORT OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
