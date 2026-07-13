"""Verify the conductor drives a TENSOR-PARALLEL partition across N worker
processes: dispatch the same batch to every rank, run them in NCCL/gloo
lockstep (a real all-reduce), and route ONLY rank 0's (replicated) output.

Uses a toy all-reduce "engine" on gloo/CPU so the conductor<->worker WIRING is
tested with no weights and no GPUs — the real GPU TP forward (sharded Qwen3
Thinker over NCCL) is proven separately in `tp_thinker_forward.py`. Together
they cover both halves: this proves the runtime dispatches + dedupes TP
correctly; that proves a real sharded model runs under the same handle.

Toy: node `tp`, each rank computes x*(rank+1) then all_reduce(SUM). For
world=2 the result is x*1 + x*2 = x*3, identical on both ranks; the conductor
routes rank 0's copy and the replica stays silent.

    python examples/verify_dist_tp.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import ModelEngine, ModelPolicy, emit, node  # noqa: E402

WORLD = 2
PORT = 29611


class ToyTPPolicy(ModelPolicy):
    """Weightless conductor-side policy: one node, single pass, emit `y`."""

    def walks(self):
        return {"fwd": node("tp", ["x"], [emit("y", modality="tensor", persist=True)])}

    def initial_inputs(self, request):
        return "fwd", [("tp", "x", [request["x"]])]

    def postprocess(self, name, modality, tensors):
        return tensors[0]


class ToyTPEngine(ModelEngine):
    """Worker-side engine: genuinely tensor-parallel — each rank scales the
    input by (rank+1) then all-reduces, so the result depends on every rank
    reaching the collective in lockstep (exactly the model-forward TP path)."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank = rank
        dist.init_process_group(
            backend="gloo", init_method=f"tcp://127.0.0.1:{PORT}",
            rank=rank, world_size=world,
        )

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            local = named["x"][0] * (self.rank + 1)
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            out[rid] = {"y": [local]}
        return out


def worker_main(worker_id: str, rank: int, world: int, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker

        Worker(worker_id, ToyTPEngine(rank, world), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_tp_")
    ids = [f"tp_rank{r}" for r in range(WORLD)]

    workers = []
    for rank, wid in enumerate(ids):
        p = ctx.Process(target=worker_main, args=(wid, rank, WORLD, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    deadline = time.time() + 20
    need = {f"{socket_dir}/{w}.ipc" for w in ids}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        time.sleep(0.1)
    bound = [w for w in ids if Path(f"{socket_dir}/{w}.ipc").exists()]
    print(f"workers bound: {bound}")

    from mstar_rs.dist import Conductor

    # The TP'd node maps to the LIST of its ranks (rank 0 first).
    cond = Conductor(ToyTPPolicy(), node_to_worker={"tp": ids}, socket_dir=socket_dir)

    scale = WORLD * (WORLD + 1) // 2  # sum_{r=1..W} r
    ok = True
    for i in range(3):
        x = torch.tensor([float(i), float(i + 1), float(i + 2)])
        rid = cond.submit({"x": x})
        results = cond.run_until_idle()
        y = results[rid][0]
        expected = x * scale
        match = torch.equal(y, expected)
        ok &= match
        print(f"request {rid}: x={x.tolist()} -> y={y.tolist()} "
              f"(expected {expected.tolist()}) {'OK' if match else 'MISMATCH'}")
        cond.results.clear()

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=3)
    for wid in ids:  # surface any worker startup crash
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")
    print("\nTP CONDUCTOR WIRING OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
