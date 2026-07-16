"""Phase 2 of the distributed in-worker scheduler: the decentralized runtime
across PROCESSES. Two worker processes each own one partition and drive their
own loop (a per-partition Driver); partition A's stream output is shipped
peer-to-peer straight to worker B (SHM descriptor + ZMQ notify), never through
a scheduler. A thin coordinator only ingests the request and collects the
emission — NO per-step conductor round-trip in the hot loop.

Contrast the centralized alternative (a conductor dispatching a batch to a worker
over ZMQ every step). CPU + toy, so it runs anywhere.

    python examples/verify_disagg.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Model  # noqa: E402
from mstar_rs.graph import connection, emit, fixed_chunk, node, partition, stream_edge  # noqa: E402

PARTITION_TO_WORKER = {"A": "wA", "B": "wB"}


class TwoPart(Model):
    """A --s--> B. A scales x by 10 and streams it; B adds 1 and emits."""

    def walks(self):
        return {
            "a": node("A", ["x"], [stream_edge("B", "s", "B")]),
            "b": node("B", ["s"], [emit("out", modality="tensor", persist=True)]),
        }

    def partitions(self):
        return (
            [partition("A", ["a"]), partition("B", ["b"])],
            [connection("A", "B", "s", fixed_chunk(1))],
        )

    def initial_walks(self, request):
        return [("a", [("A", "x", [request["x"]])]), ("b", [])]

    def next_forward(self, rid, part, walk, fwd_index, persist, stream_done):
        return None

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            if node_name == "A":
                out[rid] = {"s": [named["x"][0] * 10]}
            else:
                out[rid] = {"out": [named["s"][0] + 1]}
        return out

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def worker_main(worker_id: str, local_partitions: list, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        DisaggWorker(worker_id, TwoPart(), local_partitions, PARTITION_TO_WORKER,
                     socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_disagg_")
    specs = [("wA", ["A"]), ("wB", ["B"])]

    workers = []
    for wid, parts in specs:
        p = ctx.Process(target=worker_main, args=(wid, parts, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    deadline = time.time() + 15
    need = {f"{socket_dir}/{w}.ipc" for w, _ in specs}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        time.sleep(0.05)
    print(f"workers bound: {[w for w, _ in specs if Path(f'{socket_dir}/{w}.ipc').exists()]}")

    from mstar_rs.dist import DisaggCoordinator

    cond = DisaggCoordinator(TwoPart(), PARTITION_TO_WORKER, socket_dir)

    ok = True
    for i in range(3):
        x = torch.tensor([float(i + 1)])
        gid = cond.submit({"x": x})
        cond.run_until_idle()
        got = cond.results[gid]
        expected = [int((x[0] * 10 + 1).item())]
        match = got == expected
        ok &= match
        print(f"request {gid}: x={x.tolist()} -> {got} (expected {expected}) "
              f"{'OK' if match else 'MISMATCH'}")

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=3)
    for wid, _ in specs:
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")

    print(f"\nDISAGG (decentralized multi-process) {'OK' if ok else 'FAILED'} "
          f"(per-partition self-driving workers, peer-to-peer streaming, no per-step conductor)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
