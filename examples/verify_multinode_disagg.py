"""The decentralized runtime across NODES: the same self-driving workers and
per-request coordinator, but with the mesh split over two simulated nodes —
separate socket dirs (no shared ipc namespace) joined only by TCP endpoints.

Topology (A --s--> B streaming, multi-step; B emits per chunk):

    node1: coordinator (tcp-bound) + wA (ipc)      node2: wB (tcp-bound)
      seeds:    coordinator -> wA  same-node  -> SHM descriptors
                coordinator -> wB  cross-node -> tensor bytes INLINE
      stream:   wA -> wB           cross-node -> INLINE  (peer-to-peer)
      emission: wB -> coordinator  cross-node -> INLINE

Every consumer asserts it never maps a cross-node /dev/shm arena (both
"nodes" share this host, so a leaked SHM read would otherwise pass silently).
TP lockstep needs no changes: tp_sched/finish/shutdown are plain msgpack and
ride the same TCP mailboxes.

    python examples/verify_multinode_disagg.py
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
NODE_MAP = {"coordinator": "node1", "wA": "node1", "wB": "node2"}
STEPS = 4


class TwoPartStream(Model):
    """A streams its input one element per loop iteration; B adds 1 and emits
    per chunk — a multi-chunk cross-node stream, not a one-shot tensor."""

    def walks(self):
        return {
            "a": loop(
                "a_loop",
                node("A", ["x"], [
                    stream_edge("B", "s", "B"),
                    edge("A", "x"),   # tail feeds the next iteration
                ]),
                max_iters=10_000,
            ),
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
        # B re-arms per stream chunk until the producer's done signal has
        # crossed (for wB, over TCP) and the buffer is drained.
        if walk == "b" and not stream_done:
            return "b", []
        return None

    def loops_to_finish(self):
        stops, self._stops = self._stops, []
        return stops

    def __init__(self) -> None:
        self._stops: list = []

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            if node_name == "A":
                x = named["x"][0]
                out[rid] = {"s": [x[:1] * 10], "x": [x[1:]]}
                if x.numel() <= 1:
                    self._stops.append((rid, "a_loop"))
            else:
                out[rid] = {"out": [named["s"][0] + 1]}
        return out

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def _guard_shm(my_id: str) -> None:
    """Make a cross-node SHM read FAIL LOUDLY: both simulated nodes share this
    host, so a descriptor leak would otherwise read successfully."""
    from mstar_rs.dist import ShmPool

    orig = ShmPool.read

    def read(self, desc):
        if isinstance(desc, list):
            owner = ShmPool.entity_of(desc[0])
            if NODE_MAP.get(owner) != NODE_MAP.get(my_id):
                raise AssertionError(
                    f"{my_id} ({NODE_MAP.get(my_id)}) read cross-node SHM of "
                    f"{owner} ({NODE_MAP.get(owner)}) — should have been inline")
        return orig(self, desc)

    ShmPool.read = read


def worker_main(worker_id: str, local_partitions: list, socket_dir: str,
                endpoints: dict) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        _guard_shm(worker_id)
        DisaggWorker(worker_id, TwoPartStream(), local_partitions,
                     PARTITION_TO_WORKER, socket_dir, device="cpu",
                     node_map=NODE_MAP, endpoints=endpoints).run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    node1_dir = tempfile.mkdtemp(prefix="mstar_rs_mn_node1_")
    node2_dir = tempfile.mkdtemp(prefix="mstar_rs_mn_node2_")

    # Entities receiving cross-node traffic bind TCP; wA (only reached by the
    # same-node coordinator) keeps the ipc scheme. Fixed localhost ports.
    endpoints = {
        "coordinator": "tcp://127.0.0.1:29881",   # takes wB's emissions
        "wB": "tcp://127.0.0.1:29882",            # takes seeds + wA's stream
    }

    specs = [("wA", ["A"], node1_dir), ("wB", ["B"], node2_dir)]
    workers = []
    for wid, parts, sdir in specs:
        p = ctx.Process(target=worker_main, args=(wid, parts, sdir, endpoints),
                        daemon=True)
        p.start()
        workers.append(p)

    # wA binds ipc (file appears); wB binds tcp (no file) — give it a beat.
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{node1_dir}/wA.ipc").exists():
        time.sleep(0.05)
    time.sleep(0.5)

    from mstar_rs.dist import DisaggCoordinator

    _guard_shm("coordinator")
    cond = DisaggCoordinator(TwoPartStream(), PARTITION_TO_WORKER, node1_dir,
                             node_map=NODE_MAP, endpoints=endpoints)

    ok = True
    for i in range(3):
        x = torch.arange(1, STEPS + 1, dtype=torch.float32) + i * 100
        gid = cond.submit({"x": x})
        cond.run_until_idle()
        got = cond.results[gid]
        expected = [int(v * 10 + 1) for v in x.tolist()]
        match = got == expected
        ok &= match
        print(f"request {gid}: x={x.tolist()} -> {got} "
              f"{'OK' if match else f'MISMATCH (expected {expected})'}")

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for wid, _parts, sdir in specs:
        errf = Path(f"{sdir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")
            ok = False

    print(f"\nMULTI-NODE DISAGG {'OK' if ok else 'FAILED'} "
          f"(two socket-dir 'nodes' joined by TCP; cross-node seeds/streams/"
          f"emissions inline, same-node SHM; no cross-node /dev/shm reads)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
