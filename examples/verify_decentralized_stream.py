"""Phase 1 of the distributed in-worker scheduler: two per-partition drivers,
each owning ONE partition and running its own loop, streaming to each other
peer-to-peer — with NO central conductor and no per-step round-trip.

Partition A's driver produces a stream chunk; instead of routing it through a
scheduler, the chunk is shipped straight into partition B's driver via
`inject` (mstar's peer-to-peer streaming into a worker's local ready-queue).
This is the runtime-level proof of the decentralized design: `set_local_
partitions` scopes each runtime to its partition, cross-partition stream outputs
surface as `stream_out` events to ship, and the consumer injects them locally.
(The worker/process + SHM/TCP transport wiring on top is phase 2.)

    python examples/verify_decentralized_stream.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver, Model  # noqa: E402
from mstar_rs.graph import (  # noqa: E402
    connection,
    emit,
    fixed_chunk,
    node,
    partition,
    stream_edge,
)


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
            else:  # B
                out[rid] = {"out": [named["s"][0] + 1]}
        return out

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def main() -> int:
    model = TwoPart()
    drvA = Driver(model, local_partitions=["A"])   # owns + schedules only A
    drvB = Driver(model, local_partitions=["B"])   # owns + schedules only B

    x = torch.tensor([5.0])
    rid = drvA.submit({"x": x})    # A seeded on drvA
    ridB = drvB.submit({"x": x})   # B seeded on drvB (waits on the stream)
    assert rid == ridB, (rid, ridB)

    # 1) drive A on its own loop (no conductor) -> emits a stream_out chunk
    drvA.run_until_idle()
    shipped = [(o[0], o[1], o[2], [int(t.item()) for t in o[3]]) for o in drvA.outbox]
    print(f"A shipped (stream_out): {shipped}")

    # 2) peer-to-peer: hand A's chunk straight to B's driver
    for frm, edge, to, tensors, r in drvA.outbox:
        drvB.inject(r, frm, edge, to, tensors)
    drvA.outbox.clear()

    # 3) drive B on its own loop -> consumes the chunk, emits out
    drvB.run_until_idle()
    got = drvB.results[rid]
    expected = [int((x[0] * 10 + 1).item())]   # A: 5*10=50 ; B: 50+1=51
    ok = got == expected and shipped and shipped[0][3] == [50]
    print(f"B result: {got} (expected {expected}) {'OK' if got == expected else 'MISMATCH'}")
    print(f"\nDECENTRALIZED STREAM {'OK' if ok else 'FAILED'} "
          f"(per-partition drivers, peer-to-peer streaming, no conductor / no per-step round-trip)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
