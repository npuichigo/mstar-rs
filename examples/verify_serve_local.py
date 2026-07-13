"""Verify the in-process serving path: ServingEngine driving a Driver (scheduler
+ engine co-located) instead of a multi-process Conductor.

The AR decode loop runs entirely in-process — next_batch -> execute ->
complete_batch -> next_forward are plain function calls, with NO ZeroMQ /
msgpack / SHM round-trip per step and no worker processes. This is the
co-located answer to the per-step conductor dispatch overhead (it mirrors
mstar's in-worker MicroScheduler). Two concurrent requests confirm continuous
batching still works (the runtime groups both requests' decode steps into one
batch).

    python examples/verify_serve_local.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver, Model, edge, emit, loop, node  # noqa: E402
from mstar_rs.server import ServingEngine  # noqa: E402

STEPS = 5  # decode tokens per request


class ToyAR(Model):
    """Trivial single-partition AR model: prefill emits the seed token, then a
    decode loop increments and emits, self-feeding, for STEPS tokens."""

    def __init__(self) -> None:
        self._count: dict[int, int] = {}
        self._stops: list[tuple[int, str]] = []

    def walks(self):
        return {
            "prefill": node("m", ["x"], [emit("tok", modality="tensor", persist=True)]),
            "decode": loop(
                "decode_loop",
                node("m", ["x"], [edge("m", "x"), emit("tok", modality="tensor", persist=True)]),
                max_iters=100,
            ),
        }

    def initial_walks(self, request):
        return [("prefill", [("m", "x", [request["x"]])])]

    def next_forward(self, rid, partition, walk, fwd_index, persist, stream_done):
        if walk == "prefill":
            return ("decode", [("m", "x", persist["tok"])])
        return None

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            nxt = named["x"][0] + 1
            o = {"tok": [nxt]}
            if walk == "decode":
                o["x"] = [nxt]
                c = self._count.get(rid, 0) + 1
                self._count[rid] = c
                if c >= STEPS:
                    self._stops.append((rid, "decode_loop"))
            out[rid] = o
        return out

    def loops_to_finish(self):
        s = self._stops
        self._stops = []
        return s

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())


def main() -> int:
    eng = ServingEngine(Driver(ToyAR(), max_batch_size=8))
    eng.start()

    results: dict[int, list] = {}

    def submit(seed: int) -> None:
        results[seed] = eng.submit({"x": torch.tensor([float(seed)])})

    # two concurrent requests — exercises continuous batching in-process
    threads = [threading.Thread(target=submit, args=(s,)) for s in (10, 100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    eng.stop()

    ok = True
    for seed in (10, 100):
        got = results[seed]
        expected = [seed + i for i in range(1, STEPS + 2)]  # prefill tok + STEPS decode toks
        match = got == expected
        ok &= match
        print(f"  seed={seed}: {got} (expected {expected}) {'OK' if match else 'MISMATCH'}")

    print(f"\nIN-PROCESS SERVING {'OK' if ok else 'FAILED'} "
          f"(Driver-backed ServingEngine: continuous batching, zero per-step IPC, no workers)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
