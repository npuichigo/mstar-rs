"""The async execution pipeline, proven on CPU: a DisaggWorker with
`async_pipeline=True` runs `model.execute` on a dedicated thread while the
host thread handles messages, and pipelines batches via speculation-claim
(mstar's worker run loop: pending batch + speculate N+1 + submit-asap +
eventfd wakeup).

The engine sleeps per step (a stand-in for GPU compute). Assertions target
the MECHANISM, not throughput (a sleeping thread yields no CPU speedup):

  1. correctness — an AR echo round-trips under the async loop;
  2. speculation — nearly every decode step is a CLAIMED speculation
     (`spec_hits`), i.e. the follow-up launched without a scheduler scan;
  3. eventfd wakeup — makespan ~= steps x sleep. Without the wake, every
     step would eat up to the 50 ms poll timeout on top of its sleep and
     the makespan would blow up ~2.5x — the 'silent stall' failure mode;
  4. liveness — a request submitted mid-flight is ingested during the
     in-flight batch and completes (host work overlaps execute).

    python examples/verify_async_pipeline.py
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
from mstar_rs.graph import edge, emit, loop, node  # noqa: E402

PARTS = [{"name": "P", "walks": ["gen"]}]
STEP_SLEEP = 0.03
STEPS = 12


class SlowLoopEcho(Model):
    """AR echo as a GRAPH loop (the qwen3/orpheus decode shape — what
    speculation targets): `step` self-feeds `state`, emitting the head each
    iteration; the ENGINE signals the loop stop when a request's tokens run
    out (mstar's check_stop -> STOP_LOOPS). Each step sleeps (fake GPU)."""

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
        return None  # the graph loop drives iteration; stop comes from execute

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())

    def loops_to_finish(self):
        stops, self._pending_stops = self._pending_stops, []
        return stops

    def execute(self, node_name, walk, inputs, kv=None):
        time.sleep(STEP_SLEEP)
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            out[rid] = {"token": [state[:1]], "state": [state[1:]]}
            if state.numel() <= 1:  # last token: stop this request's loop
                self._pending_stops.append((rid, "gen_loop"))
        return out


def worker_main(socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        w = DisaggWorker("w0", SlowLoopEcho(), ["P"], {"P": "w0"}, socket_dir,
                         device="cpu", async_pipeline=True)
        try:
            w.run()
        finally:
            with open(f"{socket_dir}/w0.spec", "w") as f:
                f.write(str(w.spec_hits))
    except Exception:
        import traceback

        with open(f"{socket_dir}/w0.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_async_")
    p = ctx.Process(target=worker_main, args=(socket_dir,), daemon=True)
    p.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/w0.ipc").exists():
        time.sleep(0.1)

    from mstar_rs.dist import DisaggCoordinator

    cond = DisaggCoordinator(SlowLoopEcho(), {"P": "w0"}, socket_dir)

    # Request A starts; request B lands MID-FLIGHT (liveness under overlap).
    t0 = time.time()
    a = cond.submit({"tokens": [100 + i for i in range(STEPS)]})
    b_submitted = False
    b = None
    while len(cond.finished) < (2 if b_submitted else 1):
        cond.poll(timeout_ms=5)
        if not b_submitted and time.time() - t0 > STEP_SLEEP * 2.5:
            b = cond.submit({"tokens": [900, 901, 902]})
            b_submitted = True
    wall = time.time() - t0

    cond.shutdown_workers()
    p.join(timeout=5)
    if (errf := Path(f"{socket_dir}/w0.err")).exists():
        print(f"--- worker error ---\n{errf.read_text()}")
        return 1
    spec_hits = int(Path(f"{socket_dir}/w0.spec").read_text() or "0")

    ok_a = cond.results.get(a) == [100 + i for i in range(STEPS)]
    ok_b = cond.results.get(b) == [900, 901, 902]
    # Steps that ran: A's 12 (some co-batched with B's 3). Speculation should
    # cover nearly every continuation (first batch of each chain can't be a
    # claim; co-batching changes rid sets, breaking some chains).
    ok_spec = spec_hits >= STEPS - 4
    # eventfd wake: makespan near the serial sleep budget, far below the
    # +50ms-per-step no-wake regime.
    budget = (STEPS + 3) * STEP_SLEEP
    ok_wall = wall < budget * 1.8
    print(f"  A round-trip : {ok_a}")
    print(f"  B (mid-flight) round-trip : {ok_b}")
    print(f"  speculation claims: {spec_hits} (need >= {STEPS - 4})")
    print(f"  makespan: {wall*1000:.0f}ms (sleep budget {budget*1000:.0f}ms, "
          f"no-wake regime would be ~{(budget + 0.05*STEPS)*1000:.0f}ms+)")

    ok = ok_a and ok_b and ok_spec and ok_wall
    print(f"\nASYNC PIPELINE {'OK' if ok else 'FAILED'} "
          f"(execute on a thread; speculation-claimed follow-ups; eventfd wake)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
