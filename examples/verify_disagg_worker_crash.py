"""A decentralized worker CRASHING must fail its requests, not hang the stack.

Before this, a DisaggWorker that died mid-request (engine exception, OOM,
segfault-adjacent) simply never reported partition_done — the coordinator
(and any HTTP client behind it) waited forever. Now the dying worker reports
`worker_error` with the gids it carried; the coordinator fails those requests
(errors dict / frontend 500), tells surviving leaders to drop them, and
refuses new submits that would seed the dead worker.

Sequence: a good request completes; a poison request (token 13) kills the
worker's engine; the coordinator must (a) finish+error the poison request so
run_until_idle terminates, (b) keep the good result, (c) reject the next
submit with a clear "worker died" error.

    python examples/verify_disagg_worker_crash.py
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
from mstar_rs.graph import EMPTY_DESTINATION, edge, emit, node, sequential  # noqa: E402

PARTS = [{"name": "P", "walks": ["gen"]}]
POISON = 13


class CrashyEcho(Model):
    """Echoes tokens one per step; seeing POISON raises (killing the worker)."""

    def partitions(self):
        return (PARTS, [])

    def walks(self):
        return {
            "gen": sequential(node("step", ["state"], [
                emit("token", modality="text"),
                edge(EMPTY_DESTINATION, "rest", persist=True),
            ]))
        }

    def initial_inputs(self, request):
        tokens = list(request["tokens"])
        return "gen", [("step", "state", [torch.tensor(tokens, dtype=torch.int64)])]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        rest = persist["rest"][0]
        return None if rest.numel() == 0 else ("gen", [("step", "state", [rest])])

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            if int(state[0]) == POISON:
                raise RuntimeError("poison token: worker down")
            out[rid] = {"token": [state[:1]], "rest": [state[1:]]}
        return out


def worker_main(socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        DisaggWorker("w0", CrashyEcho(), ["P"], {"P": "w0"}, socket_dir,
                     device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/w0.err", "w") as f:
            traceback.print_exc(file=f)
        # expected for the poison request — the worker dies loudly


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_crash_")
    p = ctx.Process(target=worker_main, args=(socket_dir,), daemon=True)
    p.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/w0.ipc").exists():
        time.sleep(0.1)

    from mstar_rs.dist import DisaggCoordinator

    cond = DisaggCoordinator(CrashyEcho(), {"P": "w0"}, socket_dir)

    # The good request completes BEFORE the poison is submitted (a request
    # in flight when the worker dies is correctly failed along with it).
    good = cond.submit({"tokens": [1, 2, 3]})
    results = cond.run_until_idle()

    poison = cond.submit({"tokens": [POISON, 99]})
    # run_until_idle must TERMINATE (the crash report finishes the poison gid).
    t0 = time.time()
    results = cond.run_until_idle()
    dt = time.time() - t0

    ok_good = results.get(good) == [1, 2, 3]
    ok_err = poison in cond.errors and "poison" in cond.errors[poison]
    print(f"  good request:   {results.get(good)} {'OK' if ok_good else 'FAIL'}")
    print(f"  poison request: errors[{poison}]={cond.errors.get(poison)!r} "
          f"{'OK' if ok_err else 'FAIL'} (idle in {dt:.1f}s — no hang)")

    # New submits must be refused with a clear error, not silently queued.
    try:
        cond.submit({"tokens": [4]})
        ok_refuse = False
        print("  post-crash submit: accepted FAIL")
    except RuntimeError as e:
        ok_refuse = "died" in str(e)
        print(f"  post-crash submit: refused ({e}) {'OK' if ok_refuse else 'FAIL'}")

    p.join(timeout=5)
    ok = ok_good and ok_err and ok_refuse
    print(f"\nDISAGG WORKER-CRASH HANDLING {'OK' if ok else 'FAILED'} "
          f"(crash fails its requests; no hang; new submits refused)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
