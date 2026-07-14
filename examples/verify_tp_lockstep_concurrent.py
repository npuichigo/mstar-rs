"""CONCURRENT requests through decentralized TENSOR-PARALLEL workers: the
leader-broadcasts-batch mechanism (mstar's ScheduleTPNode) must keep both TP
ranks running IDENTICAL batches in IDENTICAL order — the property NCCL
collectives depend on. Without it, per-rank schedulers diverge under
concurrency (rank0 batches {A} while rank1 batches {A,B}) and a real TP model
deadlocks in its first all-reduce.

Two DisaggWorker processes both own partition "P" (a fake TP pair, no
collectives — CPU): w0 is the leader (initiates + ships each batch decision),
w1 the follower (never self-initiates; replays). The follower's engine runs
SLOWER (deliberate skew) so its free-running scheduler WOULD have grouped
requests differently. Requests are submitted in staggered waves to create
mixed prefill/decode readiness. Each rank appends every executed batch
(walk + per-request first-token values, in engine iteration order) to a
file; the test asserts the two sequences are byte-identical, that
multi-request batches actually occurred, and that all outputs are correct.

    python examples/verify_tp_lockstep_concurrent.py
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
N_REQUESTS = 6


class RecordingEcho(Model):
    """Echoes prompt tokens one per step; records every executed batch."""

    def __init__(self, record_path: str, delay_s: float = 0.0) -> None:
        self._record_path = record_path
        self._delay_s = delay_s

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
        if self._delay_s:
            time.sleep(self._delay_s)   # skew: shift this rank's poll timing
        heads = [int(named["state"][0][0]) for named in inputs.values()]
        with open(self._record_path, "a") as f:
            f.write(f"{walk}:{','.join(map(str, heads))}\n")
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            out[rid] = {"token": [state[:1]], "rest": [state[1:]]}
        return out


def worker_main(worker_id: str, socket_dir: str, is_leader: bool) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        model = RecordingEcho(f"{socket_dir}/{worker_id}.batches",
                              delay_s=0.0 if is_leader else 0.004)
        kw = (dict(tp_nodes=["step"], tp_followers=["w1"]) if is_leader
              else dict(tp_follow_nodes=["step"]))
        DisaggWorker(worker_id, model, ["P"], {"P": "w0"}, socket_dir,
                     device="cpu", io_leader=is_leader, **kw).run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_tp_lockstep_")

    workers = []
    for wid, lead in (("w0", True), ("w1", False)):
        p = ctx.Process(target=worker_main, args=(wid, socket_dir, lead), daemon=True)
        p.start()
        workers.append(p)
    deadline = time.time() + 15
    need = {f"{socket_dir}/w0.ipc", f"{socket_dir}/w1.ipc"}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        time.sleep(0.1)

    from mstar_rs.dist import DisaggCoordinator

    # Both ranks are the TP group of partition P; seeds go to both.
    cond = DisaggCoordinator(RecordingEcho("/dev/null"),
                             {"P": ["w0", "w1"]}, socket_dir)

    # Staggered waves: later requests land while earlier ones are mid-decode,
    # so the leader's continuous batching forms changing groups.
    gids = []
    for i in range(N_REQUESTS):
        base = (i + 1) * 100
        gids.append(cond.submit({"tokens": [base + j for j in range(8)]}))
        if i % 2 == 1:
            t_end = time.time() + 0.02   # let a couple of decode steps happen
            while time.time() < t_end:
                cond.poll(timeout_ms=5)
    results = cond.run_until_idle()

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for wid in ("w0", "w1"):
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")
            return 1

    # 1) every request echoed fully and in order
    values_ok = all(
        results[g] == [(i + 1) * 100 + j for j in range(8)]
        for i, g in enumerate(gids)
    )
    # 2) both ranks executed byte-identical batch sequences
    seq0 = Path(f"{socket_dir}/w0.batches").read_text().splitlines()
    seq1 = Path(f"{socket_dir}/w1.batches").read_text().splitlines()
    identical = seq0 == seq1
    # 3) concurrency actually happened: some batch carried >= 2 requests
    max_bs = max(len(line.split(":")[1].split(",")) for line in seq0)

    print(f"[results] {N_REQUESTS} concurrent requests, all correct: {values_ok}")
    print(f"[lockstep] leader {len(seq0)} batches, follower {len(seq1)} — "
          f"identical: {identical}")
    print(f"[batching] max requests in one batch: {max_bs}")
    if not identical:
        for i, (a, b) in enumerate(zip(seq0, seq1)):
            if a != b:
                print(f"  first divergence at batch {i}: leader={a!r} follower={b!r}")
                break

    ok = values_ok and identical and max_bs >= 2
    print(f"\nTP LOCKSTEP UNDER CONCURRENCY {'OK' if ok else 'FAILED'} "
          f"(leader-broadcasts-batch: follower replays identical batches in order)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
