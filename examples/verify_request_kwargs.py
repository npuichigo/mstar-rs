"""Verify per-request model_kwargs plumbing (mstar's model_kwargs-at-ingest)
across both runtimes — the value must reach the ENGINE before its first
execute, and be released at finish:

  1. in-process Driver          (register in submit, release at finish)
  2. decentralized DisaggWorker (kwargs ride the seed; release on finish)

The toy model echoes its prompt tokens shifted by a per-request `offset` model
kwarg — so the OUTPUT proves the kwarg reached execute (a dropped kwarg yields
offset 0 and the check fails). CPU-only, no weights.

    python examples/verify_request_kwargs.py
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


class OffsetEcho(Model):
    """Echoes prompt tokens + a per-request `offset` model kwarg, one per step.
    register_request/release_request implement the ModelEngine hooks; execute
    reads the stashed offset — exactly what real engines do with sampling
    config/voice."""

    def __init__(self) -> None:
        self._offsets: dict = {}
        self.registered: list = []   # (rid, kwargs) audit trail
        self.released: list = []

    # -- policy --
    def walks(self):
        return {
            "gen": sequential(node("step", ["state"], [
                emit("token", modality="text"),
                edge(EMPTY_DESTINATION, "rest", persist=True),
            ]))
        }

    def initial_inputs(self, request):
        tokens = list(request.get("tokens") or [1, 2, 3])
        return "gen", [("step", "state", [torch.tensor(tokens, dtype=torch.int64)])]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        rest = persist["rest"][0]
        return None if rest.numel() == 0 else ("gen", [("step", "state", [rest])])

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())

    # -- engine (the hooks under test) --
    def register_request(self, request_id, model_kwargs):
        self._offsets[request_id] = int((model_kwargs or {}).get("offset", 0))
        self.registered.append((request_id, dict(model_kwargs or {})))

    def release_request(self, request_id):
        self._offsets.pop(request_id, None)
        self.released.append(request_id)

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            off = self._offsets.get(rid, 0)   # 0 = the kwarg never arrived
            out[rid] = {"token": [state[:1] + off], "rest": [state[1:]]}
        return out


def check(label: str, got: list, want: list) -> bool:
    ok = got == want
    print(f"  {label}: {got} (want {want}) {'OK' if ok else 'FAIL'}")
    return ok


def run_driver() -> bool:
    from mstar_rs.driver import Driver

    m = OffsetEcho()
    d = Driver(m)
    rid = d.submit({"tokens": [10, 20, 30], "model_kwargs": {"offset": 100}})
    rid2 = d.submit({"tokens": [1, 2]})   # no kwargs -> offset 0
    results = d.run_until_idle()
    ok = check("driver kwargs", results[rid], [110, 120, 130])
    ok &= check("driver default", results[rid2], [1, 2])
    ok &= check("driver released", sorted(m.released), sorted([rid, rid2]))
    return ok


PARTS = [{"name": "P", "walks": ["gen"]}]


class PartitionedOffsetEcho(OffsetEcho):
    """Single named partition so the decentralized coordinator can route."""

    def partitions(self):
        return (PARTS, [])


def _disagg_worker_main(socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        DisaggWorker("w0", PartitionedOffsetEcho(), ["P"], {"P": "w0"},
                     socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/w0.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def run_disagg() -> bool:
    from mstar_rs.dist import DisaggCoordinator

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_kwargs_dis_")
    p = ctx.Process(target=_disagg_worker_main, args=(socket_dir,), daemon=True)
    p.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/w0.ipc").exists():
        time.sleep(0.1)

    cond = DisaggCoordinator(PartitionedOffsetEcho(), {"P": "w0"}, socket_dir)
    gid = cond.submit({"tokens": [10, 20, 30], "model_kwargs": {"offset": 100}})
    gid2 = cond.submit({"tokens": [1, 2]})
    results = cond.run_until_idle()
    cond.shutdown_workers()
    p.join(timeout=3)
    ok = check("disagg kwargs", results[gid], [110, 120, 130])
    ok &= check("disagg default", results[gid2], [1, 2])
    return ok


def main() -> int:
    print("[1/2] in-process Driver")
    ok = run_driver()
    print("[2/2] decentralized DisaggWorker + Coordinator")
    ok &= run_disagg()
    print(f"\nPER-REQUEST MODEL_KWARGS {'OK' if ok else 'FAILED'} "
          f"(register-at-ingest -> engine execute -> release-at-finish, both runtimes)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
