"""End-to-end serving demo: HTTP -> conductor -> worker processes -> HTTP.

Spawns worker processes, stands up the FastAPI server co-located with the
conductor, then (self-test mode) drives a few HTTP requests through the whole
multi-process stack and verifies them.

    python examples/serve.py            # self-test (spawns workers, serves, checks, exits)
    python examples/serve.py --serve    # run the server until Ctrl-C (POST /generate)

CPU toy model, so no GPU/weights needed — the point is the full path:
HTTP request -> serving engine -> UDS dispatch -> worker execute -> SHM
tensors -> result streamed back over HTTP.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Model, edge, emit, node, sequential  # noqa: E402


class ToyDisagg(Model):
    """y = 2*x on worker_0, z = y + 1 (emitted) on worker_1."""

    def walks(self):
        return {
            "fwd": sequential(
                node("a", ["x"], [edge("b", "y")]),
                node("b", ["y"], [emit("z", modality="tensor", persist=True)]),
            )
        }

    def initial_inputs(self, request):
        x = torch.as_tensor(request["x"], dtype=torch.float32)  # JSON list -> tensor
        return "fwd", [("a", "x", [x])]

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            out[rid] = {"y": [named["x"][0] * 2]} if node_name == "a" else {"z": [named["y"][0] + 1]}
        return out

    def postprocess(self, name, modality, tensors):
        return tensors[0]


def worker_main(worker_id: str, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker

        Worker(worker_id, ToyDisagg(), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def spawn_workers(socket_dir: str):
    ctx = mp.get_context("spawn")
    procs = []
    for wid in ("worker_0", "worker_1"):
        p = ctx.Process(target=worker_main, args=(wid, socket_dir), daemon=True)
        p.start()
        procs.append(p)
    deadline = time.time() + 15
    need = [f"{socket_dir}/worker_0.sock", f"{socket_dir}/worker_1.sock"]
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        time.sleep(0.1)
    return procs


def build_engine(socket_dir: str):
    from mstar_rs.dist import Conductor
    from mstar_rs.server import ServingEngine, build_app

    cond = Conductor(ToyDisagg(), node_to_worker={"a": "worker_0", "b": "worker_1"},
                     socket_dir=socket_dir)
    engine = ServingEngine(cond)
    return engine, build_app(engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="run until Ctrl-C")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_serve_")
    spawn_workers(socket_dir)
    engine, app = build_engine(socket_dir)
    engine.start()

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    if args.serve:
        print(f"serving on http://127.0.0.1:{args.port}  (POST /generate {{\"x\": [..]}})")
        server.run()
        return 0

    # Self-test: run the server in a thread, drive HTTP requests, verify.
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import requests

    base = f"http://127.0.0.1:{args.port}"
    for _ in range(100):
        try:
            if requests.get(f"{base}/health", timeout=1).json()["status"] == "ok":
                break
        except Exception:
            time.sleep(0.1)

    ok = True
    for i in range(4):
        x = [float(i), float(i + 1), float(i + 2)]
        r = requests.post(f"{base}/generate", json={"x": x}, timeout=30).json()
        got = r["result"][0]
        expected = [v * 2 + 1 for v in x]
        match = got == expected
        ok &= match
        print(f"POST /generate x={x} -> {got} (expected {expected}) {'OK' if match else 'MISMATCH'}")

    server.should_exit = True
    engine.stop()
    print("\nHTTP SERVING END-TO-END OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
