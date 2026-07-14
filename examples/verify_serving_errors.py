"""Error propagation through the full serving stack — the coverage-parity
behaviors mstar's api_server has and a hung HTTP call doesn't:

  1. worker exception mid-request  -> streaming: a terminal error event;
                                      non-streaming: HTTP 500 envelope
  2. ingest failure (bad request)  -> same, without killing the serve loop
  3. a healthy request AFTER the failures still works (isolation)

Toy character-echo model: text becomes ord() tokens, echoed one per step; a
prompt starting with '!' makes the WORKER raise mid-generation, an empty
prompt fails at INGEST (policy raises while seeding). Driven over real HTTP
through the axum binary -> bridge -> conductor -> worker.

    python examples/verify_serving_errors.py [port]
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from mstar_rs import Model  # noqa: E402
from mstar_rs.graph import EMPTY_DESTINATION, edge, emit, node, sequential  # noqa: E402


class _CharTokenizer:
    """decode(ids) -> ''.join(chr) — just enough for the conductor's detok."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(int(i)) for i in ids)


class CharEcho(Model):
    """Echo text as characters; '!'-prefixed text raises in the WORKER,
    empty text raises at INGEST."""

    tokenizer = _CharTokenizer()

    def walks(self):
        return {
            "gen": sequential(node("step", ["state"], [
                emit("token", modality="text"),
                edge(EMPTY_DESTINATION, "rest", persist=True),
            ]))
        }

    def initial_inputs(self, request):
        text = request.get("text") or ""
        if not text:
            raise ValueError("empty prompt")   # ingest-time failure
        ids = [ord(c) for c in text]
        return "gen", [("step", "state", [torch.tensor(ids, dtype=torch.int64)])]

    def next_forward(self, request_id, partition, walk, fwd_index, persist, stream_done):
        rest = persist["rest"][0]
        return None if rest.numel() == 0 else ("gen", [("step", "state", [rest])])

    def postprocess(self, name, modality, tensors):
        return int(tensors[0].item())

    def execute(self, node_name, walk, inputs, kv=None):
        out = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            if int(state[0]) == ord("!"):
                raise RuntimeError("worker failure requested by prompt")
            out[rid] = {"token": [state[:1]], "rest": [state[1:]]}
        return out


def worker_main(socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker

        Worker("worker_0", CharEcho(), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/worker_0.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _generate(port: int, text: str, streaming: bool):
    """POST /generate; returns (status, ndjson_lines | json_body)."""
    boundary = "----errcase"
    fields = {"text": text, "output_modalities": "text",
              "streaming": "true" if streaming else "false"}
    body = "".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
        for k, v in fields.items()
    ) + f"--{boundary}--\r\n"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
            return r.status, lines
    except urllib.error.HTTPError as e:
        return e.code, [json.loads(e.read().decode() or "{}")]


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8157
    binary = REPO / "target" / "release" / "mstar-server"
    if not binary.exists():
        print(f"missing {binary}; run: cargo build --release -p mstar-server")
        return 2

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_errs_")
    worker = ctx.Process(target=worker_main, args=(socket_dir,), daemon=True)
    worker.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/worker_0.ipc").exists():
        time.sleep(0.1)

    from mstar_rs.dist import Conductor

    cond = Conductor(CharEcho(), node_to_worker={"step": "worker_0"}, socket_dir=socket_dir)
    serve_thread = threading.Thread(target=cond.serve_frontend, daemon=True)
    serve_thread.start()

    server = subprocess.Popen([str(binary), "echo", str(port), socket_dir])
    up = False
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            up = True
            break
        except Exception:
            time.sleep(0.1)
    print(f"frontend up: {up}")

    ok = True
    try:
        # 1) worker exception, streaming -> a terminal {"error": ...} line
        status, lines = _generate(port, "!boom", streaming=True)
        has_err = any("error" in l for l in lines)
        print(f"  worker-error/stream: status={status} lines={lines!r} "
              f"{'OK' if has_err else 'FAIL'}")
        ok &= has_err

        # 2) worker exception, non-streaming -> HTTP 500 envelope
        status, body = _generate(port, "!boom", streaming=False)
        env = body[0].get("error", {}) if body else {}
        good = status == 500 and "worker failure" in env.get("message", "")
        print(f"  worker-error/collect: status={status} error={env.get('message')!r} "
              f"{'OK' if good else 'FAIL'}")
        ok &= good

        # 3) ingest failure (empty prompt) -> error, serve loop survives
        status, body = _generate(port, "", streaming=False)
        env = body[0].get("error", {}) if body else {}
        good = status == 500 and "empty prompt" in env.get("message", "")
        print(f"  ingest-error: status={status} error={env.get('message')!r} "
              f"{'OK' if good else 'FAIL'}")
        ok &= good

        # 4) a healthy request after all failures round-trips
        status, lines = _generate(port, "fine", streaming=True)
        text = "".join(
            base64.b64decode(l["data"]).decode() for l in lines if l.get("modality") == "text"
        )
        good = status == 200 and text == "fine"
        print(f"  recovery: status={status} text={text!r} {'OK' if good else 'FAIL'}")
        ok &= good
    finally:
        server.terminate()
        cond.stop()
        serve_thread.join(timeout=2)
        cond.shutdown_workers()
        worker.join(timeout=3)

    print(f"\nSERVING ERROR PROPAGATION {'OK' if ok else 'FAILED'} "
          f"(worker + ingest failures -> HTTP 500 / stream error; loop survives)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
