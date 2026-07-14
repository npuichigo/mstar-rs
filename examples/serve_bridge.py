"""End-to-end streaming demo: the Rust (axum) frontend in front of the Python
conductor, over the `mstar-comm` ZmqCommunicator mesh.

    HTTP  →  axum frontend (Rust)  →  conductor (Python)  →  worker (Python)
             adapt + media + SSE      model policy            model.execute
                    ↑______________ ResultChunk stream ___________↓

A faithful port of mstar's api_server surface: the frontend flattens the
request (text + media + modalities + model_kwargs) and submits it; the
conductor's policy tokenizes and drives the model; emissions stream back as
`ResultChunk`s ({modality, data, metadata}), detokenized at the frontend seam,
and the frontend serializes them (SSE / WAV / NDJSON). The model here is the
weightless `EchoAR` (echoes the prompt tokens one per step), so the reassembled
stream is a prefix of the prompt — proving the whole seam with no GPU/weights.

Echo has no OpenAI adapter, so we drive the model-agnostic `/generate` endpoint
(NDJSON), exactly as mstar's native path does.

    python examples/serve_bridge.py <tokenizer.json> [port]

Requires the release frontend binary (built once):

    cargo build --release -p mstar-server
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def worker_main(worker_id: str, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models.echo import EchoEngine

        # The worker holds the ENGINE (weights + execute).
        Worker(worker_id, EchoEngine(), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _multipart(fields: dict[str, str]) -> tuple[bytes, dict[str, str]]:
    """Hand-rolled multipart/form-data body for the /generate endpoint."""
    boundary = "----mstarrsboundary"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/serve_bridge.py <tokenizer.json> [port]")
        return 2
    tokenizer_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    binary = REPO / "target" / "release" / "mstar-server"
    if not binary.exists():
        print(f"missing {binary}; run: cargo build --release -p mstar-server")
        return 2

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_serve_")

    # 1) worker (runs EchoAR)
    worker = ctx.Process(target=worker_main, args=("worker_0", socket_dir), daemon=True)
    worker.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/worker_0.ipc").exists():
        time.sleep(0.1)
    print(f"worker bound: {Path(f'{socket_dir}/worker_0.ipc').exists()}")

    # 2) conductor bridge — the policy tokenizes the text submit and
    #    detokenizes the echoed ids back at the frontend seam.
    from transformers import PreTrainedTokenizerFast

    from mstar_rs.dist import Conductor
    from mstar_rs.models.echo import EchoPolicy

    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    cond = Conductor(EchoPolicy(tokenizer=tok), node_to_worker={"step": "worker_0"},
                     socket_dir=socket_dir)
    serve_thread = threading.Thread(target=cond.serve_frontend, daemon=True)
    serve_thread.start()
    print("conductor serving")

    # 3) Rust axum frontend (model name is arbitrary for the /generate path)
    server = subprocess.Popen(
        [str(binary), "echo", str(port), socket_dir],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    import urllib.request

    up = False
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            up = True
            break
        except Exception:
            time.sleep(0.1)
    print(f"frontend up: {up}")

    # 4) drive a streaming /generate request through the whole stack, reassemble
    ok = False
    try:
        prompt = "The quick brown fox jumps over the lazy dog."
        body, headers = _multipart({
            "text": prompt,
            "output_modalities": "text",
            "streaming": "true",
        })
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate", data=body, headers=headers,
        )
        pieces = []
        with urllib.request.urlopen(req, timeout=15) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("modality") == "text":
                    pieces.append(base64.b64decode(chunk["data"]).decode("utf-8", "replace"))
        got = "".join(pieces)
        print(f"\nprompt     : {prompt!r}")
        print(f"streamed   : {got!r}")
        # EchoAR echoes prompt tokens; the reassembled stream is a prefix of the
        # prompt's detokenization.
        ok = got.strip() != "" and prompt.startswith(got.strip()[:10])
    finally:
        server.terminate()
        cond.stop()
        serve_thread.join(timeout=2)  # exit the serving loop before teardown
        cond.shutdown_workers()
        worker.join(timeout=3)

    print("\nEND-TO-END FRONTEND↔CONDUCTOR MULTIMODAL BRIDGE OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
