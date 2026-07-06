"""End-to-end streaming demo: the Rust (axum) frontend in front of the Python
conductor, over the `mstar-comm` Mailbox mesh.

    HTTP  →  axum frontend (Rust)  →  conductor (Python)  →  worker (Python)
             tokenize + SSE            model policy           model.execute
                    ↑___________ token-id stream ______________↓

The frontend tokenizes the prompt (HF `tokenizers`, off the GIL), submits the
token-ids to the conductor, and streams the generated token-ids back —
detokenizing incrementally and serializing SSE, all in Rust. The conductor
only runs model policy per forward pass. The model here is `EchoAR` (echoes
the prompt tokens one per step), so the streamed completion should reassemble
the prompt — proving the whole seam with no GPU or weights.

    python examples/serve_bridge.py <tokenizer.json> [port]

Requires the release frontend binary (built once):

    cargo build --release -p mstar-server
"""

from __future__ import annotations

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
        from mstar_rs.models.echo import EchoAR

        Worker(worker_id, EchoAR(), socket_dir, device="cpu").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


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
    while time.time() < deadline and not Path(f"{socket_dir}/worker_0.sock").exists():
        time.sleep(0.1)
    print(f"worker bound: {Path(f'{socket_dir}/worker_0.sock').exists()}")

    # 2) conductor bridge (drives the Runtime, streams tokens to the frontend)
    from mstar_rs.dist import Conductor
    from mstar_rs.models.echo import EchoAR

    cond = Conductor(EchoAR(), node_to_worker={"step": "worker_0"}, socket_dir=socket_dir)
    serve_thread = threading.Thread(target=cond.serve_frontend, daemon=True)
    serve_thread.start()
    print("conductor serving")

    # 3) Rust axum frontend, pointed at the same socket dir
    server = subprocess.Popen(
        [str(binary), tokenizer_path, str(port), socket_dir],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    # wait for it to come up
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

    # 4) drive a streaming request through the whole stack and reassemble it
    ok = False
    try:
        prompt = "The quick brown fox jumps over the lazy dog."
        body = f'{{"messages":[{{"role":"user","content":{_json(prompt)}}}],"max_tokens":32,"stream":true}}'
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        pieces = []
        with urllib.request.urlopen(req, timeout=15) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                import json

                delta = json.loads(payload)["choices"][0]["delta"]
                if "content" in delta:
                    pieces.append(delta["content"])
        got = "".join(pieces)
        print(f"\nprompt     : {prompt!r}")
        print(f"streamed   : {got!r}")
        # EchoAR echoes prompt tokens (capped at max_tokens); the reassembled
        # stream is a prefix of the prompt's detokenization.
        ok = got.strip() != "" and prompt.startswith(got.strip()[:10])
    finally:
        server.terminate()
        cond.stop()
        serve_thread.join(timeout=2)  # exit the serving loop before teardown
        cond.shutdown_workers()
        worker.join(timeout=3)

    print("\nEND-TO-END FRONTEND↔CONDUCTOR STREAMING OK" if ok else "\nFAILED")
    return 0 if ok else 1


def _json(s: str) -> str:
    import json

    return json.dumps(s)


if __name__ == "__main__":
    sys.exit(main())
