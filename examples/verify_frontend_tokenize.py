"""The frontend-tokenizes fast path, end to end: tokenize AND detokenize in
Rust (off the GIL), token ids on the wire both ways — the
Rust-default-with-model-override mechanism from RFC #130 Step 3.

    HTTP -> axum TOKENIZES (HF tokenizers crate) -> submit token ids
         -> coordinator seeds the model with ids (no Python tokenize)
         -> emissions ship back as raw ids (modality "token", 4-byte LE)
         -> axum DETOKENIZES incrementally -> NDJSON/collect text

The conductor's EchoPolicy is built WITHOUT a tokenizer — if any Python-side
tokenize/detok were still on this path, the run would fail, which is the
point: the flag moves that work wholly into the frontend. The model-side-tokenizes
default (multimodal-safe) coexists on the same wire — see
verify_serving_errors.py, where the coordinator's policy detokenizes.

    python examples/verify_frontend_tokenize.py <tokenizer.json> [port]
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def worker_main(worker_id: str, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker
        from mstar_rs.models.echo import EchoAR

        DisaggWorker(worker_id, EchoAR(), ["P"], {"P": worker_id},
                     socket_dir, device="cpu", coordinator_id="conductor").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _generate(port: int, text: str, streaming: bool):
    boundary = "----fetok"
    fields = {"text": text, "output_modalities": "text", "tokenize": "true",
              "streaming": "true" if streaming else "false"}
    body = "".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
        for k, v in fields.items()
    ) + f"--{boundary}--\r\n"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/verify_frontend_tokenize.py <tokenizer.json> [port]")
        return 2
    tokenizer_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8163
    binary = REPO / "target" / "release" / "mstar-server"
    if not binary.exists():
        print(f"missing {binary}; run: cargo build --release -p mstar-server")
        return 2

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_fetok_")
    worker = ctx.Process(target=worker_main, args=("worker_0", socket_dir), daemon=True)
    worker.start()
    deadline = time.time() + 15
    while time.time() < deadline and not Path(f"{socket_dir}/worker_0.ipc").exists():
        time.sleep(0.1)

    from mstar_rs.dist import DisaggCoordinator
    from mstar_rs.models.echo import EchoPolicy

    # NO tokenizer on the model side: any Python tokenize/detok would fail.
    cond = DisaggCoordinator(EchoPolicy(tokenizer=None), {"P": ["worker_0"]},
                             socket_dir, my_id="conductor")
    serve_thread = threading.Thread(target=cond.serve_frontend, daemon=True)
    serve_thread.start()

    env = dict(os.environ, MSTAR_TOKENIZER=tokenizer_path)
    server = subprocess.Popen([str(binary), "echo", str(port), socket_dir], env=env)
    up = False
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            up = True
            break
        except Exception:
            time.sleep(0.1)
    print(f"frontend up: {up} (tokenizer loaded in Rust)")

    ok = False
    try:
        prompt = "The quick brown fox jumps over the lazy dog."

        # streaming NDJSON: pieces detokenized in Rust reassemble the prompt
        lines = _generate(port, prompt, streaming=True)
        got = "".join(base64.b64decode(l["data"]).decode()
                      for l in lines if l.get("modality") == "text")
        stream_ok = got == prompt
        print(f"  stream : {got!r} {'OK' if stream_ok else 'FAIL'}")

        # non-streaming collect: same, through the buffered path
        lines = _generate(port, prompt, streaming=False)
        outputs = lines[0]["outputs"]["text"]
        got2 = "".join(base64.b64decode(o["data"]).decode() for o in outputs)
        collect_ok = got2 == prompt
        print(f"  collect: {got2!r} {'OK' if collect_ok else 'FAIL'}")

        ok = stream_ok and collect_ok
    finally:
        server.terminate()
        cond.stop()
        serve_thread.join(timeout=2)
        cond.shutdown_workers()
        worker.join(timeout=3)

    print(f"\nFRONTEND TOKENIZE {'OK' if ok else 'FAILED'} "
          f"(Rust tokenize -> token ids both ways -> Rust incremental detok; "
          f"model side holds NO tokenizer)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
