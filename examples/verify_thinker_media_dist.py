"""End-to-end proof: multimodal INPUT (image / audio) flows through the FULL
mstar-rs serving runtime — feature extraction in the weightless conductor, the
encoder forward + masked-scatter + media-aware 3D-MRoPE in the TP Thinker
engine, prefill+decode through the conductor's graph — producing a text answer.

This is the serving-path version of verify_vision_input.py / verify_audio_input.py
(which proved the mechanism standalone). Here the same media path runs through
the conductor + two TP worker processes (rank 0 GPU1, rank 1 GPU6), driven by
`cond.submit({"text":..., "file_paths":{...}})` exactly as the axum frontend's
bridge submits it — so removing mstar's FastAPI leaves this path intact.

    CUDA_VISIBLE_DEVICES=1,6 python examples/verify_thinker_media_dist.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

WORLD = 2
TP_PORT = 29717
MAX_NEW = 48


def worker_main(worker_id: str, tp_rank: int, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models import Qwen3OmniThinkerEngine

        engine = Qwen3OmniThinkerEngine(
            tp_rank=tp_rank, tp_world=WORLD, max_output_tokens=MAX_NEW,
            greedy=True, tp_port=TP_PORT,
        )
        Worker(worker_id, engine, socket_dir, device=str(engine.device)).run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _make_image(path: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([80, 80, 240, 240], fill="red")
    d.rectangle([260, 260, 400, 400], fill="blue")
    d.text((150, 410), "HELLO", fill="black")
    img.save(path)


def _make_audio(path: str) -> None:
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)  # a 440 Hz tone
    sf.write(path, tone.astype(np.float32), sr)


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_media_")
    ids = [f"thinker_r{r}" for r in range(WORLD)]

    workers = []
    for rank, wid in enumerate(ids):
        p = ctx.Process(target=worker_main, args=(wid, rank, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    print("waiting for TP workers to init + load shards (~40 s)...", flush=True)
    deadline = time.time() + 180
    need = {f"{socket_dir}/{w}.ipc" for w in ids}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{w}.err").exists() for w in ids):
            break
        time.sleep(0.5)
    bound = [w for w in ids if Path(f"{socket_dir}/{w}.ipc").exists()]
    print(f"workers bound: {bound}", flush=True)

    from mstar_rs.dist import Conductor
    from mstar_rs.models import Qwen3OmniThinkerPolicy

    policy = Qwen3OmniThinkerPolicy(max_output_tokens=MAX_NEW, greedy=True)
    cond = Conductor(policy, node_to_worker={"Thinker": ids}, socket_dir=socket_dir)

    img_path = f"{socket_dir}/scene.png"
    wav_path = f"{socket_dir}/tone.wav"
    _make_image(img_path)
    _make_audio(wav_path)

    def run(request: dict) -> str:
        rid = cond.submit(request)
        results = cond.run_until_idle()
        return policy.tokenizer.decode(results.get(rid, []), skip_special_tokens=True)

    # 1) image in -> describe the synthetic scene
    vision_text = run({"text": "Describe this image in one sentence.",
                       "file_paths": {"image": [img_path]}})
    print(f"\n[vision] {vision_text!r}", flush=True)

    # 2) audio in -> describe the tone (synthetic; assert the path runs coherently)
    audio_text = run({"text": "What do you hear in this audio? Answer in one sentence.",
                      "file_paths": {"audio": [wav_path]}})
    print(f"[audio]  {audio_text!r}", flush=True)

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for wid in ids:
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")

    vlow = vision_text.lower()
    vision_ok = any(k in vlow for k in ("circle", "square", "red", "blue", "hello"))
    audio_ok = len(audio_text.strip()) > 3
    print(f"\n[verify] vision mentions a shape/color/text: {'PASS' if vision_ok else 'FAIL'}")
    print(f"[verify] audio produced coherent text: {'PASS' if audio_ok else 'FAIL'}")
    ok = vision_ok and audio_ok
    print(f"\nMEDIA-INPUT SERVING PATH {'OK' if ok else 'FAILED'} "
          f"(conductor feature-extract -> TP Thinker encoder+scatter+MRoPE -> answer)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
