"""Generate several Qwen3-Omni speech samples in ONE worker session.

Spins up the 3-worker pipeline once (Thinker TP on GPU1+6, Talker/Code2Wav on
GPU7), then submits multiple prompts sequentially, cycling the speaker voice,
saving each to its own .wav.

    CUDA_VISIBLE_DEVICES=1,6,7 python examples/gen_qwen3_audio_samples.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TP_PORT = 29744
MAX_NEW = 96
VOICE = "chelsie"  # one voice per session (fixed at engine init)
OUT_DIR = Path(os.environ.get("QWEN3_OUT_DIR", tempfile.gettempdir()))

PROMPTS = [
    "Tell me a fun fact about the ocean.",
    "Count from one to five.",
    "What is two plus two?",
    "Give me a short cheerful morning greeting.",
    "Name three primary colors.",
]


def worker_main(kind: str, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models import Qwen3OmniAudioEngine, Qwen3OmniThinkerEngine

        if kind.startswith("thinker"):
            rank = int(kind[-1])
            eng = Qwen3OmniThinkerEngine(
                tp_rank=rank, tp_world=2, max_output_tokens=MAX_NEW,
                greedy=True, tp_port=TP_PORT, audio_output=True,
            )
            Worker(kind, eng, socket_dir, device=str(eng.device)).run()
        else:
            eng = Qwen3OmniAudioEngine(device="cuda:2", voice=VOICE, max_output_tokens=MAX_NEW)
            Worker("audio", eng, socket_dir, device="cuda:2").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{kind}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def save_wav(pcm, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.numpy().tobytes())


def main() -> int:
    import torch

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_gen_")
    kinds = ["thinker0", "thinker1", "audio"]
    workers = []
    for kind in kinds:
        p = ctx.Process(target=worker_main, args=(kind, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    print("waiting for workers (~40s)...", flush=True)
    deadline = time.time() + 240
    need = {f"{socket_dir}/{k}.ipc" for k in kinds}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{k}.err").exists() for k in kinds):
            break
        time.sleep(0.5)
    print(f"workers bound: {[k for k in kinds if Path(f'{socket_dir}/{k}.ipc').exists()]}", flush=True)

    from mstar_rs.dist import Conductor
    from mstar_rs.models import Qwen3OmniAudioPolicy

    policy = Qwen3OmniAudioPolicy(max_output_tokens=MAX_NEW, voice=VOICE)
    cond = Conductor(
        policy,
        node_to_worker={"Thinker": ["thinker0", "thinker1"], "Talker": "audio", "Code2Wav": "audio"},
        socket_dir=socket_dir,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        rid = cond.submit({"prompt": prompt})
        res = cond.run_until_idle()
        dt = time.time() - t0
        out = res.get(rid, [])
        toks = [x for x in out if isinstance(x, int)]
        chunks = [x for x in out if isinstance(x, torch.Tensor)]
        text = policy.tokenizer.decode(toks, skip_special_tokens=True) if toks else ""
        pcm = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int16)
        path = OUT_DIR / f"qwen3_sample_{i}_{VOICE}.wav"
        if pcm.shape[0] > 0:
            save_wav(pcm, path)
        secs = pcm.shape[0] / 24000
        results_summary.append((prompt, text, secs, str(path) if pcm.shape[0] else "(no audio)"))
        print(f"[{i}] text={text!r} -> {secs:.2f}s in {dt:.1f}s", flush=True)
        cond.results.clear()

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for k in kinds:
        errf = Path(f"{socket_dir}/{k}.err")
        if errf.exists():
            print(f"--- {k} error ---\n{errf.read_text()}")

    print(f"\n=== SAMPLES (voice={VOICE}) ===")
    for prompt, text, secs, path in results_summary:
        print(f"  {secs:5.2f}s  {text!r}\n           {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
