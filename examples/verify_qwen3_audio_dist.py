"""CAPSTONE: the full Qwen3-Omni audio pipeline through the mstar-rs conductor.

3 worker processes: Thinker tensor-parallel on ranks 0,1 (GPU1, GPU6) +
Talker/Code2Wav on rank 2 (GPU7). The conductor drives the 3-partition graph
with cross-partition streaming: Thinker text tokens' hidden states stream to the
Talker (fixed_chunk 1, continue_after_done), the Talker's codec tokens stream
to Code2Wav (left-context 25+25). Text prompt in -> 24 kHz speech out.

This exercises every capability built this project: TP sharding, the TP-aware
conductor dispatch, cross-worker SHM tensor transport, and cross-partition
streaming between a TP partition and two single-GPU partitions.

    CUDA_VISIBLE_DEVICES=1,6,7 python examples/verify_qwen3_audio_dist.py
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

TP_PORT = 29733
PROMPT = "What is the capital of France? Answer in one sentence."
MAX_NEW = 64
VOICE = "chelsie"
OUT_WAV = os.environ.get("QWEN3_WAV", str(Path(tempfile.gettempdir()) / "qwen3_audio.wav"))


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
        else:  # audio on rank 2 (cuda:2 in the CVD=1,6,7 mapping)
            eng = Qwen3OmniAudioEngine(device="cuda:2", voice=VOICE, max_output_tokens=MAX_NEW)
            Worker("audio", eng, socket_dir, device="cuda:2").run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{kind}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_audio_")
    kinds = ["thinker0", "thinker1", "audio"]

    workers = []
    for kind in kinds:
        p = ctx.Process(target=worker_main, args=(kind, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    print("waiting for workers (Thinker shards ~35s + audio ~15s)...", flush=True)
    deadline = time.time() + 240
    need = {f"{socket_dir}/{k}.ipc" for k in ("thinker0", "thinker1", "audio")}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{k}.err").exists() for k in kinds):
            break
        time.sleep(0.5)
    bound = [k for k in ("thinker0", "thinker1", "audio") if Path(f"{socket_dir}/{k}.ipc").exists()]
    print(f"workers bound: {bound}", flush=True)

    from mstar_rs.dist import Conductor
    from mstar_rs.models import Qwen3OmniAudioPolicy

    policy = Qwen3OmniAudioPolicy(max_output_tokens=MAX_NEW, voice=VOICE)
    cond = Conductor(
        policy,
        node_to_worker={"Thinker": ["thinker0", "thinker1"], "Talker": "audio", "Code2Wav": "audio"},
        socket_dir=socket_dir,
    )

    t0 = time.time()
    rid = cond.submit({"prompt": PROMPT})
    results = cond.run_until_idle()
    dt = time.time() - t0

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for k in kinds:
        errf = Path(f"{socket_dir}/{k}.err")
        if errf.exists():
            print(f"--- {k} error ---\n{errf.read_text()}")

    # results[rid] holds interleaved text token ids (int) + audio chunks (int16 tensors).
    import torch

    out = results.get(rid, [])
    text_toks = [x for x in out if isinstance(x, int)]
    audio_chunks = [x for x in out if isinstance(x, torch.Tensor)]
    text = policy.tokenizer.decode(text_toks, skip_special_tokens=True) if text_toks else ""
    pcm = torch.cat(audio_chunks) if audio_chunks else torch.zeros(0, dtype=torch.int16)

    print(f"\n[thinker] text: {text!r} ({len(text_toks)} tokens)")
    print(f"[audio] {len(audio_chunks)} chunks, {pcm.shape[0]} samples "
          f"({pcm.shape[0] / 24000:.2f}s @ 24 kHz) in {dt:.1f}s")

    ok = pcm.shape[0] > 0 and bool(torch.isfinite(pcm.float()).all()) and int(pcm.abs().max()) > 0
    if ok:
        Path(OUT_WAV).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(OUT_WAV, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.numpy().tobytes())
        print(f"[audio] wrote {OUT_WAV}")
    print(f"\nQWEN3-OMNI AUDIO PIPELINE {'OK' if ok else 'FAILED'} "
          f"(Thinker TP + Talker + Code2Wav streaming through the conductor)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
