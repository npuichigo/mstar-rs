"""qwen3-omni audio pipeline, DECENTRALIZED across GPUs — the real flagship on
the in-worker scheduler (phase-2 applied to GPU/TP).

Three worker processes, each owning a partition and driving its OWN loop with
no per-step conductor round-trip:
  - thinker0 (GPU1) + thinker1 (GPU6): the Thinker, tensor-parallel. Both run
    the same deterministic schedule in NCCL lockstep (single request); rank 0
    is the I/O leader (ships thinker_states + emits text), rank 1 is a silent
    follower (computes for the all-reduce only).
  - audio (GPU7): Talker + Code2Wav; self-drives its decode loops, receives
    thinker_states peer-to-peer from thinker0, streams codec Talker->Code2Wav
    locally.
A thin DisaggCoordinator only ingests the request and collects text + audio.

Contrast the centralized alternative (a per-step conductor: a batch dispatched
over ZMQ every step).

    CUDA_VISIBLE_DEVICES=1,6,7 python examples/verify_qwen3_audio_disagg.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TP_PORT = 29755
PROMPT = "What is the capital of France? Answer in one sentence."
MAX_NEW = 64
VOICE = "chelsie"
OUT_WAV = str(Path(tempfile.gettempdir()) / "qwen3_audio_disagg.wav")
# partition -> worker ranks (Thinker is TP across two); first = I/O leader.
PARTITION_TO_WORKERS = {
    "Thinker": ["thinker0", "thinker1"],
    "Talker": ["audio"],
    "Code2Wav": ["audio"],
}
# for a worker shipping a stream chunk: partition -> the consumer's leader.
SHIP_TO = {"Thinker": "thinker0", "Talker": "audio", "Code2Wav": "audio"}


def _worker_model(kind: str):
    """Compose the full audio graph (policy) with this worker's engine
    (execute) — the Thinker/audio engines' walk names match the audio graph, so
    execute plugs straight in."""
    from mstar_rs import Model
    from mstar_rs.models import (
        Qwen3OmniAudioEngine,
        Qwen3OmniAudioPolicy,
        Qwen3OmniThinkerEngine,
    )

    policy = Qwen3OmniAudioPolicy(max_output_tokens=MAX_NEW, voice=VOICE)
    if kind.startswith("thinker"):
        engine = Qwen3OmniThinkerEngine(
            tp_rank=int(kind[-1]), tp_world=2, max_output_tokens=MAX_NEW,
            greedy=True, tp_port=TP_PORT, audio_output=True,
        )
    else:
        engine = Qwen3OmniAudioEngine(device="cuda:2", voice=VOICE, max_output_tokens=MAX_NEW)

    class _WorkerModel(Model):
        def __init__(self):
            self.device = engine.device
        def walks(self): return policy.walks()
        def partitions(self): return policy.partitions()
        def kv_config(self): return policy.kv_config()
        def unbatchable(self): return policy.unbatchable()
        def initial_walks(self, r): return policy.initial_walks(r)
        def next_forward(self, *a): return policy.next_forward(*a)
        def postprocess(self, *a): return policy.postprocess(*a)
        def execute(self, *a, **k): return engine.execute(*a, **k)
        def loops_to_finish(self): return engine.loops_to_finish()

    return _WorkerModel()


def worker_main(kind: str, local_partitions: list, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import DisaggWorker

        model = _worker_model(kind)
        DisaggWorker(kind, model, local_partitions, SHIP_TO, socket_dir,
                     device=str(model.device), io_leader=(kind != "thinker1")).run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{kind}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    import torch

    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_qwen_disagg_")
    specs = [("thinker0", ["Thinker"]), ("thinker1", ["Thinker"]),
             ("audio", ["Talker", "Code2Wav"])]

    workers = []
    for kind, parts in specs:
        p = ctx.Process(target=worker_main, args=(kind, parts, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    print("waiting for workers (Thinker shards ~35s + audio ~15s)...", flush=True)
    deadline = time.time() + 240
    need = {f"{socket_dir}/{k}.ipc" for k, _ in specs}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{k}.err").exists() for k, _ in specs):
            break
        time.sleep(0.5)
    print(f"workers bound: {[k for k, _ in specs if Path(f'{socket_dir}/{k}.ipc').exists()]}", flush=True)

    from mstar_rs.dist import DisaggCoordinator
    from mstar_rs.models import Qwen3OmniAudioPolicy

    policy = Qwen3OmniAudioPolicy(max_output_tokens=MAX_NEW, voice=VOICE)
    cond = DisaggCoordinator(policy, PARTITION_TO_WORKERS, socket_dir)

    t0 = time.time()
    gid = cond.submit({"prompt": PROMPT})
    cond.run_until_idle()
    dt = time.time() - t0

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for k, _ in specs:
        errf = Path(f"{socket_dir}/{k}.err")
        if errf.exists():
            print(f"--- {k} error ---\n{errf.read_text()}")

    out = cond.results.get(gid, [])
    toks = [x for x in out if isinstance(x, int)]
    chunks = [x for x in out if isinstance(x, torch.Tensor)]
    text = policy.tokenizer.decode(toks, skip_special_tokens=True) if toks else ""
    pcm = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int16)

    print(f"\n[thinker] text: {text!r} ({len(toks)} tokens)")
    print(f"[audio] {len(chunks)} chunks, {pcm.shape[0]} samples "
          f"({pcm.shape[0] / 24000:.2f}s) in {dt:.1f}s")

    ok = pcm.shape[0] > 0 and "Paris" in text
    if pcm.shape[0] > 0:
        with wave.open(OUT_WAV, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm.numpy().tobytes())
        print(f"[audio] wrote {OUT_WAV}")
    print(f"\nQWEN3-OMNI DECENTRALIZED {'OK' if ok else 'FAILED'} "
          f"(3 self-driving workers, TP Thinker + peer-to-peer streaming, no per-step conductor)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
