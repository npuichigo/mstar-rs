"""Profile the audio-path per-frame cost to locate the RTF gap.

Each codec frame -> 1920 samples @ 24 kHz = 80 ms of audio. So RTF 0.1 needs
~8 ms/frame; my eager pipeline is ~10x that. This times the Talker decode step
(backbone + the 16-step code-predictor depth loop, all eager) and the vocoder,
so we can see where the time goes and what CUDA-graph capture would buy.

    CUDA_VISIBLE_DEVICES=7 python examples/profile_qwen3_audio.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs.models import Qwen3OmniAudioEngine  # noqa: E402

FRAMES = 60
SAMPLES_PER_FRAME = 1920
SR = 24000


def main() -> int:
    torch.manual_seed(0)
    eng = Qwen3OmniAudioEngine(device="cuda")
    thinker_hidden = eng.cfg.thinker_hidden_size
    last_hidden = torch.randn(1, thinker_hidden, device="cuda") * 0.1
    rid = "prof"

    # warm up (first frame builds any lazy state) then time.
    codes, tie, layer0, sp = eng.last_prefill(rid, last_hidden, eng.voice, 0)
    torch.cuda.synchronize()

    dec_ms = []
    for _ in range(FRAMES):
        torch.cuda.synchronize(); t = time.time()
        codes, tie, layer0, sp = eng.decode_step(rid, tie, eng.tts_pad(), sp)
        torch.cuda.synchronize()
        dec_ms.append((time.time() - t) * 1000)
        if int(layer0.reshape(-1)[0].item()) == eng.cfg.talker.codec_eos_token_id:
            break

    n = len(dec_ms)
    # steady-state = drop the first few (warm) frames
    steady = dec_ms[3:] or dec_ms
    frame_ms = sum(steady) / len(steady)

    # vocoder: 25-frame chunk. Warm it first (the compiled graph + cuDNN plan
    # cost ~hundreds of ms on the FIRST call for a new shape; steady-state is
    # far lower). Measure the warm steady-state.
    codec = torch.stack([torch.randint(0, 2000, (16,), device="cuda") for _ in range(25)])
    for _ in range(4):
        eng.code2wav_chunk(codec, first_chunk=True)
        torch.cuda.synchronize()
    reps = 5
    t = time.time()
    for _ in range(reps):
        eng.code2wav_chunk(codec, first_chunk=True)
    torch.cuda.synchronize()
    voc_ms = (time.time() - t) * 1000 / reps
    voc_ms_per_frame = voc_ms / 25

    audio_ms_per_frame = SAMPLES_PER_FRAME / SR * 1000  # 80 ms
    compute_ms_per_frame = frame_ms + voc_ms_per_frame
    rtf = compute_ms_per_frame / audio_ms_per_frame

    mode = "CUDA-graph" if eng.cuda_graph else "eager"
    print(f"\n--- audio-path per-frame profile ({n} decode frames, {mode}) ---")
    print(f"Talker decode (backbone + 16-step depth loop, {mode}): {frame_ms:6.1f} ms/frame")
    print(f"Code2Wav vocoder:                                     {voc_ms_per_frame:6.1f} ms/frame")
    print(f"total compute:                                        {compute_ms_per_frame:6.1f} ms/frame")
    print(f"audio produced:                                       {audio_ms_per_frame:6.1f} ms/frame")
    print(f"=> audio-path RTF (compute/audio):                    {rtf:6.2f}")
    print(f"   target ~0.10  =>  need ~{audio_ms_per_frame*0.10:.1f} ms/frame "
          f"(~{compute_ms_per_frame/(audio_ms_per_frame*0.10):.0f}x speedup)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
