"""Slice-2 mechanism check: the Talker (backbone + code-predictor depth loop)
and Code2Wav vocoder run end-to-end on one GPU, driven by mstar-rs.

Fed SYNTHETIC Thinker conditioning (random hidden), so the audio is not
meaningful speech — this validates the MECHANISM (the hard parts): the 6-token
assistant prefix, the nested 1-semantic + 15-residual depth loop producing 16
valid codec tokens/frame, the self-fed talker_input_embeds across decode steps,
and the vocoder turning codec frames into 24 kHz PCM of the right length. It is
the audio-path analog of tp_thinker_forward.py; meaningful speech (real Thinker
states streamed in) is the next slice (the full pipeline through the conductor).

    CUDA_VISIBLE_DEVICES=7 python examples/verify_talker_code2wav.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs.models import Qwen3OmniAudioEngine  # noqa: E402

FRAMES = 30


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    torch.manual_seed(0)
    print("loading Talker + Code2Wav (TP=1, one GPU)...", flush=True)
    eng = Qwen3OmniAudioEngine(device="cuda")
    print(f"loaded; GPU mem={torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    cfg = eng.cfg
    hidden = cfg.talker_hidden_size
    thinker_hidden = cfg.thinker_hidden_size
    codec_eos = cfg.talker.codec_eos_token_id
    layer0_vocab = cfg.talker_text.vocab_size          # 3072
    resid_vocab = cfg.code_predictor.vocab_size        # 2048
    voice = next(iter((cfg.talker.speaker_id or {"ethan": 0}).keys()))

    # synthetic "last Thinker hidden" (what the assistant prefix conditions on)
    last_hidden = torch.randn(1, thinker_hidden, device="cuda") * 0.1
    rid = "audio0"

    codes0, tie, layer0, seq_pos = eng.last_prefill(rid, last_hidden, voice, 0)
    frames = [codes0.reshape(-1)]
    print(f"[last_prefill] voice={voice!r} frame0 codes={codes0.reshape(-1).tolist()[:6]}... "
          f"seq_pos={seq_pos}", flush=True)

    for _ in range(FRAMES - 1):
        # no more Thinker streaming -> feed tts_pad conditioning
        codes, tie, layer0, seq_pos = eng.decode_step(rid, tie, eng.tts_pad(), seq_pos)
        frames.append(codes.reshape(-1))
        if int(layer0.reshape(-1)[0].item()) == codec_eos:
            print(f"[decode] hit codec_eos at frame {len(frames)}", flush=True)
            break

    codec = torch.stack(frames)                         # (F, 16)
    F = codec.shape[0]
    print(f"[talker] generated {F} frames x {codec.shape[1]} codes", flush=True)

    pcm = eng.code2wav_chunk(codec, first_chunk=True)    # (F*1920,) int16
    print(f"[code2wav] {pcm.shape[0]} samples @ {eng.sr} Hz "
          f"({pcm.shape[0] / eng.sr:.2f}s), dtype={pcm.dtype}", flush=True)

    # verification: shapes, code ranges, finite audio of the right length.
    checks = {
        "16 codes/frame": codec.shape[1] == eng.num_codes == 16,
        "layer0 in range": bool((codec[:, 0] >= 0).all() and (codec[:, 0] < layer0_vocab).all()),
        "residuals in range": bool((codec[:, 1:] >= 0).all() and (codec[:, 1:] < resid_vocab).all()),
        "audio finite": bool(torch.isfinite(pcm.float()).all()),
        "audio len == F*1920": pcm.shape[0] == F * eng.code2wav.total_upsample,
        "audio int16": pcm.dtype == torch.int16,
        "audio non-silent": int(pcm.abs().max().item()) > 0,
    }
    for k, v in checks.items():
        print(f"[verify] {k}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nTALKER + CODE2WAV MECHANISM {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
