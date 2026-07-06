"""Orpheus TTS end-to-end on mstar-rs: the streaming tier in action.

The LLM partition autoregressively decodes audio tokens; each token streams
into the SNAC partition, which windows them (28/7) and decodes 24 kHz audio
chunks concurrently. Writes the concatenated audio to a WAV.

Requires HuggingFace access to the gated repos `canopylabs/orpheus-3b-0.1-ft`
(weights) and `-0.1-pretrained` (tokenizer). Accept the license on the model
pages, then re-run.

Usage: python examples/run_orpheus.py [--text "..."] [--voice tara] [--out out.wav]
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi, I'm a speech model running on the Rust runtime.")
    parser.add_argument("--voice", default="tara")
    parser.add_argument("--out", default="orpheus_out.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mstar_rs.models import Orpheus

    try:
        print("loading orpheus (canopylabs/orpheus-3b-0.1-ft) ...")
        t0 = time.perf_counter()
        model = Orpheus(device=args.device, max_output_tokens=args.max_tokens)
        print(f"loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:  # gated-repo / missing-access guidance
        msg = str(exc)
        if "gated" in msg.lower() or "restricted" in msg.lower() or "401" in msg or "403" in msg:
            print(
                "\nOrpheus weights are gated. Accept the license at\n"
                "  https://huggingface.co/canopylabs/orpheus-3b-0.1-ft\n"
                "  https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained\n"
                "then `hf auth login` and re-run.",
                file=sys.stderr,
            )
            return 2
        raise

    driver = Driver(model, max_batch_size=1)
    rid = driver.submit({"prompt": args.text, "voice": args.voice, "seed": args.seed})
    t0 = time.perf_counter()
    results = driver.run_until_idle()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    chunks = results[rid]
    if not chunks:
        print("no audio produced", file=sys.stderr)
        return 1
    pcm = torch.cat(chunks).numpy()
    seconds = len(pcm) / model.cfg.sample_rate
    print(
        f"\nstreamed {len(chunks)} audio chunk(s) = {seconds:.2f}s of audio "
        f"in {elapsed:.2f}s (RTF {elapsed / seconds:.3f})"
    )

    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(model.cfg.sample_rate)
        w.writeframes(pcm.tobytes())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
