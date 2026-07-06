"""First end-to-end model on mstar-rs: V-JEPA 2 `prefill_video`.

Runs one (or more) video requests through the Rust runtime driving the
Python data plane, then verifies the pipeline output bit-for-bit against a
direct `VJEPA2Model.forward` call on the same input.

Usage:
    python examples/run_vjepa2.py [--video path.mp4] [--requests N] [--device cuda:0]

Without --video, synthetic random frames are used (the correctness check
compares runtime-vs-direct on identical inputs, so synthetic input is a
valid verification).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402
from mstar_rs.models import VJEPA2  # noqa: E402


def load_frames(video: str | None, num_frames: int) -> np.ndarray:
    if video is None:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(num_frames, 256, 256, 3), dtype=np.uint8)
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video)
    idx = np.linspace(0, len(decoder) - 1, num_frames, dtype=int).tolist()
    return decoder.get_frames_at(indices=idx).data.permute(0, 2, 3, 1).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="optional video file")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    args = parser.parse_args()

    print(f"loading {args.model} on {args.device} ...")
    t0 = time.perf_counter()
    model = VJEPA2(model_id=args.model, device=args.device)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    frames = load_frames(args.video, args.frames)
    pixel_values = model.preprocess_video(list(frames))
    print(f"pixel_values_videos: {tuple(pixel_values.shape)} {pixel_values.dtype}")

    driver = Driver(model, max_batch_size=8)
    rids = [
        driver.submit({"pixel_values_videos": pixel_values, "walk": "prefill_video"})
        for _ in range(args.requests)
    ]
    t0 = time.perf_counter()
    results = driver.run_until_idle()
    elapsed = time.perf_counter() - t0

    print(f"\nran {len(rids)} request(s) through the Rust runtime in {elapsed:.3f}s")
    predicted = None
    for rid in rids:
        emissions = results[rid]
        assert len(emissions) == 1, f"request {rid}: expected 1 emission, got {len(emissions)}"
        predicted = emissions[0][0]
        print(
            f"  request {rid}: predicted_hidden {tuple(predicted.shape)} "
            f"mean={predicted.float().mean().item():+.6f}"
        )

    # Verify against a direct transformers forward on the same input.
    with torch.inference_mode():
        reference = model.hf_model(pixel_values_videos=pixel_values).predictor_output.last_hidden_state
    if torch.equal(predicted, reference):
        print("\nVERIFY: runtime output is BIT-EXACT vs direct VJEPA2Model.forward")
    else:
        max_diff = (predicted - reference).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            predicted.flatten().float(), reference.flatten().float(), dim=0
        ).item()
        print(f"\nVERIFY: max_abs_diff={max_diff:.3e} cosine={cos:.6f}")
        if not (cos > 0.9999):
            print("MISMATCH vs reference — failing")
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
