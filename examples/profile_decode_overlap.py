"""Quantify the L2 (GPU/CPU overlap) opportunity for orpheus batched decode.

Per decode step we currently do, SERIALLY: GPU replay (batched paged decode +
in-graph sample) THEN host work (seen-mask sync, per-request EOS `.item()`,
output build). An overlap pipeline (L2) can hide whichever is smaller behind
the other, so the achievable speedup ceiling is:

    wall / max(gpu, host)     (== 1.0 if host is fully hidden already)

This measures gpu vs host per step at bs=1 and bs=8. If host << gpu at bs=8,
L2 won't help (don't build it). If host ~ gpu, L2 has a real ceiling.

    CUDA_VISIBLE_DEVICES=3 python examples/profile_decode_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402
from mstar_rs.models import Orpheus  # noqa: E402


def profile(model, bs: int, max_tokens: int) -> None:
    # warm up (captures the per-N decode graph, first-token JIT) then measure.
    prompts = [f"tell me a short story number {i}" for i in range(bs)]
    for measure in (False, True):
        model.prof = {"wall": 0.0, "gpu": 0.0, "host": 0.0, "n": 0} if measure else None
        drv = Driver(model, max_batch_size=8)
        for i, p in enumerate(prompts):
            drv.submit({"prompt": p, "voice": "tara", "seed": i + 1})
        drv.run_until_idle()
        torch.cuda.synchronize()
    pr = model.prof
    n = max(pr["n"], 1)
    gpu, host, wall = pr["gpu"] / n, pr["host"] / n, pr["wall"] / n
    ceiling = wall / max(gpu, host) if max(gpu, host) > 0 else 1.0
    print(f"  bs={bs}: {pr['n']} steps | gpu={gpu:.3f}ms host={host:.3f}ms "
          f"wall={wall:.3f}ms | host/gpu={host/gpu:.2f} | overlap ceiling={ceiling:.2f}x")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    print("loading canopylabs/orpheus-3b-0.1-ft ...")
    model = Orpheus(device="cuda", greedy=True, max_output_tokens=200)
    print("orpheus batched-decode per-step GPU vs host (serial today):")
    profile(model, bs=1, max_tokens=200)
    profile(model, bs=8, max_tokens=200)
    print("\nIf host << gpu at bs=8 -> overlap ceiling ~1x -> L2 not worth it.")
    print("If host ~ gpu at bs=8   -> ceiling toward 2x  -> L2 worth building.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
