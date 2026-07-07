"""Verify orpheus batched decode == per-request (GPU, real canopylabs weights).

Submits two DIFFERENT prompts together so the runtime batches them through the
decode loop (BatchedFlashInferAttention + the [max_bs] SamplerBuffers batched
sampler), then runs each alone, and compares the **step-1 decode logits** per
request. Logits (not tokens) are the right check: greedy token streams can't
bit-match across batch sizes because bf16 kernel-order noise flips the discrete
argmax (mstar's batched vs sequential would too). At step 1 the KV state is
identical to the solo run, so batched logits[i] must match solo logits[i]
within bf16 tolerance (cos≈1) — while differing BETWEEN requests (no aliasing).

    python examples/verify_orpheus_batched.py [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402
from mstar_rs.models import Orpheus  # noqa: E402


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    print("loading canopylabs/orpheus-3b-0.1-ft ...")
    # max_output_tokens small: we only need the first decode step's logits.
    model = Orpheus(device=args.device, greedy=True, max_output_tokens=8)
    p0 = "Hello there, how are you today?"
    p1 = "The quick brown fox jumps over the lazy dog."

    # --- batched: both prompts together (they share each decode batch) ---
    model.logit_log = {}
    drv = Driver(model, max_batch_size=8)
    r0 = drv.submit({"prompt": p0, "voice": "tara", "seed": 1})
    r1 = drv.submit({"prompt": p1, "voice": "tara", "seed": 2})
    drv.run_until_idle()
    b0, b1 = model.logit_log[r0][0], model.logit_log[r1][0]  # step-1 logits

    # --- sequential: each prompt alone (batch size 1) ---
    def solo(prompt, seed):
        model.logit_log = {}
        d = Driver(model, max_batch_size=8)
        r = d.submit({"prompt": prompt, "voice": "tara", "seed": seed})
        d.run_until_idle()
        return model.logit_log[r][0]

    a0, a1 = solo(p0, 1), solo(p1, 2)

    ok = True
    for name, bat, alone in [("prompt0", b0, a0), ("prompt1", b1, a1)]:
        d = float((bat - alone).abs().max())
        c = _cos(bat, alone)
        good = d < 5e-2 and c > 0.9999
        ok &= good
        print(f"  {name}: batched vs solo step-1 logits  max_abs_diff={d:.3e} cos={c:.6f} "
              f"{'OK' if good else 'MISMATCH'}")
    # No aliasing: each batched request must match ITS OWN solo run more
    # closely than the other request's. (Step-1 logits are largely prompt-
    # independent — a generic audio onset — so absolute cos between prompts is
    # high; what matters is that batched b0 tracks a0, not a1.)
    own0, cross0 = _cos(b0, a0), _cos(b0, a1)
    own1, cross1 = _cos(b1, a1), _cos(b1, a0)
    no_alias = own0 > cross0 and own1 > cross1
    print(f"  no-aliasing: cos(b0,a0)={own0:.6f} > cos(b0,a1)={cross0:.6f} and "
          f"cos(b1,a1)={own1:.6f} > cos(b1,a0)={cross1:.6f}  {'OK' if no_alias else 'ALIASED'}")
    ok &= no_alias

    print("\nORPHEUS BATCHED decode VERIFIED (logit-level)" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
