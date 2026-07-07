"""Verify pi05 batched action_gen == per-request (GPU, real lerobot/pi05_base).

Submits several requests with distinct seeds TOGETHER (so the runtime batches
them through the action_gen euler loop via the per-batch-size CUDA graph), then
runs each request ALONE (batch size 1), and checks the (50, 32) action
trajectories match. Since bs=1 is already verified against the dense fp32
reference (examples/run_pi05.py), matching bs=1 proves the batched path.

    python examples/verify_pi05_batched.py [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402
from mstar_rs.models import PI05  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    print("loading lerobot/pi05_base ...")
    model = PI05(device=args.device)
    torch.manual_seed(42)
    images = torch.rand(model.cfg.num_cameras, 3, 224, 224)
    robot_state = torch.rand(32) * 2 - 1
    prompt = "pick up the red cube and place it in the bin"

    def req(seed: int) -> dict:
        return {"images": images, "prompt": prompt, "robot_state": robot_state, "seed": seed}

    seeds = [1, 2, 3]

    # --- batched: submit all together so they share each action_gen batch ---
    drv = Driver(model, max_batch_size=8)
    rids = [drv.submit(req(s)) for s in seeds]
    res = drv.run_until_idle()
    batched = {s: res[r][0].clone() for s, r in zip(seeds, rids)}

    # --- sequential: each request alone (batch size 1) ---
    alone = {}
    for s in seeds:
        d = Driver(model, max_batch_size=8)
        r = d.submit(req(s))
        alone[s] = d.run_until_idle()[r][0].clone()

    ok = True
    for s in seeds:
        d = float((batched[s] - alone[s]).abs().max())
        c = float(torch.nn.functional.cosine_similarity(
            batched[s].flatten(), alone[s].flatten(), dim=0))
        good = d < 1e-2 and c > 0.9999
        ok &= good
        print(f"  seed={s}: batched vs alone  max_abs_diff={d:.3e} cos={c:.6f} "
              f"{'OK' if good else 'MISMATCH'}")

    # Sanity: distinct seeds must give distinct actions (guards against every
    # request aliasing one buffer — the exact bug the batched path must avoid).
    distinct = float((batched[seeds[0]] - batched[seeds[1]]).abs().max())
    print(f"  distinct seeds differ by max_abs={distinct:.3e} "
          f"{'OK' if distinct > 1e-3 else 'SUSPICIOUS (aliased?)'}")
    ok &= distinct > 1e-3

    print("\nPI05 BATCHED action_gen VERIFIED" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
