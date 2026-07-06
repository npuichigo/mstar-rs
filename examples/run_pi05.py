"""Pi0.5 end-to-end on mstar-rs: prefill (paged KV write) + 10-step
flow-matching action_gen loop (paged KV read) through the Rust runtime.

Verification: the same weights and inputs are run through an INDEPENDENT
reference path — contiguous per-layer K/V lists + SDPA, no paging, no
runtime — and the final (50, 32) action trajectories are compared. This
isolates what the T4 tier adds (Rust scheduling/loop/page tables + paged
cache) from the model math itself (which mstar's own parity suite covers
against openpi/lerobot).

Usage: python examples/run_pi05.py [--device cuda] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs import Driver  # noqa: E402
from mstar_rs.models import PI05  # noqa: E402


def gemma_rope(q, k, positions, theta):
    """Standard HF Gemma RoPE (half-rotation), fp32 math — part of the
    independent reference, deliberately not shared with the runtime path."""
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    qf, kf = q.float(), k.float()
    return (qf * cos + rotate_half(qf) * sin).to(q.dtype), (
        kf * cos + rotate_half(kf) * sin
    ).to(k.dtype)


class ContiguousRefHandle:
    """Independent reference cache: per-layer contiguous K/V, plain SDPA,
    plain-torch RoPE, fp32 — shares no code with the FlashInfer runtime path."""

    def __init__(self, scale: float, pos_start: int, write: bool):
        self.scale = scale
        self.pos_start = pos_start
        self.write = write
        self.layer_idx = 0
        self.store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_active_label(self, label):
        pass

    def set_layer_idx(self, idx):
        self.layer_idx = idx

    def apply_rope(self, q, k, rope_theta=10000.0, **_):
        pos = torch.arange(self.pos_start, self.pos_start + q.shape[0], device=q.device)
        return gemma_rope(q, k, pos, rope_theta)

    def run_attention(self, q, k, v):
        past = self.store.get(self.layer_idx)
        if past is not None:
            k_full = torch.cat([past[0], k], dim=0)
            v_full = torch.cat([past[1], v], dim=0)
        else:
            k_full, v_full = k, v
        if self.write:
            self.store[self.layer_idx] = (k_full, v_full)
        qh = q.permute(1, 0, 2).unsqueeze(0)
        kh = k_full.permute(1, 0, 2).unsqueeze(0)
        vh = v_full.permute(1, 0, 2).unsqueeze(0)
        rep = qh.shape[1] // kh.shape[1]
        kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, scale=self.scale, is_causal=False
        )
        return out.squeeze(0).permute(1, 0, 2)

    def advance_seq_lens(self, *a, **k):
        pass


@torch.inference_mode()
def reference_actions(model: PI05, images, prompt, robot_state, seed) -> torch.Tensor:
    """Dense no-runtime reference: prefill + 10 euler steps, contiguous KV."""
    pv = model.vit._prepare_one(images.to(model.device))
    features = model.vit.encoder(pv.float())
    img_emb = features.reshape(-1, features.shape[-1]) * model.llm._image_embed_scale
    text_ids = model.tokenize(prompt, robot_state).to(model.device)
    text_emb = model.llm._embed_tokens_scaled(text_ids)
    prefix_emb = torch.cat([img_emb, text_emb], dim=0)
    prefix_len = prefix_emb.shape[0]

    scale = model.cfg.head_dim**-0.5
    prefill_handle = ContiguousRefHandle(scale, pos_start=0, write=True)
    model.llm.paligemma(query_sequence=prefix_emb, cache_handle=prefill_handle, write_cache=True)

    generator = torch.Generator(device=model.device).manual_seed(seed)
    noisy = torch.randn(
        model.cfg.action_horizon, model.cfg.action_dim,
        device=model.device, generator=generator,
    )
    ts = torch.zeros(1, device=model.device, dtype=torch.long)
    for _ in range(model.cfg.num_flow_steps):
        step_handle = ContiguousRefHandle(scale, pos_start=prefix_len, write=False)
        step_handle.store = prefill_handle.store  # frozen prefix
        noisy, ts = model.llm._euler_step(
            noisy, ts, model._fraction, model._time_emb_buffer, step_handle
        )
    return noisy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5, help="timed requests")
    args = parser.parse_args()

    print("loading lerobot/pi05_base via mstar's Pi05Model ...")
    t0 = time.perf_counter()
    model = PI05(device=args.device)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    torch.manual_seed(42)
    images = torch.rand(model.cfg.num_cameras, 3, 224, 224)  # [0,1] floats
    robot_state = torch.rand(32) * 2 - 1
    prompt = "pick up the red cube and place it in the bin"
    request = {
        "images": images,
        "prompt": prompt,
        "robot_state": robot_state,
        "seed": args.seed,
    }

    driver = Driver(model, max_batch_size=8)
    rid = driver.submit(request)
    t0 = time.perf_counter()
    results = driver.run_until_idle()
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    actions = results[rid][0]
    print(f"\nmstar-rs pi05: actions {tuple(actions.shape)} in {e2e*1000:.1f}ms "
          f"(prefill + {model.cfg.num_flow_steps} flow steps)")
    pages, seq_pos = driver.runtime.kv_state(rid, "main")
    print(f"  kv after run: pages={pages} seq_pos={seq_pos} (freed on finish)")

    # --- verify against the dense fp32 no-runtime reference ---
    ref = reference_actions(model, images, prompt, robot_state, args.seed).cpu()
    max_diff = (actions - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actions.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"\nVERIFY vs dense fp32 reference: max_abs_diff={max_diff:.3e} cosine={cos:.6f}")
    # The runtime path is bf16 (mstar's engine config); mstar's own
    # full-stack tolerance vs lerobot is 1e-2.
    if max_diff > 5e-2 or cos < 0.999:
        print("MISMATCH — failing")
        return 1

    # --- timed loop (steady-state latency) ---
    times = []
    for i in range(args.iters):
        rid = driver.submit(dict(request, seed=args.seed + i + 1))
        t0 = time.perf_counter()
        driver.run_until_idle()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"\ntimed: mean={sum(times)/len(times)*1000:.1f}ms "
          f"p50={times[len(times)//2]*1000:.1f}ms min={times[0]*1000:.1f}ms "
          f"over {args.iters} requests")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
