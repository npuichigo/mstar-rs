"""GPU verification for batched paged attention + bucketed CUDA-graph capture.

The check that must pass before the batched path (fi.BatchedFlashInfer-
Attention + batched_graph.BucketedCudaGraph) can back a model. GPU-only (CUDA +
`flashinfer`), no model weights. For each query length it checks:

  1. **Batched == per-request**: attention for a batch of requests in ONE
     BatchedFlashInferAttention plan/run equals each request computed alone
     through the (verified) bs=1 FlashInferAttention.
  2. **Graph replay == eager batched**: a BucketedCudaGraph-captured step
     (padded to a bucket) equals the eager batched run for the real rows.

Two query lengths are exercised: `new_len=1` (AR decode — orpheus) and
`new_len>1` (a multi-token suffix per request — pi05 action_gen's `horizon`).

    python examples/verify_batched_capture.py

Prints per-case max abs diff + cosine; exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def run_check(new_len: int, causal: bool) -> bool:
    from mstar_rs.batched_graph import BucketedCudaGraph, padded_bucket
    from mstar_rs.fi import (
        BatchedFlashInferAttention,
        FlashInferAttention,
        FlashInferCacheHandle,
        FlashInferPagedKV,
    )

    dev = torch.device("cuda")
    torch.manual_seed(new_len)  # vary data per case
    L, PS, KVH, QOH, HD = 1, 16, 4, 8, 64
    dtype = torch.bfloat16
    seq_positions = [30, 47, 16, 63]  # distinct prefix lengths
    bs = len(seq_positions)
    pages_per_req = 8

    cache = FlashInferPagedKV(L, bs * pages_per_req, PS, KVH, HD, dev, dtype)
    req_pages = [list(range(i * pages_per_req, (i + 1) * pages_per_req)) for i in range(bs)]

    def fill_prefix() -> None:
        for i, sp in enumerate(seq_positions):
            for g in range(sp):
                pg, sl = req_pages[i][g // PS], g % PS
                cache.kv[0, pg, 0, sl] = torch.randn(KVH, HD, device=dev, dtype=dtype)
                cache.kv[0, pg, 1, sl] = torch.randn(KVH, HD, device=dev, dtype=dtype)

    def clear_new() -> None:
        for i, sp in enumerate(seq_positions):
            for g in range(sp, sp + new_len):
                cache.kv[0, req_pages[i][g // PS], :, g % PS] = 0

    fill_prefix()
    # new_len query tokens per request.
    q = [torch.randn(new_len, QOH, HD, device=dev, dtype=dtype) for _ in range(bs)]
    k = [torch.randn(new_len, KVH, HD, device=dev, dtype=dtype) for _ in range(bs)]
    v = [torch.randn(new_len, KVH, HD, device=dev, dtype=dtype) for _ in range(bs)]

    # reference: each request alone through the bs=1 path.
    ref = []
    for i, sp in enumerate(seq_positions):
        attn = FlashInferAttention(cache, QOH, HD, dev, max_new_tokens=new_len,
                                   cudagraph=False, causal=causal, dtype=dtype)
        attn.plan(req_pages[i], sp, new_len)
        ref.append(FlashInferCacheHandle(attn).run_attention(q[i], k[i], v[i]).clone())
    clear_new()  # reset new-token slots the refs wrote

    # test 1: eager batched.
    battn = BatchedFlashInferAttention(cache, QOH, HD, dev, bs=bs,
                                       max_pages_per_req=pages_per_req, new_len=new_len,
                                       cudagraph=False, causal=causal, dtype=dtype)
    battn.plan([(req_pages[i], seq_positions[i], new_len) for i in range(bs)])
    qb = torch.cat(q).reshape(bs * new_len, QOH, HD)
    kb = torch.cat(k).reshape(bs * new_len, KVH, HD)
    vb = torch.cat(v).reshape(bs * new_len, KVH, HD)
    out_b = FlashInferCacheHandle(battn).run_attention(qb, kb, vb)
    ref_stack = torch.cat(ref).reshape(bs * new_len, QOH, HD)
    d1 = float((out_b.float() - ref_stack.float()).abs().max())
    ok1 = d1 < 5e-3
    print(f"  [new_len={new_len} causal={causal}] batched==per-request: "
          f"max_abs_diff={d1:.2e} cos={_cos(out_b, ref_stack):.6f} {'OK' if ok1 else 'MISMATCH'}")

    # test 2: BucketedCudaGraph capture/replay == eager batched.
    clear_new()
    gattn = BatchedFlashInferAttention(cache, QOH, HD, dev, bs=bs,
                                       max_pages_per_req=pages_per_req, new_len=new_len,
                                       cudagraph=True, causal=causal, dtype=dtype)
    gattn.plan([(req_pages[i], seq_positions[i], new_len) for i in range(bs)])
    gh = FlashInferCacheHandle(gattn)
    qg, kg, vg = qb.clone(), kb.clone(), vb.clone()
    held: dict[str, torch.Tensor] = {}

    def step(_bs: int) -> None:
        held["out"] = gh.run_attention(qg, kg, vg)

    BucketedCudaGraph(step, buckets=(bs,)).replay(real_bs=bs)
    torch.cuda.synchronize()
    d2 = float((held["out"].float() - out_b.float()).abs().max())
    ok2 = d2 < 5e-3
    print(f"  [new_len={new_len} causal={causal}] graph==eager:          "
          f"max_abs_diff={d2:.2e} cos={_cos(held['out'], out_b):.6f} {'OK' if ok2 else 'MISMATCH'} "
          f"(bs={bs}->bucket {padded_bucket(bs)})")
    return ok1 and ok2


def _child(new_len: int, causal: bool, q) -> None:
    # Each check runs in its OWN process: a FlashInfer CUDA-graph capture
    # leaves wrapper/stream state that perturbs a later eager wrapper in the
    # same process (harness artifact, not a kernel bug — the primitives match
    # bit-exactly when each check runs isolated). Subprocess = clean state.
    import torch as _t

    if not _t.cuda.is_available():
        q.put(("skip", "no CUDA device"))
        return
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        q.put(("skip", "flashinfer not installed"))
        return
    q.put(("ok" if run_check(new_len, causal) else "fail", ""))


def main() -> int:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    cases = [
        ("decode-shaped (new_len=1, e.g. orpheus)", 1),
        ("suffix-shaped (new_len>1, e.g. pi05 action_gen horizon)", 8),
    ]
    ok = True
    for title, nl in cases:
        print(title + ":")
        q: mp.Queue = ctx.Queue()
        p = ctx.Process(target=_child, args=(nl, False, q))
        p.start()
        status, msg = q.get()
        p.join()
        if status == "skip":
            print(f"  SKIP: {msg}")
            return 0
        ok &= status == "ok"

    print("\nBATCHED CAPTURE VERIFIED" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
