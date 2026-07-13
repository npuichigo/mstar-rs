"""Foundational TP milestone: run the Qwen3-Omni Thinker tensor-parallel across
2 GPUs, driven by mstar-rs's own `fi.py` handle.

The Thinker is a ~39B MoE (~78 GB bf16 resident) — it does NOT fit on one 80 GB
card with room for KV + activations. Its intended deployment is tensor-parallel.
This script stands up mstar's TP machinery (`TPCommGroup` + `WorkerTPGroups`,
NCCL SPMD, one process per GPU), builds the Thinker SHARDED (each rank allocates
and loads only its ~31 GB slice — shard-on-load, not load-then-slice), and runs
a text prefill + greedy decode through mstar-rs's `FlashInferCacheHandle` (with a
per-rank head-sharded KV pool). MRoPE is applied by the model itself, so the same
handle used for pi05/orpheus drives the Thinker unchanged.

Verification (no TP=1 baseline is possible — TP=1 OOMs, that's the point):
  1. loads sharded without OOM (~31 GB/rank),
  2. both ranks produce IDENTICAL tokens (the residual stream is replicated and
     each layer's two all-reduces re-replicate it — a mismatch means the NCCL
     collectives or sharding are wrong),
  3. the output is coherent ("Paris" for the capital-of-France prompt) — the
     mstar tp2-vs-tp1 smoke bar.

    CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 \\
        --master_port=29555 examples/tp_thinker_forward.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar.distributed.communication import TPCommGroup, WorkerTPGroups  # noqa: E402
from mstar.model.qwen3_omni.components.rope import (  # noqa: E402
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_text,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar_rs.fi import (  # noqa: E402
    FlashInferAttention,
    FlashInferCacheHandle,
    FlashInferPagedKV,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MROPE = [24, 20, 20]
PROMPT = "What is the capital of France? Answer in one sentence."
NUM_PAGES, PAGE_SIZE = 64, 128
MAX_NEW = 40


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(0)  # replicate the worker's seed discipline (both ranks)

    def log(*a: object) -> None:
        if rank == 0:
            print(*a, flush=True)

    # (a) comm group + NCCL init, exactly as a mstar worker does.
    members = list(range(world))
    cg = TPCommGroup(my_global_rank=rank, my_group_rank=rank, group_members=members)
    tp = WorkerTPGroups(
        num_workers=world, global_rank=rank, any_tp=True,
        world_tp_groups=[tuple(members)],
    )
    tp.add("Thinker", cg)
    tp.init_dist(init_method="env://")  # sets cuda:rank, nccl init, attaches device_group
    device = torch.device(f"cuda:{rank}")
    log(f"[dist] world={world} rank={rank} device={device} "
        f"tp_world_size={cg.world_size}")

    # (b) build the SHARDED Thinker on this rank (shard-on-load).
    t0 = time.time()
    model = Qwen3OmniModel(model_path_hf=MODEL_ID)
    thinker = model.get_submodule(
        "Thinker", device=str(device), tp_group=cg, autocast_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    log(f"[load] sharded Thinker in {time.time() - t0:.0f}s; "
        f"GPU mem={torch.cuda.memory_allocated(device) / 1e9:.1f} GB/rank")

    tc = thinker.config.thinker_text
    head_dim = getattr(thinker.config, "thinker_head_dim", None) or getattr(tc, "head_dim", 128)
    n_layers = tc.num_hidden_layers
    qo = tc.num_attention_heads // world           # 28 // 2 = 14 per rank
    kv = max(1, tc.num_key_value_heads // world)   # 4 // 2 = 2 per rank
    inv_freq = compute_rope_freqs(head_dim, tc.rope_theta, device)
    log(f"[shard] layers={n_layers} head_dim={head_dim} "
        f"qo_heads/rank={qo} kv_heads/rank={kv}")

    # (c) per-rank head-sharded KV pool + the mstar-rs handle.
    cache = FlashInferPagedKV(
        n_layers, NUM_PAGES, PAGE_SIZE, kv, head_dim, device, torch.bfloat16,
    )
    attn = FlashInferAttention(
        cache, qo, head_dim, device, max_new_tokens=4096,
        cudagraph=False, causal=True, dtype=torch.bfloat16,
    )
    handle = FlashInferCacheHandle(attn)
    pages = list(range(NUM_PAGES))

    tmodel = thinker.model               # Qwen3OmniThinkerModel
    embed = tmodel.model.embed_tokens
    lm_head = tmodel.lm_head

    def step(embeds: torch.Tensor, start_pos: int, new_len: int) -> torch.Tensor:
        if new_len > 1:
            pos3d = get_rope_index_text(new_len, start_pos, device)
        else:
            pos3d = torch.full((3, 1), float(start_pos), dtype=torch.float, device=device)
        cos, sin = compute_3d_cos_sin(pos3d, inv_freq, MROPE, target_dtype=torch.bfloat16)
        attn.plan(pages, seq_pos=start_pos, new_len=new_len)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = tmodel(
                input_embeds=embeds, cache_handle=handle,
                cos_sin_3d=(cos, sin), mrope_section=MROPE,
            )
            return lm_head(hidden[-1:])   # (1, vocab)

    # tokenize with the instruct chat template (text-only -> no modality tokens).
    text = model._processor.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True,
    )
    ids = model.tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
    S = int(ids.shape[0])

    # prefill + greedy decode.
    with torch.no_grad():
        logits = step(embed(ids).to(torch.bfloat16), 0, S)
    tok = int(logits.argmax(-1).item())
    out = [tok]
    start_pos = S
    eos = model.tokenizer.eos_token_id
    t0 = time.time()
    for _ in range(MAX_NEW):
        with torch.no_grad():
            e = embed(torch.tensor([tok], device=device)).to(torch.bfloat16)
        tok = int(step(e, start_pos, 1).argmax(-1).item())
        start_pos += 1
        if tok == eos:
            break
        out.append(tok)
    torch.cuda.synchronize(device)
    gen = model.tokenizer.decode(out, skip_special_tokens=True)
    log(f"[gen] {len(out)} tokens in {time.time() - t0:.1f}s: {gen!r}")

    # cross-rank agreement: both ranks must emit identical tokens.
    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, out)
    tp.barrier_all()
    if rank == 0:
        agree = all(g == out for g in gathered)
        paris = "Paris" in gen
        print(f"[verify] cross-rank token agreement: {'PASS' if agree else 'FAIL'}")
        print(f"[verify] coherent (contains 'Paris'): {'PASS' if paris else 'FAIL'}")
        ok = agree and paris
        print(f"\nTP THINKER FORWARD {'OK' if ok else 'FAILED'} "
              f"(sharded across {world} GPUs, ~"
              f"{torch.cuda.memory_allocated(device) / 1e9:.0f} GB/rank)")
        rc = 0 if ok else 1
    else:
        rc = 0
    dist.destroy_process_group()  # avoid the NCCL resource-leak warning at exit
    return rc


if __name__ == "__main__":
    sys.exit(main())
