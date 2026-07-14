"""Multimodal VISION INPUT: feed an image to the Qwen3-Omni Thinker and get a
text description. Mirrors the audio-input path, plus the two things vision adds:
  * DeepStack — the vision encoder returns intermediate features that get added
    into the Thinker's first layers at the image-token positions.
  * grid-based 3D-MRoPE — image tokens get spatial (h/w) positions from the
    patch grid, and the grid span exceeds the token count, so the KV-cache
    position (token-based) and the RoPE position (grid-based) advance separately.

The vision encoder (HF, FlashAttention-2) is stateless; the Thinker runs
tensor-parallel (encoder replicated per rank). Reuses mstar's
get_submodule("vision_encoder"), embed_tokens, thinker.model, rope.py.

    QWEN3_IMAGE=/path/to.png CUDA_VISIBLE_DEVICES=1,6 \\
        torchrun --nproc_per_node=2 --master_port=29564 examples/verify_vision_input.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar.distributed.communication import TPCommGroup, WorkerTPGroups  # noqa: E402
from mstar.model.qwen3_omni.components.rope import (  # noqa: E402
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_text,
    get_rope_index_vision,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar_rs.fi import (  # noqa: E402
    FlashInferAttention,
    FlashInferCacheHandle,
    FlashInferPagedKV,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MROPE = [24, 20, 20]
IMAGE_PAD, IMAGE_BOS, IMAGE_EOS = 151655, 151652, 151653
IMAGE = os.environ.get("QWEN3_IMAGE", "image.png")
QUESTION = "Describe this image in one sentence."
NUM_PAGES, PAGE_SIZE, MAX_NEW = 64, 128, 48


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(0)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    members = list(range(world))
    cg = TPCommGroup(my_global_rank=rank, my_group_rank=rank, group_members=members)
    tp = WorkerTPGroups(num_workers=world, global_rank=rank, any_tp=True,
                        world_tp_groups=[tuple(members)])
    tp.add("Thinker", cg)
    tp.init_dist(init_method="env://")
    device = torch.device(f"cuda:{rank}")

    t0 = time.time()
    model = Qwen3OmniModel(model_path_hf=MODEL_ID)
    thinker = model.get_submodule("Thinker", device=str(device), tp_group=cg,
                                  autocast_dtype=torch.bfloat16)
    vis_enc = model.get_submodule("vision_encoder", device=str(device)).vision_encoder
    vis_enc = vis_enc.to(device)
    for mod in vis_enc.modules():  # move stray CPU tensor attributes (like audio's PE)
        for name, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                setattr(mod, name, val.to(device))
    enc_dtype = next(vis_enc.parameters()).dtype
    log(f"[load] Thinker(shard)+vision_encoder in {time.time()-t0:.0f}s; "
        f"{torch.cuda.memory_allocated(device)/1e9:.1f} GB/rank")

    tc = thinker.config.thinker_text
    head_dim = getattr(thinker.config, "thinker_head_dim", None) or getattr(tc, "head_dim", 128)
    qo, kv = tc.num_attention_heads // world, max(1, tc.num_key_value_heads // world)
    inv_freq = compute_rope_freqs(head_dim, tc.rope_theta, device)
    cache = FlashInferPagedKV(tc.num_hidden_layers, NUM_PAGES, PAGE_SIZE, kv, head_dim,
                              device, torch.bfloat16)
    attn = FlashInferAttention(cache, qo, head_dim, device, max_new_tokens=4096,
                               cudagraph=False, causal=True, dtype=torch.bfloat16)
    handle = FlashInferCacheHandle(attn)
    pages = list(range(NUM_PAGES))
    tmodel, embed, lm_head = thinker.model, thinker.model.model.embed_tokens, thinker.model.lm_head

    # --- tokenize + image processing ---
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model.local_dir, trust_remote_code=True)
    if os.path.exists(IMAGE):
        img = Image.open(IMAGE).convert("RGB")
    else:  # self-contained default: a synthetic scene the model can describe
        from PIL import ImageDraw
        img = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(img)
        d.ellipse([80, 80, 240, 240], fill="red")
        d.rectangle([260, 260, 400, 400], fill="blue")
        d.text((150, 410), "HELLO", fill="black")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": QUESTION}]}]
    out = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    ids = out["input_ids"][0].to(device)
    pixel_values = out["pixel_values"].to(device=device, dtype=enc_dtype)
    grid_thw = out["image_grid_thw"].to(device)

    # --- vision encoder (stateless): -> vision_embeds + deepstack features ---
    with torch.no_grad():
        enc_out = vis_enc(pixel_values, grid_thw=grid_thw)
    if isinstance(enc_out, tuple):
        vision_embeds, deepstack = enc_out
    else:
        vision_embeds = enc_out.pooler_output   # spatial-merged (196), not last_hidden_state (784)
        deepstack = enc_out.deepstack_features
    if isinstance(deepstack, torch.Tensor):
        deepstack = [deepstack]
    vis_mask = ids == IMAGE_PAD
    n_vis = int(vis_mask.sum())
    log(f"[vision] grid={grid_thw.tolist()} -> {vision_embeds.shape[0]} embeds, "
        f"{len(deepstack)} deepstack; {n_vis} placeholders in {ids.shape[0]}-token prompt")
    assert vision_embeds.shape[0] == n_vis, (vision_embeds.shape[0], n_vis)

    # --- merge: scatter vision embeds into text embeddings ---
    with torch.no_grad():
        inp = embed(ids)
        inp = inp.masked_scatter(vis_mask.unsqueeze(-1), vision_embeds.to(inp.dtype))
    # deepstack: full-sequence tensors, features placed at the image positions,
    # added into the Thinker's first layers (thinker.py:234-238).
    hidden = inp.shape[-1]
    ds_full = []
    for ds in deepstack:
        f = torch.zeros(ids.shape[0], hidden, dtype=inp.dtype, device=device)
        f[vis_mask] = ds.to(inp.dtype)
        ds_full.append(f)

    # --- 3D-MRoPE: text increments all 3; image span uses the spatial grid ---
    v0 = int(vis_mask.nonzero()[0])
    seq = ids.shape[0]
    pos = torch.empty(3, seq, dtype=torch.float, device=device)
    pos[:, :v0] = get_rope_index_text(v0, 0.0, device)
    pos[:, v0:v0 + n_vis] = get_rope_index_vision(
        grid_thw, float(v0), position_id_per_seconds=thinker.config.thinker.position_id_per_seconds,
        device=device, spatial_merge_size=thinker.config.vision.spatial_merge_size,
        seconds_per_grid=None)
    nxt = float(pos[:, v0:v0 + n_vis].max().item()) + 1   # grid span > token count
    if v0 + n_vis < seq:
        pos[:, v0 + n_vis:] = get_rope_index_text(seq - v0 - n_vis, nxt, device)
    rope_pos = int(pos.max().item()) + 1   # RoPE position for the first decode token

    def step(embeds, pos3d, kv_seq_pos, n, ds_embeds=None):
        cos, sin = compute_3d_cos_sin(pos3d, inv_freq, MROPE, target_dtype=torch.bfloat16)
        attn.plan(pages, kv_seq_pos, n)   # KV position is token-based
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h, _, _ = tmodel(input_embeds=embeds, cache_handle=handle, cos_sin_3d=(cos, sin),
                             mrope_section=MROPE, deepstack_visual_embeds=ds_embeds)
            return lm_head(h[-1:])

    logits = step(inp, pos, 0, seq, ds_embeds=ds_full)   # prefill with deepstack
    tok = int(logits.argmax(-1).item())
    gen, kv_pos = [tok], seq
    eos = proc.tokenizer.eos_token_id
    for i in range(MAX_NEW):
        if tok == eos:
            break
        p1 = torch.full((3, 1), float(rope_pos + i), dtype=torch.float, device=device)
        with torch.no_grad():
            e = embed(torch.tensor([tok], device=device)).to(torch.bfloat16)
        tok = int(step(e, p1, kv_pos, 1).argmax(-1).item())   # no deepstack in decode
        kv_pos += 1
        gen.append(tok)
    text = proc.tokenizer.decode([t for t in gen if t != eos], skip_special_tokens=True)
    log(f"\n[answer] {text!r}")
    import torch.distributed as dist
    if rank == 0:
        ok = len(text.strip()) > 3
        print(f"\nVISION INPUT {'OK' if ok else 'FAILED'} "
              f"(FA2 encoder + deepstack -> masked-scatter -> TP Thinker prefill -> answer)")
        dist.destroy_process_group()
        return 0 if ok else 1
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
