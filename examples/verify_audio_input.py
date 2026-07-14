"""Multimodal AUDIO INPUT: feed an audio clip to the Qwen3-Omni Thinker and get
a text answer about it. The audio encoder (HF, FlashAttention-2) is stateless;
its embeddings are masked-scattered into the Thinker's prefill at the expanded
audio-placeholder positions, with audio-aware 3D-MRoPE. The Thinker runs
tensor-parallel (it's ~39B); the encoder is replicated on each rank.

Reuses mstar's own modules — get_submodule("audio_encoder") (FA2 varlen path),
the Thinker's embed_tokens + thinker.model, and rope.py — behind mstar-rs's
FlashInferCacheHandle. Model-specific glue (feature extraction, placeholder
masked-scatter, audio rope) is replicated per mstar.

    QWEN3_AUDIO_WAV=/path/to/audio.wav CUDA_VISIBLE_DEVICES=1,6 \\
        torchrun --nproc_per_node=2 --master_port=29560 examples/verify_audio_input.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar.distributed.communication import TPCommGroup, WorkerTPGroups  # noqa: E402
from mstar.model.qwen3_omni.components.rope import (  # noqa: E402
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
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
AUDIO_PAD, AUDIO_BOS, AUDIO_EOS = 151675, 151669, 151670
# Path to a .wav to describe; override with QWEN3_AUDIO_WAV.
AUDIO_WAV = os.environ.get("QWEN3_AUDIO_WAV", "audio.wav")
QUESTION = "What do you hear in this audio? Answer in one sentence."
NUM_PAGES, PAGE_SIZE, MAX_NEW = 64, 128, 48


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(0)

    def log(*a: object) -> None:
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
    audio_enc = model.get_submodule("audio_encoder", device=str(device)).audio_encoder
    audio_enc = audio_enc.to(device)
    # the sinusoidal PE is a plain tensor attribute (not a registered buffer),
    # so .to() skips it — move it (and any other stray CPU tensors) explicitly.
    for mod in audio_enc.modules():
        for name, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                setattr(mod, name, val.to(device))
    enc_dtype = next(audio_enc.parameters()).dtype
    log(f"[load] Thinker(shard)+audio_encoder in {time.time()-t0:.0f}s; "
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

    # --- feature extraction + tokenization (both ranks, deterministic) ---
    import soundfile as sf
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model.local_dir, trust_remote_code=True)
    wav, sr = sf.read(AUDIO_WAV)
    if sr != 16000:  # feature extractor is 16 kHz
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(np.linspace(0, len(wav), n, endpoint=False), np.arange(len(wav)), wav)
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": wav},
                                         {"type": "text", "text": QUESTION}]}]
    out = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    ids = out["input_ids"][0].to(device)
    feats = out["input_features"].permute(0, 2, 1)[out["feature_attention_mask"].bool()].permute(1, 0)
    audio_features = feats.to(device=device, dtype=enc_dtype)
    audio_seqlens = out["feature_attention_mask"].sum(-1).to(torch.long).to(device)

    # --- run the FA2 audio encoder (stateless) ---
    with torch.no_grad():
        audio_embeds = audio_enc(audio_features, feature_lens=audio_seqlens,
                                 return_dict=True).last_hidden_state
    if audio_embeds.dim() == 3:
        audio_embeds = audio_embeds.squeeze(0)
    audio_mask = ids == AUDIO_PAD
    n_aud = int(audio_mask.sum())
    log(f"[audio] {int(audio_seqlens.item())} mel frames -> {audio_embeds.shape[0]} embeds; "
        f"{n_aud} placeholders in {ids.shape[0]}-token prompt")
    assert audio_embeds.shape[0] == n_aud, (audio_embeds.shape[0], n_aud)

    # --- merge: scatter audio embeds into the text embeddings ---
    with torch.no_grad():
        inp = embed(ids)
        inp = inp.masked_scatter(audio_mask.unsqueeze(-1), audio_embeds.to(inp.dtype))

    # --- 3D-MRoPE: text increments all 3; audio span increments temporal only ---
    a0 = int(audio_mask.nonzero()[0])
    seq = ids.shape[0]
    pos = torch.empty(3, seq, dtype=torch.float, device=device)
    pos[:, :a0] = get_rope_index_text(a0, 0.0, device)
    pos[:, a0:a0 + n_aud] = get_rope_index_audio(n_aud, float(a0), device,
                                                 thinker.config.thinker.position_id_per_seconds)
    if a0 + n_aud < seq:
        pos[:, a0 + n_aud:] = get_rope_index_text(seq - a0 - n_aud, float(a0 + n_aud), device)

    def step(embeds, pos3d, seq_pos, n):
        cos, sin = compute_3d_cos_sin(pos3d, inv_freq, MROPE, target_dtype=torch.bfloat16)
        attn.plan(pages, seq_pos, n)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden, _, _ = tmodel(input_embeds=embeds, cache_handle=handle,
                                  cos_sin_3d=(cos, sin), mrope_section=MROPE)
            return lm_head(hidden[-1:])

    # prefill (audio+text) then greedy decode
    logits = step(inp, pos, 0, seq)
    tok = int(logits.argmax(-1).item())
    gen, sp = [tok], seq
    eos = proc.tokenizer.eos_token_id
    for _ in range(MAX_NEW):
        if tok == eos:
            break
        p1 = torch.full((3, 1), float(sp), dtype=torch.float, device=device)
        with torch.no_grad():
            e = embed(torch.tensor([tok], device=device)).to(torch.bfloat16)
        tok = int(step(e, p1, sp, 1).argmax(-1).item())
        sp += 1
        gen.append(tok)
    text = proc.tokenizer.decode([t for t in gen if t != eos], skip_special_tokens=True)
    log(f"\n[answer] {text!r}")
    if rank == 0:
        ok = len(text.strip()) > 3
        print(f"\nAUDIO INPUT {'OK' if ok else 'FAILED'} "
              f"(FA2 encoder -> masked-scatter -> TP Thinker prefill -> answer)")
        import torch.distributed as dist
        dist.destroy_process_group()
        return 0 if ok else 1
    import torch.distributed as dist
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
