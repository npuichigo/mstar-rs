"""Orpheus on mstar-rs — the first streaming model (T4 streaming tier).

Two concurrent partitions (mstar's `get_partitions`):

- ``LLM`` — ``prefill`` (embed the voice+text prompt, causal, sample the
  first audio token) then a ``decode`` Loop (one token/step, KV append 1,
  sample with temperature/top-p/repetition-penalty, EOS-stop). Every pass
  streams the sampled ``new_token`` to the SNAC partition.
- ``SNAC`` — a self-triggered ``snac_chunk`` walk. The runtime's stream
  buffer collects tokens under a SlidingWindow(28, 7) policy and feeds each
  28-token window to the vendored SNAC decoder, which emits a 24 kHz int16
  audio chunk. Partition-done rides the pass consuming the final window.

Like pi05, this reuses mstar's own nn modules unmodified — `OrpheusForCausalLM`
behind a FlashInfer causal cache handle (llama3 RoPE), mstar's `Sampler`, and
the vendored `SNAC` decoder + its token->code de-interleaving.

NOTE: `canopylabs/orpheus-3b-0.1-ft` (weights) and `-0.1-pretrained`
(tokenizer) are gated on HuggingFace; this runs once that access is granted.
"""

from __future__ import annotations

from typing import Any

import torch

from ..fi import FlashInferAttention, FlashInferCacheHandle, FlashInferPagedKV
from ..graph import (
    connection,
    edge,
    emit,
    loop,
    node,
    partition,
    sliding_window,
    stream_edge,
    EMPTY_DESTINATION,
)
from ..model import Model, NextWalk

DEFAULT_MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"
KV_LABEL = "main"
PAGE_SIZE = 128
NUM_PAGES = 256  # 32k tokens; prompt is short, decode up to max_output_tokens


class Orpheus(Model):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        max_output_tokens: int = 2048,
    ) -> None:
        from mstar.model.orpheus.orpheus_model import OrpheusModel
        from mstar.utils.sampling import Sampler

        self.device = torch.device(device)
        self.max_output_tokens = max_output_tokens
        self._mstar = OrpheusModel(model_path_hf=model_id)
        self.cfg = self._mstar.config
        self.tokenizer = self._mstar.tokenizer
        llm_sub = self._mstar.get_submodule(
            "LLM", device=str(device), autocast_dtype=torch.bfloat16
        )
        self.embed_tokens = llm_sub.embed_tokens
        self.language_model = llm_sub.language_model  # OrpheusForCausalLM
        self.lm_head = llm_sub.lm_head
        self.snac_sub = self._mstar.get_submodule("snac_decoder", device=str(device))

        self.kv_cache = FlashInferPagedKV(
            num_layers=self.cfg.num_layers,
            num_pages=NUM_PAGES,
            page_size=PAGE_SIZE,
            num_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        # Causal attention for both prefill (multi-token) and decode (1 token).
        self._prefill_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_qo_heads, self.cfg.head_dim,
            self.device, max_new_tokens=2048, cudagraph=False, causal=True,
        )
        self._decode_attn = FlashInferAttention(
            self.kv_cache, self.cfg.num_qo_heads, self.cfg.head_dim,
            self.device, max_new_tokens=1, cudagraph=False, causal=True,
        )
        self.sampler = Sampler(device=self.device)
        self._pending_stops: list[tuple[int, str]] = []

    # -- graph + topology -------------------------------------------------

    def walks(self) -> dict[str, Any]:
        return {
            "prefill": node(
                "LLM",
                ["text_inputs"],
                [
                    edge(EMPTY_DESTINATION, "new_token", persist=True),
                    stream_edge("snac_decoder", "new_token", "SNAC"),
                ],
            ),
            "decode": loop(
                "decode_loop",
                node(
                    "LLM",
                    ["text_inputs"],
                    [
                        edge("LLM", "text_inputs"),
                        stream_edge("snac_decoder", "new_token", "SNAC"),
                    ],
                ),
                max_iters=self.max_output_tokens,
            ),
            "snac_chunk": node(
                "snac_decoder",
                ["new_token"],
                [emit("audio_chunk", modality="audio")],
            ),
        }

    def kv_config(self):
        return [(KV_LABEL, NUM_PAGES, PAGE_SIZE)], {"LLM": KV_LABEL}

    def partitions(self):
        return (
            [
                partition("LLM", ["prefill", "decode"]),
                partition("SNAC", ["snac_chunk"]),
            ],
            [
                connection(
                    "LLM", "SNAC", "new_token",
                    sliding_window(
                        self.cfg.snac_window_tokens, self.cfg.snac_stride_tokens
                    ),
                )
            ],
        )

    # -- request ingestion ------------------------------------------------

    def _tokenize(self, prompt: str, voice: str) -> torch.Tensor:
        toks = self.tokenizer(f"{voice}: {prompt}", return_tensors="pt").input_ids[0]
        start = torch.tensor([self.cfg.start_token_id], dtype=torch.long)
        end = torch.tensor(self.cfg.end_token_ids, dtype=torch.long)
        return torch.cat([start, toks, end]).to(self.device)

    def initial_walks(self, request: dict[str, Any]) -> list[NextWalk]:
        rid = request["request_id"]
        text_ids = self._tokenize(request.get("prompt", ""), request.get("voice", "tara"))
        # Per-request sampler state (faithful to mstar's config + seed).
        srid = str(rid)
        self.sampler.add_request(srid)
        self.sampler.set_config(
            srid,
            vocab_size=self.cfg.vocab_size,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            ignore_eos=False,
        )
        self.sampler._sampling_config[srid].set_seed(request.get("seed", 0))
        # LLM prefill (KV append = prompt length) + self-triggered SNAC walk.
        return [
            ("prefill", [("LLM", "text_inputs", [text_ids])], {KV_LABEL: text_ids.shape[0]}),
            ("snac_chunk", []),
        ]

    # -- node execution ----------------------------------------------------

    def _run_llm(self, text_ids: torch.Tensor, view, attn: FlashInferAttention):
        """Embed -> causal LLM over the paged cache -> logits for the last
        token. `text_ids` is the prompt (prefill) or the single prev token."""
        attn.plan(view["pages"], view["seq_pos"], view["append_len"])
        handle = FlashInferCacheHandle(attn)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = self.embed_tokens(text_ids)
            hidden = self.language_model(emb, cache_handle=handle)
            logits = self.lm_head(hidden[-1:])  # (1, vocab)
        return logits

    @torch.inference_mode()
    def execute(self, node_name, walk, inputs, kv=None):
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        for rid, named in inputs.items():
            if node_name == "LLM":
                attn = self._prefill_attn if walk == "prefill" else self._decode_attn
                logits = self._run_llm(named["text_inputs"][0], kv[rid], attn)
                token = self.sampler.sample([str(rid)], logits, apply_penalty=True)
                token = token.view(1).to(torch.long)  # (1,)
                # EOS -> stop the decode loop (mstar's check_stop).
                if walk == "decode" and int(token.item()) == self.cfg.stop_token_id:
                    self._pending_stops.append((rid, "decode_loop"))
                out = {"new_token": [token]}
                if walk == "decode":
                    out["text_inputs"] = [token]  # loop-back
                outputs[rid] = out
            elif node_name == "snac_decoder":
                # A window of token-id tensors (in order) -> one audio chunk.
                window = torch.stack(named["new_token"]).flatten().to(self.device)
                codes = self.snac_sub._tokens_to_codes(window)  # (N, 4, 7)
                c0, c1, c2 = self.snac_sub._extract_snac_codes(codes)
                with torch.inference_mode():
                    audio = self.snac_sub.snac_model.decode([c0, c1, c2])
                audio = audio[
                    :, :, self.cfg.snac_audio_slice_start:self.cfg.snac_audio_slice_end
                ]
                pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).squeeze(1).squeeze(0)
                outputs[rid] = {"audio_chunk": [pcm]}
            else:
                raise ValueError(f"unknown node: {node_name}")
        return outputs

    # -- policy ------------------------------------------------------------

    def next_forward(
        self, request_id, partition, walk, fwd_index, persist, stream_done
    ) -> NextWalk | None:
        if partition == "LLM":
            if walk == "prefill":
                # Seed decode's first token with the token sampled at prefill.
                first = persist["new_token"]
                return (
                    "decode",
                    [("LLM", "text_inputs", first)],
                    {KV_LABEL: 1},
                )
            self.sampler.remove_request(str(request_id))
            return None  # decode finished -> LLM partition done
        # SNAC: re-arm until the final window is consumed.
        return None if stream_done else ("snac_chunk", [])

    def loops_to_finish(self) -> list[tuple[int, str]]:
        stops = self._pending_stops
        self._pending_stops = []
        return stops

    def postprocess(self, name, modality, tensors):
        assert modality == "audio"
        return tensors[0].detach().cpu()  # int16 PCM, 24 kHz
