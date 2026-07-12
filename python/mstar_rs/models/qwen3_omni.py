"""Qwen3-Omni on mstar-rs — the flagship 3-partition dual-AR streaming model.

Ports `mstar/model/qwen3_omni/` (Qwen/Qwen3-Omni-30B-A3B-Instruct): a Thinker
(30B-A3B MoE, text AR) whose per-token hidden states stream to a Talker (MoE
audio-token AR, with a nested 15-step code-predictor depth loop per frame),
whose codec tokens stream to a Code2Wav vocoder (24 kHz audio). Multimodal in
(text/audio/image/video via audio_encoder + vision_encoder), text + speech out.

This file is being brought up in two stages:

  * **Stage 1 (here): the control plane** — `Qwen3OmniPolicy` declares the 8
    walks, 3 partitions, 3 streaming connections, and 2 KV labels. This
    validates that the Rust runtime can *express and compile* qwen3-omni's
    dual-AR streaming topology (the hardest control shape so far) with NO model
    weights — see `examples/verify_qwen3_graph.py`.
  * **Stage 2 (next): the engine** — `Qwen3OmniEngine` will load the Thinker /
    Talker / Code2Wav + encoders (reusing mstar's nn modules behind `fi.py`
    handles) and run `execute`, verified vs a dense reference on GPU. That
    needs the ~60 GB weights + the dual RoPE regimes, the nested depth loop,
    the cross-partition hidden-state streaming, and the conductor `talker_
    trigger` gating — none of which have an orpheus analog.

Topology (from mstar's get_graph_walk_graphs / get_partitions / get_partition_
topology):

    Thinker ──thinker_states/mask (fixed chunk 1)──▶ Talker
                                                       │
                                          codec_tokens (left-context 25+25)
                                                       ▼
                                                   Code2Wav ──▶ audio
"""

from __future__ import annotations

from typing import Any

from ..graph import (
    EMPTY_DESTINATION,
    connection,
    edge,
    emit,
    fixed_chunk,
    left_context,
    loop,
    node,
    partition,
    sequential,
    stream_edge,
)
from ..model import ModelPolicy

DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
THINKER_KV = "thinker"
TALKER_KV = "talker"
PAGE_SIZE = 128
THINKER_PAGES = 512   # 64k text-context tokens
TALKER_PAGES = 256
MAX_OUTPUT_TOKENS = 2048  # mstar: get_max_output_tokens() (both AR loops)
# Talker→Code2Wav windowing (mstar: codec_chunk_frames / codec_left_context).
CODEC_CHUNK_FRAMES = 25
CODEC_LEFT_CONTEXT_FRAMES = 25


class Qwen3OmniPolicy(ModelPolicy):
    """Control plane: the 3-partition dual-AR streaming graph. Weightless —
    the Thinker/Talker/Code2Wav weights live in `Qwen3OmniEngine` (Stage 2)."""

    def __init__(self, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> None:
        self.max_output_tokens = max_output_tokens

    # -- walks: 8 (4 Thinker, 3 Talker, 1 Code2Wav) -----------------------

    def _thinker_out(self) -> list[dict[str, Any]]:
        """Thinker's outward edges: emit the text token to the client, and
        stream the (reduced) hidden states + mask to the Talker, one per step
        (FixedChunkPolicy chunk=1)."""
        return [
            emit("new_token", modality="text", persist=True),
            stream_edge("Talker", "thinker_states", "Talker"),
            stream_edge("Talker", "thinker_mask", "Talker"),
        ]

    def _talker_out(self) -> list[dict[str, Any]]:
        """Talker's outward edge: stream the 16 codec tokens/frame to Code2Wav
        (LeftContextChunkPolicy 25+25)."""
        return [stream_edge("Code2Wav", "codec_tokens", "Code2Wav")]

    def walks(self) -> dict[str, Any]:
        return {
            # --- Thinker partition ---
            "prefill_text": node("Thinker", ["text_inputs"], self._thinker_out()),
            "prefill_audio": sequential(
                node("audio_encoder", ["audio_features", "audio_seqlens"],
                     [edge("Thinker", "audio_embeds")]),
                node("Thinker", ["audio_embeds"], self._thinker_out()),
            ),
            "prefill_vision": sequential(
                node("vision_encoder", ["pixel_values", "image_grid_thw"],
                     [edge("Thinker", "vision_embeds"), edge("Thinker", "deepstack")]),
                node("Thinker",
                     ["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
                     self._thinker_out()),
            ),
            "thinker_decode": loop(
                "thinker_decode_loop",
                node("Thinker", ["text_inputs"],
                     [edge("Thinker", "text_inputs")] + self._thinker_out()),  # self-feedback + emit/stream
                max_iters=self.max_output_tokens,
            ),
            # --- Talker partition (fed streaming thinker_states + a conductor
            # talker_trigger; the last prefill step starts emitting codec) ---
            "talker_prefill": node(
                "Talker", ["thinker_states", "thinker_mask", "talker_trigger"], []
            ),  # KV-extend only, no outputs
            "talker_last_prefill": sequential(
                node("Talker", ["thinker_states", "thinker_mask", "talker_trigger"],
                     [edge(EMPTY_DESTINATION, "talker_input_embeds", persist=True)]
                     + self._talker_out()),
            ),
            "talker_decode": loop(
                "talker_decode_loop",
                node("Talker", ["thinker_states", "thinker_mask", "talker_input_embeds"],
                     [edge("Talker", "talker_input_embeds")] + self._talker_out()),  # self-feedback + stream
                max_iters=self.max_output_tokens,
            ),
            # --- Code2Wav partition ---
            "code2wav_chunk": node("Code2Wav", ["codec_tokens"],
                                   [emit("audio_chunk", modality="audio")]),
        }

    # -- partitions + streaming topology ----------------------------------

    def partitions(self):
        return (
            [
                partition("Thinker", ["prefill_text", "prefill_audio",
                                      "prefill_vision", "thinker_decode"]),
                partition("Talker", ["talker_prefill", "talker_last_prefill",
                                     "talker_decode"]),
                partition("Code2Wav", ["code2wav_chunk"]),
            ],
            [
                # Thinker feeds the Talker one hidden-state token at a time;
                # continue_after_done keeps the Talker's stream alive past the
                # Thinker's EOS so its own decode can drain.
                connection("Thinker", "Talker", "thinker_states",
                           fixed_chunk(1, continue_after_done=True)),
                connection("Thinker", "Talker", "thinker_mask",
                           fixed_chunk(1, continue_after_done=True)),
                # Talker feeds Code2Wav in 25-frame chunks with 25-frame
                # left-context overlap (warms the causal ConvNet; trimmed from
                # emitted audio in the vocoder).
                connection("Talker", "Code2Wav", "codec_tokens",
                           left_context(CODEC_CHUNK_FRAMES, CODEC_LEFT_CONTEXT_FRAMES)),
            ],
        )

    def kv_config(self):
        # Two separate paged caches — one per AR partition (mstar keeps a
        # Thinker cache and a Talker cache, both label "main"; flattened here to
        # distinct labels). The Code Predictor's dense depth-loop KV is a plain
        # tensor inside the engine, not an engine KV label.
        return (
            [(THINKER_KV, THINKER_PAGES, PAGE_SIZE), (TALKER_KV, TALKER_PAGES, PAGE_SIZE)],
            {"Thinker": THINKER_KV, "Talker": TALKER_KV},
        )
