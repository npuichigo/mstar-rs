"""V-JEPA 2 on mstar-rs — the first model.

Ports the core walks of ``mstar/model/vjepa2/vjepa2_model.py``:

- ``prefill_video``:  video_frames -> [video_encoder] -> encoder_hidden
                      -> [predictor] -> predicted_hidden -> EMIT_TO_CLIENT
- ``prefill_video_encoder_only``: video_frames -> [video_encoder]
                      -> encoder_hidden -> EMIT_TO_CLIENT

Both nodes are stateless (no KV cache), which is exactly why this is the
first target: it proves the whole conductor -> scheduler -> engine -> emit
path with the smallest possible data plane.

Compute uses HF transformers' ``VJEPA2Model`` (encoder + predictor
submodules), the same checkpoints mstar's native vjepa2 loads.
"""

from __future__ import annotations

from typing import Any

import torch

from ..graph import emit, edge, node, sequential
from ..model import Model, WalkInputs

DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2(Model):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        autocast_dtype: torch.dtype | None = None,
        compile_modules: bool = False,
    ) -> None:
        """autocast_dtype/compile_modules mirror mstar's StatelessEngine fast
        path (bf16 autocast + torch.compile'd submodules)."""
        from transformers import AutoVideoProcessor, VJEPA2Model

        self.device = torch.device(device)
        self.dtype = dtype
        self.autocast_dtype = autocast_dtype
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.hf_model = (
            VJEPA2Model.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        )
        self._encoder_fwd = self.hf_model.encoder
        self._predictor_fwd = self.hf_model.predictor
        if compile_modules:
            self._encoder_fwd = torch.compile(self.hf_model.encoder)
            self._predictor_fwd = torch.compile(self.hf_model.predictor)

    # -- graph declaration ------------------------------------------------

    def walks(self) -> dict[str, Any]:
        return {
            "prefill_video": sequential(
                node(
                    "video_encoder",
                    ["video_frames"],
                    [edge("predictor", "encoder_hidden")],
                ),
                node(
                    "predictor",
                    ["encoder_hidden"],
                    [emit("predicted_hidden", modality="video", persist=True)],
                ),
            ),
            "prefill_video_encoder_only": sequential(
                node(
                    "video_encoder",
                    ["video_frames"],
                    [emit("encoder_hidden", modality="video", persist=True)],
                ),
            ),
        }

    # -- request ingestion (mstar: process_prompt) ------------------------

    def preprocess_video(self, frames) -> torch.Tensor:
        """frames: [T, H, W, C] uint8 (numpy or tensor) -> pixel_values_videos
        [1, T, C, H', W'] on the target device."""
        processed = self.processor(frames, return_tensors="pt")
        return processed["pixel_values_videos"].to(self.device, self.dtype)

    def initial_inputs(self, request: dict[str, Any]) -> tuple[str, WalkInputs]:
        pixel_values = request["pixel_values_videos"]
        if pixel_values.dim() == 4:  # [T, C, H, W] -> add batch dim
            pixel_values = pixel_values.unsqueeze(0)
        walk = request.get("walk", "prefill_video")
        return walk, [("video_encoder", "video_frames", [pixel_values])]

    # -- node execution (mstar: StatelessEngine + submodules) --------------

    @torch.inference_mode()
    def execute(
        self,
        node_name: str,
        walk: str,
        inputs: dict[int, dict[str, list[torch.Tensor]]],
        kv=None,
    ) -> dict[int, dict[str, list[torch.Tensor]]]:
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        autocast = torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        )
        with autocast:
            if node_name == "video_encoder":
                for rid, named in inputs.items():
                    hidden = self._encoder_fwd(
                        pixel_values_videos=named["video_frames"][0]
                    ).last_hidden_state
                    outputs[rid] = {"encoder_hidden": [hidden]}
            elif node_name == "predictor":
                for rid, named in inputs.items():
                    hidden = named["encoder_hidden"][0]
                    batch, num_patches = hidden.shape[0], hidden.shape[1]
                    # Full-coverage masks, as mstar's vjepa2 submodule defaults.
                    positions = (
                        torch.arange(num_patches, device=hidden.device)
                        .unsqueeze(0)
                        .repeat(batch, 1)
                    )
                    predicted = self._predictor_fwd(
                        encoder_hidden_states=hidden,
                        context_mask=[positions],
                        target_mask=[positions],
                    ).last_hidden_state
                    outputs[rid] = {"predicted_hidden": [predicted]}
            else:
                raise ValueError(f"unknown node: {node_name}")
        return outputs

    # next_forward: default None — one forward pass, then done.
