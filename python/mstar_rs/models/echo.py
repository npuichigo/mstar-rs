"""A weightless toy autoregressive model — the frontend↔conductor bridge's
end-to-end test target.

It "generates" by echoing the prompt token-ids back one per forward pass: the
`step` node takes the remaining token-ids as `state`, emits the first as a
`text` token (EMIT_TO_CLIENT), and persists the rest. The policy
(`next_forward`) restarts `gen` with the remainder until it is empty. This is
deliberately the smallest thing that exercises the real serving path — one
emission per decode step, continued across forward passes — with no CUDA, no
checkpoint, and no math, so a CPU box can prove the whole
HTTP → axum → conductor → worker → streamed-tokens → SSE loop.
"""

from __future__ import annotations

from typing import Any

import torch

from ..graph import EMPTY_DESTINATION, edge, emit, node, sequential
from ..model import Model, WalkInputs


class EchoAR(Model):
    def walks(self) -> dict[str, Any]:
        return {
            # One decode step: emit the head token to the client, persist the
            # tail for the policy to feed back in.
            "gen": sequential(
                node(
                    "step",
                    ["state"],
                    [
                        emit("token", modality="text"),
                        edge(EMPTY_DESTINATION, "rest", persist=True),
                    ],
                )
            )
        }

    def initial_inputs(self, request: dict[str, Any]) -> tuple[str, WalkInputs]:
        tokens = list(request["tokens"])[: request.get("max_tokens", len(request["tokens"]))]
        state = torch.tensor(tokens, dtype=torch.int64)
        return "gen", [("step", "state", [state])]

    @torch.inference_mode()
    def execute(
        self,
        node_name: str,
        walk: str,
        inputs: dict[int, dict[str, list[torch.Tensor]]],
        kv=None,
    ) -> dict[int, dict[str, list[torch.Tensor]]]:
        outputs: dict[int, dict[str, list[torch.Tensor]]] = {}
        for rid, named in inputs.items():
            state = named["state"][0]
            outputs[rid] = {"token": [state[:1]], "rest": [state[1:]]}
        return outputs

    def next_forward(
        self,
        request_id: int,
        partition: str,
        walk: str,
        fwd_index: int,
        persist: dict[str, list[torch.Tensor]],
        stream_done: bool,
    ):
        rest = persist["rest"][0]
        if rest.numel() == 0:
            return None  # emitted the whole prompt -> request done
        return "gen", [("step", "state", [rest])]

    def postprocess(self, name, modality, tensors) -> int:
        return int(tensors[0].item())
