"""A weightless toy autoregressive model — the frontend↔conductor bridge's
end-to-end test target, and the demonstrator for the policy/engine split.

It "generates" by echoing the prompt token-ids back one per forward pass: the
`step` node takes the remaining token-ids as `state`, emits the first as a
`text` token (EMIT_TO_CLIENT), and persists the rest. The policy
(`next_forward`) restarts `gen` with the remainder until it is empty. This is
deliberately the smallest thing that exercises the real serving path — one
emission per decode step, continued across forward passes — with no CUDA, no
checkpoint, and no math, so a CPU box can prove the whole
HTTP → axum → conductor → worker → streamed-tokens → SSE loop.

It is split into `EchoPolicy` (control plane — no weights) and `EchoEngine`
(data plane — where `execute` and any weights would live) to show the
conductor/worker separation: a `Conductor(EchoPolicy())` never constructs the
engine, so it never loads weights. `EchoAR` bundles both for the
single-process `Driver` and for passing one object to both roles.
"""

from __future__ import annotations

from typing import Any

import torch

from ..graph import EMPTY_DESTINATION, edge, emit, node, sequential
from ..model import Model, ModelEngine, ModelPolicy, WalkInputs


class EchoPolicy(ModelPolicy):
    """Control plane: graph, walk seeding, continuation, postprocess. No
    weights — this is what the conductor holds."""

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
        if not tokens:
            # An empty prompt (or max_tokens=0) has nothing to echo; the gen
            # walk always emits one token, so guard here instead of emitting an
            # empty tensor that would crash postprocess's `.item()`.
            raise ValueError("EchoAR requires at least one prompt token (after max_tokens)")
        state = torch.tensor(tokens, dtype=torch.int64)
        return "gen", [("step", "state", [state])]

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


class EchoEngine(ModelEngine):
    """Data plane: `execute` (and, for a real model, the weights). This is
    what the worker holds. The `weights_loaded` sentinel stands in for a real
    model's GPU weight allocation — it lets a test assert the conductor never
    constructs the engine."""

    def __init__(self) -> None:
        self.weights_loaded = True  # a real engine would load checkpoints here

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


class EchoAR(EchoPolicy, EchoEngine, Model):
    """Both halves in one object — for the single-process `Driver`, or for
    handing the same instance to both conductor and worker."""

    def __init__(self) -> None:
        EchoEngine.__init__(self)
