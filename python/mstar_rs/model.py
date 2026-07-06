"""The model contract for mstar-rs — the Python half of the runtime.

Compare with ``mstar/model/base.py``: `walks()` is `get_graph_walk_graphs()`,
`initial_inputs()` is `get_initial_forward_pass_args()`, `next_forward()` is
`get_partition_forward_pass_args()`, and `execute()` stands in for the engine
layer (all torch compute) until dedicated engines land.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

# (node_name, input_name, [tensor, ...]) triples seeding a walk.
WalkInputs = list[tuple[str, str, list[torch.Tensor]]]
# A walk to start: (walk_name, inputs[, kv_appends[, kv_scratch]]).
# kv_appends = {label: tokens appended per execution of that label's KV
# node} (mstar's per_label_seq_info); kv_scratch = {label: transient tokens
# past the committed sequence each execution also needs pages for}.
NextWalk = (
    tuple[str, WalkInputs]
    | tuple[str, WalkInputs, dict[str, int]]
    | tuple[str, WalkInputs, dict[str, int], dict[str, int]]
)


class Model(ABC):
    @abstractmethod
    def walks(self) -> dict[str, Any]:
        """Named walk graphs (dicts built with mstar_rs.graph helpers)."""

    def kv_config(
        self,
    ) -> tuple[list[tuple[str, int, int]], dict[str, str]] | None:
        """Paged-KV declaration: ([(label, num_pages, page_size)],
        {kv_node_name: label}) or None for stateless models. Mirrors
        mstar's get_kv_cache_config + get_node_engine_types."""
        return None

    def partitions(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Streaming topology: (partition_specs, connection_specs) or None
        for single-partition models. Build with mstar_rs.graph.partition /
        connection. Mirrors mstar's get_partitions + get_partition_topology.
        Every walk from `walks()` must belong to exactly one partition."""
        return None

    def initial_inputs(self, request: dict[str, Any]) -> NextWalk:
        """First walk (+ optional kv_appends) for a new single-partition
        request. Implement this OR `initial_walks` (streaming models seed
        every partition via the latter)."""
        raise NotImplementedError(
            "a model must implement initial_inputs or initial_walks"
        )

    def initial_walks(self, request: dict[str, Any]) -> list[NextWalk]:
        """Walks to seed at ingest, one per partition that starts
        immediately (mstar kicks off all partitions at ingest — e.g. orpheus
        seeds LLM `prefill` AND SNAC `snac_chunk` with empty inputs). Default:
        just the single `initial_inputs` walk."""
        return [self.initial_inputs(request)]

    @abstractmethod
    def execute(
        self,
        node: str,
        walk: str,
        inputs: dict[int, dict[str, list[torch.Tensor]]],
        kv: dict[int, dict[str, Any]] | None = None,
    ) -> dict[int, dict[str, list[torch.Tensor]]]:
        """Run one node for a batch of requests. All torch compute lives
        here. For KV nodes, `kv[rid]` carries the Rust runtime's view:
        {"label", "pages", "seq_pos", "append_len", "scratch_len"}."""

    def next_forward(
        self,
        request_id: int,
        partition: str,
        walk: str,
        fwd_index: int,
        persist: dict[str, list[torch.Tensor]],
        stream_done: bool,
    ) -> NextWalk | None:
        """Policy: after a partition's walk completes, return the next walk
        for that partition, or None to finish it. `stream_done` is True when
        this walk consumed the final chunk of an incoming stream (mstar's
        partition-done-on-the-consuming-pass). A request finishes when every
        partition is done. Default: single forward pass, no continuation."""
        return None

    def loops_to_finish(self) -> list[tuple[int, str]]:
        """After a batch executes (before its outputs are routed), return
        (request_id, loop_name) pairs whose loop should terminate this
        iteration — mstar's `check_stop -> STOP_LOOPS` (e.g. an EOS token
        detected during `execute`). Default: none."""
        return []

    def postprocess(
        self, name: str, modality: str | None, tensors: list[torch.Tensor]
    ) -> Any:
        """Turn an emission into client-facing output. Default: passthrough."""
        return tensors
