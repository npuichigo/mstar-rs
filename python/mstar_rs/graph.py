"""Builders for the walk-graph JSON the Rust core consumes.

Mirrors the construction API of ``mstar/graph/base.py`` (`GraphNode`,
`GraphEdge`, `Sequential`, `Parallel`, `Loop`) as plain dicts matching
mstar-core's serde `Section` schema.
"""

from __future__ import annotations

from typing import Any

EMIT_TO_CLIENT = "EMIT_TO_CLIENT"
EMPTY_DESTINATION = "EMPTY_DESTINATION"


def edge(
    next_node: str,
    name: str,
    *,
    persist: bool = False,
    output_modality: str | None = None,
    target_partition: str | None = None,
) -> dict[str, Any]:
    return {
        "next_node": next_node,
        "name": name,
        "persist": persist,
        "output_modality": output_modality,
        "target_partition": target_partition,
    }


def stream_edge(next_node: str, name: str, target_partition: str) -> dict[str, Any]:
    """A ``StreamingGraphEdge``: routes into the stream buffer of the
    connection targeting `target_partition` instead of a node in this walk."""
    return edge(next_node, name, target_partition=target_partition)


def emit(name: str, *, modality: str | None = None, persist: bool = False) -> dict[str, Any]:
    """Edge to the client (``EMIT_TO_CLIENT``)."""
    return edge(EMIT_TO_CLIENT, name, persist=persist, output_modality=modality)


def node(name: str, input_names: list[str], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "node",
        "name": name,
        "input_names": input_names,
        "outputs": outputs,
    }


def sequential(*sections: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "sequential", "sections": list(sections)}


def parallel(*sections: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "parallel", "sections": list(sections)}


def loop(
    name: str,
    body: dict[str, Any],
    max_iters: int,
    *,
    outputs: list[dict[str, Any]] | None = None,
    accumulated_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "loop",
        "name": name,
        "body": body,
        "max_iters": max_iters,
        "outputs": outputs or [],
        "accumulated_outputs": accumulated_outputs or [],
    }


# --- streaming topology (mstar PartitionDefinition / Connection) ---


def partition(name: str, walks: list[str]) -> dict[str, Any]:
    return {"name": name, "walks": walks}


def sliding_window(window: int, stride: int) -> dict[str, Any]:
    return {"kind": "sliding_window", "window": window, "stride": stride}


def ramp_sliding_window(
    first_window: int, first_stride: int, window: int, stride: int
) -> dict[str, Any]:
    return {
        "kind": "ramp_sliding_window",
        "first_window": first_window,
        "first_stride": first_stride,
        "window": window,
        "stride": stride,
    }


def left_context(chunk: int, left_context: int) -> dict[str, Any]:  # noqa: A002
    return {"kind": "left_context", "chunk": chunk, "left_context": left_context}


def fixed_chunk(chunk_size: int, *, continue_after_done: bool = False) -> dict[str, Any]:
    return {
        "kind": "fixed",
        "chunk_size": chunk_size,
        "continue_after_done": continue_after_done,
    }


def connection(
    from_partition: str, to_partition: str, edge_name: str, policy: dict[str, Any]
) -> dict[str, Any]:
    return {
        "from": from_partition,
        "to": to_partition,
        "edge_name": edge_name,
        "policy": policy,
    }
