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
) -> dict[str, Any]:
    return {
        "next_node": next_node,
        "name": name,
        "persist": persist,
        "output_modality": output_modality,
    }


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
