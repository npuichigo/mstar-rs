"""mstar-rs: Rust control-plane runtime for M* with Python model execution.

The Rust extension (`mstar_rs._core`) owns graph-walk state, scheduling, and
routing over tensor *descriptors*; this package owns everything that touches
a real ``torch.Tensor``: the uuid -> tensor object store, node execution, and
model policies.
"""

from mstar_rs._core import EMIT_TO_CLIENT, EMPTY_DESTINATION, Runtime

from .driver import Driver
from .graph import (
    connection,
    edge,
    emit,
    fixed_chunk,
    left_context,
    loop,
    node,
    parallel,
    partition,
    ramp_sliding_window,
    sequential,
    sliding_window,
    stream_edge,
)
from .model import Model
from .store import TensorStore

__all__ = [
    "EMIT_TO_CLIENT",
    "EMPTY_DESTINATION",
    "Runtime",
    "Driver",
    "Model",
    "TensorStore",
    "node",
    "edge",
    "emit",
    "stream_edge",
    "sequential",
    "parallel",
    "loop",
    "partition",
    "connection",
    "sliding_window",
    "ramp_sliding_window",
    "left_context",
    "fixed_chunk",
]
