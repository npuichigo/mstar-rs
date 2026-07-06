"""mstar-rs: Rust control-plane runtime for M* with Python model execution.

The Rust extension (`mstar_rs._core`) owns graph-walk state, scheduling, and
routing over tensor *descriptors*; this package owns everything that touches
a real ``torch.Tensor``: the uuid -> tensor object store, node execution, and
model policies.
"""

from mstar_rs._core import EMIT_TO_CLIENT, EMPTY_DESTINATION, Runtime

from .driver import Driver
from .graph import edge, emit, loop, node, parallel, sequential
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
    "sequential",
    "parallel",
    "loop",
]
