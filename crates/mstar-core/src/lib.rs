//! mstar-core: the M* graph IR and per-request walk state machine.
//!
//! This crate is the control-plane heart of mstar-rs. It knows nothing about
//! torch, GPUs, sockets, or processes: tensors exist only as [`TensorRef`]
//! descriptors (uuid + dims + dtype), mirroring Python mstar's rule that the
//! control plane carries `TensorPointerInfo` descriptors while tensor bytes
//! move out-of-band.
//!
//! Semantics are ported from `mstar/graph/base.py`:
//! - a node is ready when `input_names ⊆ received inputs`;
//! - an input arriving at a node that already has it (or already ran this
//!   iteration) buffers into the node's *next-iteration* slot — this is how
//!   loop-back edges work (`ready_next_iter`);
//! - loops terminate on `max_iters` or an explicit finish signal, snapshot
//!   `outputs` from the last iteration and append `accumulated_outputs`
//!   across iterations, and re-inject external inputs on each advance.

pub mod error;
pub mod graph;
pub mod tensor;
pub mod walk;

pub use error::{CoreError, Result};
pub use graph::{
    CompiledLoop, CompiledWalk, EdgeSpec, LoopSpec, NodeSpec, Section, WalkSet,
    EMIT_TO_CLIENT, EMPTY_DESTINATION,
};
pub use tensor::{TensorRef, Uuid};
pub use walk::{CompletionResult, IncomingInput, RouteEvent, WalkState};
