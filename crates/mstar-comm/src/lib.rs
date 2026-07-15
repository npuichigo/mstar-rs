//! mstar-comm: the control-plane message mesh for multi-process mstar-rs.
//!
//! Mirrors mstar's ZMQ PUSH/PULL layer (`communication/communicator.py`)
//! directly: each entity binds one **PULL inbox** at `ipc://<dir>/<id>.ipc`
//! and connects a lazily-cached **PUSH** socket per peer. Sends are
//! fire-and-forget (queued, and auto-reconnecting on peer restart — libzmq
//! handles both); receives drain the inbox. Messages are msgpack-serialized
//! (a real wire format replacing mstar's pickle), one message per zmq frame,
//! so the envelope schema is explicit and cross-process-stable.
//!
//! Transport and encoding are separate layers (the migration seam):
//! [`RawZmqCommunicator`] moves opaque byte frames — pickle, msgpack, or
//! msgpack pass through untouched — over ipc *or* tcp endpoints, with
//! wakeup-fd polling (an eventfd wakes the receive loop immediately);
//! [`ZmqCommunicator`] adds a typed [`Codec`] on top ([`MsgpackCodec`] by
//! default). Message *types* are the runtime's concern.

mod communicator;
pub mod shm;

pub use communicator::{
    Codec, CommError, MsgpackCodec, RawZmqCommunicator, RecvEvent, ZmqCommunicator,
};
pub use shm::{SegmentedShmArena, ShmArena, ShmError, ALIGN};
