//! mstar-comm: the control-plane message mesh for multi-process mstar-rs.
//!
//! Mirrors mstar's ZMQ PUSH/PULL layer (`communication/communicator.py`)
//! directly: each entity binds one **PULL inbox** at `ipc://<dir>/<id>.ipc`
//! and connects a lazily-cached **PUSH** socket per peer. Sends are
//! fire-and-forget (queued, and auto-reconnecting on peer restart — libzmq
//! handles both); receives drain the inbox. Messages are bincode-serialized
//! (a real wire format replacing mstar's pickle), one message per zmq frame,
//! so the envelope schema is explicit and cross-process-stable.
//!
//! Transport is deliberately the only thing here; message *types* are the
//! runtime's concern, so [`ZmqCommunicator`] is generic over any
//! `Serialize + DeserializeOwned` message.

mod communicator;
pub mod shm;

pub use communicator::{ZmqCommunicator, CommError};
pub use shm::{ShmArena, ShmError, ALIGN};
