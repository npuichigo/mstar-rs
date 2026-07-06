//! mstar-comm: the control-plane message mesh for multi-process mstar-rs.
//!
//! Ports the semantics of mstar's ZMQ PUSH/PULL layer
//! (`communication/communicator.py`) without the libzmq dependency: each
//! entity binds one **inbox** (a Unix-domain listener at
//! `<dir>/<entity_id>.sock`) and opens lazily-cached **outbound** streams to
//! its peers. Sends are fire-and-forget (like PUSH); receives drain the
//! inbox (like PULL). Messages are length-prefixed bincode frames — a real
//! wire format replacing mstar's pickle, so the envelope schema is explicit
//! and cross-process-stable.
//!
//! Transport is deliberately the only thing here; message *types* are the
//! runtime's concern, so [`Mailbox`] is generic over any
//! `Serialize + DeserializeOwned` message.

mod frame;
mod mailbox;

pub use frame::{read_frame, write_frame, FrameError};
pub use mailbox::{Mailbox, MailboxError};
