//! Per-entity mailbox over ZeroMQ PUSH/PULL — the direct analogue of mstar's
//! `ZMQCommunicator`.
//!
//! Each entity binds one **PULL** inbox at `ipc://<dir>/<my_id>.ipc` and
//! connects a lazily-cached **PUSH** socket per peer. PUSH/PULL gives
//! fire-and-forget, ordered, load-balanced delivery; libzmq queues to a
//! not-yet-bound peer and transparently reconnects when a peer restarts (so,
//! unlike the earlier hand-rolled UDS transport, there is no "unreachable"
//! error and no inode-reconnect bookkeeping to maintain). Messages are
//! bincode-serialized, one message per zmq frame.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CommError {
    #[error("zmq: {0}")]
    Zmq(#[from] zmq::Error),
    #[error("serialize: {0}")]
    Serialize(bincode::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// The ipc endpoint an entity binds/connects to.
fn endpoint(dir: &Path, id: &str) -> String {
    format!("ipc://{}/{}.ipc", dir.display(), id)
}

/// The backing socket file (for clearing a stale one before bind).
fn sock_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.ipc"))
}

/// A named message endpoint. `M` is the entity's message type.
///
/// Field order matters for `Drop`: the sockets must close before the
/// `Context` is dropped (`zmq_ctx_term` blocks until its sockets are gone).
pub struct ZmqCommunicator<M> {
    my_id: String,
    dir: PathBuf,
    // PULL inbox. Behind a Mutex because a zmq `Socket` is `!Sync` (and must be
    // used from one thread at a time); the drive loop is the single consumer.
    pull: Mutex<zmq::Socket>,
    // peer id -> connected PUSH socket (created on first send to that peer).
    peers: Mutex<HashMap<String, zmq::Socket>>,
    ctx: zmq::Context,
    _marker: PhantomData<fn() -> M>,
}

impl<M> ZmqCommunicator<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Bind this entity's PULL inbox at `ipc://<dir>/<my_id>.ipc`.
    pub fn bind(my_id: impl Into<String>, dir: impl Into<PathBuf>) -> Result<Self, CommError> {
        let my_id = my_id.into();
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let _ = std::fs::remove_file(sock_path(&dir, &my_id)); // clear a stale socket
        let ctx = zmq::Context::new();
        let pull = ctx.socket(zmq::PULL)?;
        pull.set_linger(0)?; // don't block on close
        pull.bind(&endpoint(&dir, &my_id))?;
        Ok(Self {
            my_id,
            dir,
            pull: Mutex::new(pull),
            peers: Mutex::new(HashMap::new()),
            ctx,
            _marker: PhantomData,
        })
    }

    pub fn id(&self) -> &str {
        &self.my_id
    }

    /// Send `msg` to peer `peer_id` (fire-and-forget). Queues if the peer
    /// isn't bound yet and reconnects transparently if it restarted — libzmq
    /// owns both, so this never reports "unreachable".
    pub fn send(&self, peer_id: &str, msg: &M) -> Result<(), CommError> {
        let bytes = bincode::serialize(msg).map_err(CommError::Serialize)?;
        let mut peers = self.peers.lock().expect("peers lock");
        if !peers.contains_key(peer_id) {
            let push = self.ctx.socket(zmq::PUSH)?;
            push.set_linger(0)?;
            push.connect(&endpoint(&self.dir, peer_id))?;
            peers.insert(peer_id.to_string(), push);
        }
        peers.get(peer_id).expect("just inserted").send(&bytes, 0)?;
        Ok(())
    }

    /// Non-blocking: next inbound message, or None.
    pub fn try_recv(&self) -> Option<M> {
        let pull = self.pull.lock().expect("pull lock");
        match pull.recv_bytes(zmq::DONTWAIT) {
            Ok(b) => bincode::deserialize(&b).ok(),
            Err(_) => None, // EAGAIN when the inbox is empty
        }
    }

    /// Block until the next inbound message.
    pub fn recv(&self) -> Option<M> {
        let pull = self.pull.lock().expect("pull lock");
        pull.recv_bytes(0)
            .ok()
            .and_then(|b| bincode::deserialize(&b).ok())
    }

    /// Block up to `timeout` for the next inbound message.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<M> {
        let pull = self.pull.lock().expect("pull lock");
        let ms = timeout.as_millis().min(i64::MAX as u128) as i64;
        let readable = {
            let mut items = [pull.as_poll_item(zmq::POLLIN)];
            zmq::poll(&mut items, ms).unwrap_or(0) > 0 && items[0].is_readable()
        };
        if readable {
            pull.recv_bytes(zmq::DONTWAIT)
                .ok()
                .and_then(|b| bincode::deserialize(&b).ok())
        } else {
            None
        }
    }

    /// Drain all currently-queued inbound messages.
    pub fn drain(&self) -> Vec<M> {
        let pull = self.pull.lock().expect("pull lock");
        let mut out = Vec::new();
        while let Ok(b) = pull.recv_bytes(zmq::DONTWAIT) {
            if let Ok(m) = bincode::deserialize(&b) {
                out.push(m);
            }
        }
        out
    }
}

impl<M> Drop for ZmqCommunicator<M> {
    fn drop(&mut self) {
        // Sockets (pull + peers) close first via field-drop order; zmq unlinks
        // the bound ipc file on close, but remove it defensively too.
        let _ = std::fs::remove_file(sock_path(&self.dir, &self.my_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum Msg {
        Hello(String),
        Batch { id: u64, node: String },
    }

    fn tmpdir(tag: &str) -> PathBuf {
        // Unique per test via the tag + thread id (no Date/rand available).
        let t = format!("{:?}", std::thread::current().id());
        let dir = std::env::temp_dir().join(format!("mstar_comm_{tag}_{t}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn wait_for<M, F: Fn(&ZmqCommunicator<M>) -> Option<Msg>>(mb: &ZmqCommunicator<M>, f: F) -> Msg
    where
        M: Serialize + DeserializeOwned + Send + 'static,
    {
        for _ in 0..500 {
            if let Some(m) = f(mb) {
                return m;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        panic!("timed out waiting for message");
    }

    #[test]
    fn two_entities_exchange_messages() {
        let dir = tmpdir("exch");
        let conductor: ZmqCommunicator<Msg> = ZmqCommunicator::bind("conductor", &dir).unwrap();
        let worker: ZmqCommunicator<Msg> = ZmqCommunicator::bind("worker_0", &dir).unwrap();

        conductor
            .send("worker_0", &Msg::Batch {
                id: 1,
                node: "LLM".into(),
            })
            .unwrap();
        let got = wait_for(&worker, |w| w.try_recv());
        assert_eq!(got, Msg::Batch { id: 1, node: "LLM".into() });

        worker.send("conductor", &Msg::Hello("done".into())).unwrap();
        let got = wait_for(&conductor, |c| c.try_recv());
        assert_eq!(got, Msg::Hello("done".into()));
    }

    #[test]
    fn many_messages_arrive_in_order_from_one_sender() {
        let dir = tmpdir("order");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        let b: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
        for i in 0..100 {
            a.send("b", &Msg::Batch { id: i, node: "n".into() }).unwrap();
        }
        // PUSH->PULL over one connection preserves FIFO order.
        let mut seen = 0u64;
        for _ in 0..1000 {
            for m in b.drain() {
                if let Msg::Batch { id, .. } = m {
                    assert_eq!(id, seen);
                    seen += 1;
                }
            }
            if seen == 100 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(seen, 100);
    }

    #[test]
    fn send_before_peer_binds_is_queued() {
        // Unlike the old UDS transport (which errored on a missing peer), zmq
        // PUSH queues to a not-yet-bound endpoint and delivers once it binds.
        let dir = tmpdir("queue");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        a.send("late", &Msg::Hello("queued".into())).unwrap(); // no peer yet — no error
        let late: ZmqCommunicator<Msg> = ZmqCommunicator::bind("late", &dir).unwrap();
        let got = wait_for(&late, |m| m.try_recv());
        assert_eq!(got, Msg::Hello("queued".into()));
    }

    #[test]
    fn reconnects_after_peer_restart() {
        let dir = tmpdir("restart");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        {
            let b: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
            a.send("b", &Msg::Hello("1".into())).unwrap();
            wait_for(&b, |b| b.try_recv());
        } // b drops; zmq unlinks its inbox
        std::thread::sleep(Duration::from_millis(20));
        // New b at the same id: a's cached PUSH auto-reconnects to it.
        let b2: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
        a.send("b", &Msg::Hello("2".into())).unwrap();
        let got = wait_for(&b2, |b| b.try_recv());
        assert_eq!(got, Msg::Hello("2".into()));
    }
}
