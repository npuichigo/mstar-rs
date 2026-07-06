//! Per-entity mailbox: one inbound listener + lazily-cached outbound streams.
//!
//! Mirrors mstar's `ZMQCommunicator`: bind one inbox, PUSH to peers by id.
//! Sends are fire-and-forget and reconnect transparently if a peer restarted;
//! receives drain a channel fed by background reader threads.

use std::collections::HashMap;
use std::io::BufWriter;
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

use crate::frame::{read_frame, write_frame, FrameError};

#[derive(Debug, Error)]
pub enum MailboxError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("frame: {0}")]
    Frame(#[from] FrameError),
    #[error("peer '{0}' unreachable")]
    Unreachable(String),
}

fn sock_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.sock"))
}

/// A named message endpoint. `M` is the entity's message type.
pub struct Mailbox<M> {
    my_id: String,
    dir: PathBuf,
    // Behind a Mutex so `Mailbox` is `Sync` (a bare `Receiver` is `!Sync`) —
    // lets it be held by a PyO3 pyclass shared across Python threads.
    rx: Mutex<Receiver<M>>,
    /// peer id -> (cached writer, socket-file inode). A restarted peer
    /// re-creates its socket with a new inode, which invalidates the cache.
    peers: Mutex<HashMap<String, (BufWriter<UnixStream>, u64)>>,
    shutdown: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
}

impl<M> Mailbox<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Bind this entity's inbox at `<dir>/<my_id>.sock` and start accepting.
    pub fn bind(my_id: impl Into<String>, dir: impl Into<PathBuf>) -> Result<Self, MailboxError> {
        let my_id = my_id.into();
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let path = sock_path(&dir, &my_id);
        let _ = std::fs::remove_file(&path); // clear a stale socket
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;

        let (tx, rx) = channel::<M>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let accept_thread = Some(Self::spawn_accept_loop(listener, tx, shutdown.clone()));

        Ok(Self {
            my_id,
            dir,
            rx: Mutex::new(rx),
            peers: Mutex::new(HashMap::new()),
            shutdown,
            accept_thread,
        })
    }

    pub fn id(&self) -> &str {
        &self.my_id
    }

    /// Accept connections (non-blocking poll so drop can stop us); each
    /// connection gets a reader thread that frames messages into `tx`.
    fn spawn_accept_loop(
        listener: UnixListener,
        tx: Sender<M>,
        shutdown: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                match stream {
                    Ok(stream) => {
                        let tx = tx.clone();
                        let shutdown = shutdown.clone();
                        std::thread::spawn(move || Self::reader_loop(stream, tx, shutdown));
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        })
    }

    fn reader_loop(mut stream: UnixStream, tx: Sender<M>, shutdown: Arc<AtomicBool>) {
        stream.set_nonblocking(false).ok();
        loop {
            match read_frame::<M, _>(&mut stream) {
                Ok(Some(msg)) => {
                    if tx.send(msg).is_err() {
                        break; // receiver dropped (mailbox gone)
                    }
                }
                Ok(None) => break, // peer closed cleanly
                Err(_) => break,   // transport error: drop this connection
            }
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
        }
    }

    /// Send `msg` to peer `peer_id` (fire-and-forget). Transparently
    /// reconnects if the peer restarted (socket inode changed) or the cached
    /// stream broke.
    pub fn send(&self, peer_id: &str, msg: &M) -> Result<(), MailboxError> {
        let ino = self.peer_inode(peer_id)?;
        let mut peers = self.peers.lock().expect("peers lock");
        // Drop a stale cache entry when the peer's socket was re-created.
        if peers.get(peer_id).map(|(_, i)| *i) != Some(ino) {
            peers.remove(peer_id);
        }
        if !peers.contains_key(peer_id) {
            let w = BufWriter::new(self.connect(peer_id)?);
            peers.insert(peer_id.to_string(), (w, ino));
        }
        // Write on the cached stream; reconnect once on a transport error
        // (broken pipe on an inode that hasn't changed yet).
        if write_frame(&mut peers.get_mut(peer_id).unwrap().0, msg).is_ok() {
            return Ok(());
        }
        let mut w = BufWriter::new(self.connect(peer_id)?);
        write_frame(&mut w, msg)?;
        peers.insert(peer_id.to_string(), (w, ino));
        Ok(())
    }

    fn peer_inode(&self, peer_id: &str) -> Result<u64, MailboxError> {
        std::fs::metadata(sock_path(&self.dir, peer_id))
            .map(|m| m.ino())
            .map_err(|_| MailboxError::Unreachable(peer_id.to_string()))
    }

    fn connect(&self, peer_id: &str) -> Result<UnixStream, MailboxError> {
        UnixStream::connect(sock_path(&self.dir, peer_id))
            .map_err(|_| MailboxError::Unreachable(peer_id.to_string()))
    }

    /// Non-blocking: next inbound message, or None.
    pub fn try_recv(&self) -> Option<M> {
        self.rx.lock().expect("rx lock").try_recv().ok()
    }

    /// Block until the next inbound message.
    pub fn recv(&self) -> Option<M> {
        self.rx.lock().expect("rx lock").recv().ok()
    }

    /// Block up to `timeout` for the next inbound message.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<M> {
        self.rx.lock().expect("rx lock").recv_timeout(timeout).ok()
    }

    /// Drain all currently-queued inbound messages.
    pub fn drain(&self) -> Vec<M> {
        let rx = self.rx.lock().expect("rx lock");
        let mut out = Vec::new();
        while let Ok(m) = rx.try_recv() {
            out.push(m);
        }
        out
    }
}

impl<M> Drop for Mailbox<M> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Unblock the accept poll by connecting to ourselves once.
        let _ = UnixStream::connect(sock_path(&self.dir, &self.my_id));
        if let Some(h) = self.accept_thread.take() {
            let _ = h.join();
        }
        let _ = std::fs::remove_file(sock_path(&self.dir, &self.my_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::time::Duration;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum Msg {
        Hello(String),
        Batch { id: u64, node: String },
    }

    fn tmpdir(tag: &str) -> PathBuf {
        // Unique per test via the tag + process/thread id (no Date/rand).
        let t = format!("{:?}", std::thread::current().id());
        let dir = std::env::temp_dir().join(format!("mstar_comm_{tag}_{t}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn wait_for<M, F: Fn(&Mailbox<M>) -> Option<Msg>>(mb: &Mailbox<M>, f: F) -> Msg
    where
        M: Serialize + DeserializeOwned + Send + 'static,
    {
        for _ in 0..500 {
            if let Some(m) = f(mb) {
                return m;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for message");
    }

    #[test]
    fn two_entities_exchange_messages() {
        let dir = tmpdir("exch");
        let conductor: Mailbox<Msg> = Mailbox::bind("conductor", &dir).unwrap();
        let worker: Mailbox<Msg> = Mailbox::bind("worker_0", &dir).unwrap();

        conductor
            .send("worker_0", &Msg::Batch {
                id: 1,
                node: "LLM".into(),
            })
            .unwrap();
        let got = wait_for(&worker, |w| w.try_recv());
        assert_eq!(got, Msg::Batch { id: 1, node: "LLM".into() });

        // Reply the other direction.
        worker.send("conductor", &Msg::Hello("done".into())).unwrap();
        let got = wait_for(&conductor, |c| c.try_recv());
        assert_eq!(got, Msg::Hello("done".into()));
    }

    #[test]
    fn many_messages_arrive_in_order_from_one_sender() {
        let dir = tmpdir("order");
        let a: Mailbox<Msg> = Mailbox::bind("a", &dir).unwrap();
        let b: Mailbox<Msg> = Mailbox::bind("b", &dir).unwrap();
        for i in 0..100 {
            a.send("b", &Msg::Batch { id: i, node: "n".into() }).unwrap();
        }
        // Drain 100 in FIFO order (single sender = single connection = ordered).
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
    fn send_to_unbound_peer_errors() {
        let dir = tmpdir("unbound");
        let a: Mailbox<Msg> = Mailbox::bind("a", &dir).unwrap();
        assert!(matches!(
            a.send("ghost", &Msg::Hello("x".into())),
            Err(MailboxError::Unreachable(_))
        ));
    }

    #[test]
    fn reconnects_after_peer_restart() {
        let dir = tmpdir("restart");
        let a: Mailbox<Msg> = Mailbox::bind("a", &dir).unwrap();
        {
            let b: Mailbox<Msg> = Mailbox::bind("b", &dir).unwrap();
            a.send("b", &Msg::Hello("1".into())).unwrap();
            wait_for(&b, |b| b.try_recv());
        } // b drops (socket removed)
        std::thread::sleep(Duration::from_millis(10));
        // New b at the same id; a's cached stream is now stale -> reconnect.
        let b2: Mailbox<Msg> = Mailbox::bind("b", &dir).unwrap();
        // First send may land on the dead stream; send twice so the retry path
        // re-binds to b2 (mirrors a real restart where a keeps pushing).
        let _ = a.send("b", &Msg::Hello("2".into()));
        a.send("b", &Msg::Hello("3".into())).unwrap();
        wait_for(&b2, |b| b.try_recv());
    }
}
