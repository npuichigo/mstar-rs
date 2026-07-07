//! Bridge from the axum frontend to the Python conductor over the
//! `mstar-comm` `ZmqCommunicator`.
//!
//! The frontend tokenizes the prompt (Rust), submits the token-ids to the
//! conductor, and receives a stream of generated token-ids back — which it
//! detokenizes incrementally and streams as SSE (also Rust). No Python
//! touches the request/token hot path; the conductor only runs the model
//! policy per forward pass.
//!
//! Wire format is msgpack (the conductor is Python and speaks msgpack), sent
//! as the `ZmqCommunicator`'s opaque byte payload. Messages are dynamic
//! `serde_json::Value` maps, so the schema is shared by convention:
//!   frontend -> conductor: {"t":"submit","rid":u64,"tokens":[u32],"max_tokens":u32}
//!   conductor -> frontend: {"t":"token","rid":u64,"id":u32} | {"t":"done","rid":u64}

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mstar_comm::ZmqCommunicator;
use serde_json::json;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// One generated token, or end-of-stream, delivered to a request's SSE task.
#[derive(Debug)]
pub enum StreamItem {
    Token(u32),
    Done,
}

pub struct Bridge {
    mbox: Arc<ZmqCommunicator<Vec<u8>>>,
    next_rid: AtomicU64,
    // rid -> the SSE task's sender; the demux thread routes tokens here.
    routes: Arc<Mutex<HashMap<u64, UnboundedSender<StreamItem>>>>,
}

impl Bridge {
    /// Bind the frontend inbox and start the demux thread that fans inbound
    /// conductor messages out to per-request channels.
    pub fn new(socket_dir: &str) -> Result<Self, String> {
        let mbox = Arc::new(
            ZmqCommunicator::<Vec<u8>>::bind("frontend", socket_dir).map_err(|e| e.to_string())?,
        );
        let routes: Arc<Mutex<HashMap<u64, UnboundedSender<StreamItem>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let mbox_r = mbox.clone();
        let routes_r = routes.clone();
        std::thread::spawn(move || Self::demux_loop(mbox_r, routes_r));

        Ok(Self {
            mbox,
            next_rid: AtomicU64::new(1),
            routes,
        })
    }

    fn demux_loop(mbox: Arc<ZmqCommunicator<Vec<u8>>>, routes: Arc<Mutex<HashMap<u64, UnboundedSender<StreamItem>>>>) {
        loop {
            let Some(payload) = mbox.recv_timeout(Duration::from_millis(500)) else {
                continue;
            };
            let Ok(msg) = rmp_serde::from_slice::<serde_json::Value>(&payload) else {
                continue;
            };
            let rid = msg.get("rid").and_then(|v| v.as_u64()).unwrap_or(0);
            let item = match msg.get("t").and_then(|v| v.as_str()) {
                Some("token") => msg
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .map(|id| StreamItem::Token(id as u32)),
                Some("done") => Some(StreamItem::Done),
                _ => None,
            };
            if let Some(item) = item {
                let done = matches!(item, StreamItem::Done);
                let mut routes = routes.lock().expect("routes lock");
                if let Some(tx) = routes.get(&rid) {
                    let _ = tx.send(item);
                }
                if done {
                    routes.remove(&rid);
                }
            }
        }
    }

    /// Submit token-ids for generation; returns the request id + a receiver
    /// that yields generated tokens until `Done`.
    pub fn submit(&self, tokens: Vec<u32>, max_tokens: u32) -> (u64, UnboundedReceiver<StreamItem>) {
        let rid = self.next_rid.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = unbounded_channel();
        self.routes.lock().expect("routes lock").insert(rid, tx);
        let msg = json!({"t": "submit", "rid": rid, "tokens": tokens, "max_tokens": max_tokens});
        let payload = rmp_serde::to_vec_named(&msg).expect("encode submit");
        let _ = self.mbox.send("conductor", &payload);
        (rid, rx)
    }
}
