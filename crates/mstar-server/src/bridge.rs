//! Bridge from the axum frontend to the Python conductor over the
//! `mstar-comm` `ZmqCommunicator`.
//!
//! Mirrors mstar's `APIServer.submit_request` / result-collection contract:
//! the frontend flattens an OpenAI request into a submit (text + media file
//! paths + in/out modalities + model_kwargs) and receives a stream of
//! multimodal `ResultChunk`s ({modality, data, metadata}) back — which the
//! serving handlers translate into SSE / WAV / NDJSON. Media preprocessing and
//! the model run in the Python data plane; no Python touches the request/HTTP
//! hot path.
//!
//! Wire format is msgpack (the conductor speaks msgpack), sent as the
//! `ZmqCommunicator`'s opaque byte payload:
//!   frontend -> conductor: SubmitMsg {t:"submit", rid, text, file_paths,
//!       input_modalities, output_modalities, model_kwargs, streaming}
//!   conductor -> frontend: {t:"chunk", rid, modality, data(bin), metadata}
//!                        | {t:"done",  rid}

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mstar_comm::RawZmqCommunicator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::adapters::SubmitArgs;

/// One chunk of generated output, matching mstar's `ResultChunk`.
#[derive(Debug, Clone)]
pub struct ResultChunk {
    pub modality: String,
    pub data: Vec<u8>,
    pub metadata: Value,
}

/// A result chunk, or end-of-stream, delivered to a request's serving task.
#[derive(Debug)]
pub enum StreamItem {
    Chunk(ResultChunk),
    Done,
}

// ---- wire messages ----

#[derive(Serialize)]
struct SubmitMsg<'a> {
    t: &'static str,
    rid: &'a str,
    text: Option<&'a str>,
    file_paths: &'a BTreeMap<String, Vec<String>>,
    input_modalities: &'a [String],
    output_modalities: &'a [String],
    model_kwargs: &'a serde_json::Map<String, Value>,
    streaming: bool,
}

/// Inbound message; `t` selects chunk vs done. `data` is msgpack binary.
#[derive(Deserialize)]
struct Inbound {
    t: String,
    rid: String,
    #[serde(default)]
    modality: String,
    #[serde(default)]
    data: serde_bytes::ByteBuf,
    #[serde(default)]
    metadata: Value,
}

pub struct Bridge {
    mbox: Arc<RawZmqCommunicator>,
    // rid -> the serving task's sender; the demux thread routes chunks here.
    routes: Arc<Mutex<HashMap<String, UnboundedSender<StreamItem>>>>,
}

impl Bridge {
    /// Bind the frontend inbox and start the demux thread that fans inbound
    /// conductor messages out to per-request channels.
    pub fn new(socket_dir: &str) -> Result<Self, String> {
        let mbox = Arc::new(
            RawZmqCommunicator::bind("frontend", socket_dir).map_err(|e| e.to_string())?,
        );
        let routes: Arc<Mutex<HashMap<String, UnboundedSender<StreamItem>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let mbox_r = mbox.clone();
        let routes_r = routes.clone();
        std::thread::spawn(move || Self::demux_loop(mbox_r, routes_r));

        Ok(Self { mbox, routes })
    }

    fn demux_loop(
        mbox: Arc<RawZmqCommunicator>,
        routes: Arc<Mutex<HashMap<String, UnboundedSender<StreamItem>>>>,
    ) {
        loop {
            let Some(payload) = mbox.recv_timeout(Duration::from_millis(500)) else {
                continue;
            };
            let Ok(msg) = rmp_serde::from_slice::<Inbound>(&payload) else {
                continue;
            };
            let (item, done) = match msg.t.as_str() {
                "chunk" => (
                    Some(StreamItem::Chunk(ResultChunk {
                        modality: msg.modality,
                        data: msg.data.into_vec(),
                        metadata: msg.metadata,
                    })),
                    false,
                ),
                "done" => (Some(StreamItem::Done), true),
                _ => (None, false),
            };
            if let Some(item) = item {
                let mut routes = routes.lock().expect("routes lock");
                if let Some(tx) = routes.get(&msg.rid) {
                    let _ = tx.send(item);
                }
                if done {
                    routes.remove(&msg.rid);
                }
            }
        }
    }

    /// Submit a flattened request under `request_id`; returns a receiver that
    /// yields `ResultChunk`s until `Done`. The `request_id` is caller-supplied
    /// (the serving handler mints `chatcmpl-…`/`speech-…`), matching mstar,
    /// whose per-request RNG seed derives from `hash(request_id)`.
    pub fn submit(
        &self,
        args: &SubmitArgs,
        request_id: &str,
        streaming: bool,
    ) -> UnboundedReceiver<StreamItem> {
        let (tx, rx) = unbounded_channel();
        self.routes
            .lock()
            .expect("routes lock")
            .insert(request_id.to_string(), tx);
        let msg = SubmitMsg {
            t: "submit",
            rid: request_id,
            text: args.text.as_deref(),
            file_paths: &args.file_paths,
            input_modalities: &args.input_modalities,
            output_modalities: &args.output_modalities,
            model_kwargs: &args.model_kwargs,
            streaming,
        };
        let payload = rmp_serde::to_vec_named(&msg).expect("encode submit");
        let _ = self.mbox.send("conductor", &payload);
        rx
    }
}
