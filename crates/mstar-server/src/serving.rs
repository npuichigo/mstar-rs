//! Shared serving state + the result-chunk source.
//!
//! `result_stream` is the single source both output surfaces draw from (as in
//! mstar's `iter_result_chunks`): the `/v1/*` handlers translate the chunks
//! into SSE / WAV / JSON, `/generate` into NDJSON. Two sources, selected at
//! startup by whether a conductor socket dir was given:
//!  * **real** — submit the flattened request to the Python conductor over the
//!    bridge, stream `ResultChunk`s back;
//!  * **mock** — a self-contained generator (echoes text, emits a short silent
//!    PCM for audio, a 1x1 PNG for image) that proves the HTTP/adapter/media/
//!    streaming layer standalone with no conductor.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::stream::Stream;
use uuid::Uuid;

use crate::adapters::SubmitArgs;
use crate::bridge::{Bridge, ResultChunk, StreamItem};

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub upload_dir: PathBuf,
    /// Fetching arbitrary http(s) media is SSRF surface; off by default.
    pub allow_remote: bool,
    /// Output audio sample rate (mstar reads it from the model; the frontend
    /// has no model, so it's configured, default 24 kHz).
    pub sample_rate: u32,
    /// Present when a conductor socket dir is given (real streaming); absent =
    /// the self-contained mock generator.
    pub bridge: Option<Arc<Bridge>>,
}

pub type ChunkStream = Pin<Box<dyn Stream<Item = ResultChunk> + Send>>;

/// The unified chunk source: bridge to the conductor, or the mock generator.
pub fn result_stream(
    state: &AppState,
    args: &SubmitArgs,
    request_id: &str,
    streaming: bool,
) -> ChunkStream {
    match &state.bridge {
        Some(b) => {
            let rx = b.submit(args, request_id, streaming);
            Box::pin(futures::stream::unfold(rx, |mut rx| async move {
                match rx.recv().await {
                    Some(StreamItem::Chunk(c)) => Some((c, rx)),
                    _ => None, // Done, or the sender was dropped
                }
            }))
        }
        None => Box::pin(futures::stream::iter(mock_chunks(args, state.sample_rate))),
    }
}

/// Mock output: one chunk per requested output modality.
fn mock_chunks(args: &SubmitArgs, sample_rate: u32) -> Vec<ResultChunk> {
    let mut out = Vec::new();
    let prompt = args.text.clone().unwrap_or_default();
    for modality in &args.output_modalities {
        match modality.as_str() {
            "text" => out.push(ResultChunk {
                modality: "text".to_string(),
                data: format!("[mock] {prompt}").into_bytes(),
                metadata: serde_json::Value::Null,
            }),
            "audio" => out.push(ResultChunk {
                modality: "audio".to_string(),
                // 0.1 s of silence as 16-bit PCM at the model rate.
                data: vec![0u8; (sample_rate as usize / 10) * 2],
                metadata: serde_json::Value::Null,
            }),
            "image" => out.push(ResultChunk {
                modality: "image".to_string(),
                data: ONE_PX_PNG.to_vec(),
                metadata: serde_json::Value::Null,
            }),
            _ => {}
        }
    }
    out
}

/// A 1x1 opaque PNG, for the mock image path.
const ONE_PX_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
    0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

/// Unix timestamp (seconds), for OpenAI `created` fields.
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `<prefix>-<uuid4-hex>`, matching mstar's request-id shape.
pub fn rid(prefix: &str) -> String {
    format!("{}-{}", prefix, Uuid::new_v4().simple())
}
