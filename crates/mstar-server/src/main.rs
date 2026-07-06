//! mstar-server: the Rust (axum) HTTP frontend — the "scale path".
//!
//! Moves the work that saturates a Python API server off the GIL: HTTP,
//! **tokenization** (HF `tokenizers` crate — same `tokenizer.json`, no
//! Python), and **per-token detokenization + SSE streaming**. This is
//! exactly the layer vLLM and SGLang moved to Rust. The conductor stays
//! Python (model policy, per-forward-pass, off the hot path); this frontend
//! talks to it over the `mstar-comm` `Mailbox` (submit token-ids, receive a
//! token-id stream) — see `bridge`.
//!
//! Two token sources, selected at startup by whether a conductor socket dir
//! is passed:
//!   - **real** (socket dir given): submit token-ids to the Python conductor
//!     over the Mailbox, stream the generated token-ids back (`sse_from_stream`).
//!   - **mock** (no socket dir): a self-contained generator that echoes the
//!     prompt tokens (`mockgen`) — proves the HTTP/tokenize/detok/SSE layer
//!     standalone, no conductor needed.
//! Either way, tokenization, incremental detokenization, and SSE
//! serialization all happen in Rust, off the GIL.

mod bridge;
mod tokenizer;

use std::sync::Arc;

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

use bridge::{Bridge, StreamItem};
use tokenizer::{IncrementalDecoder, Tok};

#[derive(Clone)]
struct AppState {
    tok: Arc<Tok>,
    /// Present when a conductor socket dir is given: real streaming via the
    /// Mailbox bridge. Absent: the self-contained mock generator.
    bridge: Option<Arc<Bridge>>,
}

#[tokio::main]
async fn main() {
    let tokenizer_path = std::env::args()
        .nth(1)
        .expect("usage: mstar-server <tokenizer.json> [port] [conductor_socket_dir]");
    let port: u16 = std::env::args()
        .nth(2)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let socket_dir = std::env::args().nth(3);

    eprintln!("loading tokenizer: {tokenizer_path}");
    let tok = Arc::new(Tok::from_file(&tokenizer_path).expect("load tokenizer"));
    eprintln!("tokenizer loaded ({} vocab)", tok.vocab_size());
    let bridge = socket_dir.map(|dir| {
        eprintln!("bridging to conductor at {dir}");
        Arc::new(Bridge::new(&dir).expect("bind bridge"))
    });
    let state = AppState { tok, bridge };

    let app = Router::new()
        .route("/health", get(health))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("mstar-server (Rust frontend) on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

// ---- tokenize / detokenize (prove Rust tokenization off the GIL) ----

#[derive(Deserialize)]
struct TokenizeReq {
    text: String,
}

#[derive(Serialize)]
struct TokenizeResp {
    tokens: Vec<u32>,
    count: usize,
}

async fn tokenize(
    State(st): State<AppState>,
    Json(req): Json<TokenizeReq>,
) -> Json<TokenizeResp> {
    let tokens = st.tok.encode(&req.text);
    Json(TokenizeResp {
        count: tokens.len(),
        tokens,
    })
}

#[derive(Deserialize)]
struct DetokenizeReq {
    tokens: Vec<u32>,
}

async fn detokenize(
    State(st): State<AppState>,
    Json(req): Json<DetokenizeReq>,
) -> Json<serde_json::Value> {
    Json(json!({"text": st.tok.decode(&req.tokens)}))
}

// ---- streaming chat completions (per-token SSE, incremental detok) ----

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[allow(dead_code)] // accepted for OpenAI compatibility; we always stream
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize {
    64
}

#[derive(Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    content: String,
}

/// Stream generated tokens as OpenAI-style SSE `chat.completion.chunk`s. The
/// token source here is a mock generator (echoes the prompt's tokens); the
/// real path replaces it with a stream of token-ids arriving from the
/// conductor over the Mailbox. Either way, tokenization, incremental
/// detokenization, and SSE serialization all happen in Rust.
async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatReq>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let prompt: String = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let prompt_ids = st.tok.encode(&prompt);

    let stream: futures::future::Either<_, _> = match &st.bridge {
        // Real path: submit token-ids to the conductor, stream the generated
        // token-ids back, detokenize incrementally in Rust.
        Some(b) => {
            let (_rid, rx) = b.submit(prompt_ids, req.max_tokens as u32);
            futures::future::Either::Left(sse_from_stream(st.tok.clone(), rx))
        }
        // Mock path: echo the prompt tokens (self-contained, no conductor).
        None => {
            let mut ids = prompt_ids;
            ids.truncate(req.max_tokens);
            futures::future::Either::Right(mockgen::sse_per_token(st.tok.clone(), ids))
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// SSE from a live token-id stream (the bridge path): each token is
/// detokenized incrementally (same `IncrementalDecoder` as the mock path) and
/// emitted as a delta chunk; `Done` closes with a finish chunk + `[DONE]`.
fn sse_from_stream(
    tok: Arc<Tok>,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamItem>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    enum Phase {
        Stream,
        Finish,
        End,
    }
    let init = (rx, IncrementalDecoder::new(tok), Phase::Stream);
    futures::stream::unfold(init, |(mut rx, mut detok, phase)| async move {
        match phase {
            Phase::Stream => match rx.recv().await {
                Some(StreamItem::Token(id)) => {
                    let piece = detok.step(id).unwrap_or_default();
                    let chunk = json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": piece}, "index": 0, "finish_reason": null}],
                    });
                    let ev = Ok(Event::default().data(chunk.to_string()));
                    Some((ev, (rx, detok, Phase::Stream)))
                }
                _ => {
                    let done = json!({
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                    });
                    let ev = Ok(Event::default().data(done.to_string()));
                    Some((ev, (rx, detok, Phase::Finish)))
                }
            },
            Phase::Finish => {
                let ev = Ok(Event::default().data("[DONE]"));
                Some((ev, (rx, detok, Phase::End)))
            }
            Phase::End => None,
        }
    })
}

/// Mock token source: build the per-token SSE events eagerly and replay them
/// as a stream. Used when no conductor is attached — the real path
/// (`sse_from_stream`) produces the same SSE from token-ids arriving over the
/// Mailbox; only the token *source* differs (tokenization, incremental
/// detokenization, and SSE serialization stay identical, all Rust off the GIL).
mod mockgen {
    use super::*;
    use futures::stream;

    pub fn sse_per_token(
        tok: Arc<Tok>,
        ids: Vec<u32>,
    ) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
        let mut detok = IncrementalDecoder::new(tok);
        let mut events: Vec<Result<Event, std::convert::Infallible>> = Vec::new();
        for id in ids {
            if let Some(piece) = detok.step(id) {
                let chunk = json!({
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": piece}, "index": 0, "finish_reason": null}],
                });
                events.push(Ok(Event::default().data(chunk.to_string())));
            }
        }
        let done = json!({
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        });
        events.push(Ok(Event::default().data(done.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));
        stream::iter(events)
    }
}
