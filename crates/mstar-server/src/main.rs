//! mstar-server: the Rust (axum) HTTP frontend — the "scale path".
//!
//! Moves the work that saturates a Python API server off the GIL: HTTP,
//! **tokenization** (HF `tokenizers` crate — same `tokenizer.json`, no
//! Python), and **per-token detokenization + SSE streaming**. This is
//! exactly the layer vLLM and SGLang moved to Rust. The conductor stays
//! Python (model policy, per-forward-pass, off the hot path); this frontend
//! talks to it over the `mstar-comm` `Mailbox` (submit token-ids, receive a
//! token-id stream — wired in the next increment).
//!
//! Increment 1 (this file): a self-contained, runnable server proving the
//! architecture — `/tokenize`, `/detokenize`, and a streaming
//! `/v1/chat/completions` whose token source is a local mock generator, with
//! incremental detokenization + OpenAI-style SSE, entirely off the GIL.

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

use tokenizer::Tok;

#[derive(Clone)]
struct AppState {
    tok: Arc<Tok>,
}

#[tokio::main]
async fn main() {
    let tokenizer_path = std::env::args()
        .nth(1)
        .expect("usage: mstar-server <tokenizer.json> [port]");
    let port: u16 = std::env::args()
        .nth(2)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    eprintln!("loading tokenizer: {tokenizer_path}");
    let tok = Arc::new(Tok::from_file(&tokenizer_path).expect("load tokenizer"));
    eprintln!("tokenizer loaded ({} vocab)", tok.vocab_size());
    let state = AppState { tok };

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

    // Mock "generation": echo the prompt's own token stream back, capped.
    let mut gen_ids = st.tok.encode(&prompt);
    gen_ids.truncate(req.max_tokens);

    let tok = st.tok.clone();
    let stream = mockgen::sse_per_token(tok, gen_ids);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Mock token source: build the per-token SSE events eagerly and replay them
/// as a stream. Increment 2 replaces this with events produced as token-ids
/// arrive from the conductor over the Mailbox — the tokenization,
/// incremental detokenization, and SSE serialization stay identical (all
/// Rust, off the GIL); only the token *source* changes.
mod mockgen {
    use super::*;
    use futures::stream;

    pub fn sse_per_token(
        tok: Arc<Tok>,
        ids: Vec<u32>,
    ) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
        let mut detok = tok.decode_stream();
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
