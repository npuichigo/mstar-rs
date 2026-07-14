# OpenAI-surface coverage: `mstar-server` vs mstar's `api_server`

The coverage diff for RFC #130 Step 3 (API server → Rust). Compared against
mstar's `api_server` (`openai/router.py`, `openai/protocol.py`,
`openai/serving_*.py`, `media_io.py`, `adapters.py`, `entrypoint.py`).

## Endpoints

| Endpoint | mstar | mstar-server | Notes |
|---|---|---|---|
| `GET /v1/models` | ✓ | ✓ | same ModelCard/ModelList shapes |
| `POST /v1/chat/completions` | ✓ | ✓ | SSE + JSON; text/audio/image out |
| `POST /v1/audio/speech` | ✓ | ✓ | streaming open-ended WAV, or container |
| `POST /v1/images/generations` | ✓ | ✓ | `{created, data: [{b64_json, url: null}]}` |
| `POST /v1/images/edits` | ✓ | ✓ | multipart; unknown fields JSON-parsed into model_kwargs |
| `POST /generate` (native) | ✓ | ✓ | multipart → NDJSON stream or JSON collect |
| `GET /health` | ✓ | ✓* | see *warming gate* below |
| model-specific extension routers | ✓ | — | per-model custom surfaces are out of scope here; the adapter registry is where a model opts into OpenAI |

## Request fields

All request models accept unknown fields and pass them through as
`model_kwargs` (pydantic `extra="allow"` ↔ `#[serde(flatten)]`), so the OpenAI
client's `extra_body` reaches the model on both.

Declared-but-unused fields are matched exactly so they do **not** leak into
`extra_body` passthrough: chat `n`, `stop`; speech `speed`; images `n`,
`size`, `response_format`. Handled fields (`temperature`, `top_p`,
`max_tokens`/`max_completion_tokens`, `seed`, `voice`, `modalities`,
`audio.voice`, `stream`, `response_format`) map onto the same per-model
sampling keys via the same adapter rules (`setdefault`, so an explicit
`extra_body` value wins).

Chat `messages[].content` accepts the same shapes: plain string, or content
parts `text` / `image_url` / `video_url` / `audio_url` / `input_audio`
(base64 + format). Media persists under `upload_dir`; `data:` URLs and
`http(s)` URLs both resolve (remote fetch is gated — see below).

## Behaviors

| Behavior | mstar | mstar-server |
|---|---|---|
| Error envelope | `{"error": {message, type, code}}` | same |
| Worker/backend failure | HTTP 500 (collect) / connection drop (stream) | HTTP 500 (collect) / terminal SSE-or-NDJSON error event (stream) — strictly more informative |
| Ingest failure isolation | per-request | per-request (serve loop survives; verified) |
| Request timeout | 600 s → HTTP 500 + `abort_request` | same (`MSTAR_REQUEST_TIMEOUT_S`, default 600) + abort to the backend |
| Client disconnect | aborts request, frees engine state | same (drop-guard sends abort; backend releases runtime + engine state) |
| Upload cleanup | deleted 60 s after handling | same (only files under `upload_dir`) |
| CORS | allow-all middleware | allow-all `CorsLayer` |
| Remote media fetch | on by default (SSRF noted) | implemented; **off by default** (`MSTAR_ALLOW_REMOTE=1`) — deliberate hardening, 30 s timeout |
| Streaming audio deltas (chat) | base64 PCM16 in `delta.audio.data` | same |
| Non-stream chat audio | `message.audio {id, data(b64 wav), expires_at, transcript}` | same |
| Compressed audio out (`mp3`/`flac`/…) | via optional `soundfile`, else degrade to WAV | degrade to WAV (same fallback path; encoder never shipped in the frontend) |
| `/health` warming gate | 503 until first-request warmup | not needed in the same way: the launcher starts the HTTP binary only after workers report ready. In the decoupled (`--no-frontend`) deploy the frontend binds immediately — front it with the backend's readiness if that matters |
| Per-request profiling (`--log-stats`) | server-side report | not ported (backend concern, not API surface) |

## Architecture note

The largest deliberate difference: mstar's api_server tokenizes/detokenizes in
the (Python) server; here tokenization and feature extraction live with the
model (conductor policy + engine), and the frontend ships
`{text, file_paths, modalities, model_kwargs}` and receives
`{modality, bytes, metadata}` chunks — mandatory for multimodal inputs and the
proposed Rust-default-with-model-override tokenization hook composes with it
(the wire supports token-id submits as well).

Verified by `examples/serve_bridge.py` (echo round-trip over real HTTP),
`examples/verify_serving_errors.py` (failure surfaces + loop survival), and
the mock path (`mstar-server <model> <port>` with no backend).
