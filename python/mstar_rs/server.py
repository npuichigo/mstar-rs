"""HTTP serving surface for the multi-process runtime — the **dev path** (T5).

A FastAPI app co-located with the conductor. HTTP requests are enqueued;
a single **background drive thread** owns the `Conductor` (and thus the Rust
`Runtime` — all runtime access stays on one thread, as PyO3 requires) and
loops `conductor.poll()`, which continuously-batches across every in-flight
request and dispatches to the worker processes. When a request completes,
its future is resolved and the HTTP handler returns.

This is one of two front-ends; they are complementary, not redundant:

  * **T5 (this file)** — FastAPI co-located with the conductor. Simplest to
    run (one process, no extra hop); right for dev / low concurrency. HTTP,
    tokenization, detokenization and JSON/SSE all run in Python under the GIL.
  * **T6 (`crates/mstar-server`, axum)** — the *scale* path, for **high
    concurrency**. Under many concurrent requests a Python FastAPI/uvicorn
    front-end saturates on the GIL — per-request tokenize + per-token
    detokenize + JSON/SSE serialize all contend for one interpreter lock —
    which is exactly why **both vLLM and SGLang moved their front-ends to
    Rust**. T6 does the same: HTTP + tokenize + detokenize + SSE in Rust
    (off the GIL), talking to the same Python conductor over the ZmqCommunicator. It
    does NOT replace the conductor (which must stay Python — it runs the model
    policy); the conductor is per-forward-pass and GPU-bound, not the
    concurrency bottleneck. The extra process/hop buys GIL-free request
    handling that scales with concurrent connections.

Relation to mstar: mstar's front-end is 100% Python (FastAPI/uvicorn) and,
notably, runs the api_server as a *separate process* from the conductor
(ZMQ IPC) — so T5's co-location is a deliberate simplification, and T6's Rust
front-end for GIL offload is a departure mstar has no precedent for.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from .dist import Conductor


class ServingEngine:
    """Owns the request-driver on a background thread; HTTP handlers submit here.

    Drives either a multi-process `dist.Conductor` (batches dispatched to worker
    processes over ZeroMQ every step — for genuinely-distributed deployments) or
    an in-process `driver.Driver` (scheduler + engine co-located, no per-step
    IPC — the co-located fast path). Both expose the same submit/poll/finished/
    errors/shutdown_workers surface, so the HTTP layer is identical for both."""

    def __init__(self, conductor) -> None:  # Conductor | Driver
        self.cond = conductor
        self._submit_q: queue.Queue = queue.Queue()
        self._futures: dict[int, threading.Event] = {}
        self._results: dict[int, Any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drive, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        # Only touch the mailbox from here once the drive thread has actually
        # exited; otherwise shutdown_workers() would race its poll() sends.
        if not self._thread.is_alive():
            self.cond.shutdown_workers()

    def submit(self, request: dict[str, Any], timeout: float = 120.0) -> Any:
        """Blocking submit: enqueue, wait for completion, return the result."""
        done = threading.Event()
        self._submit_q.put((request, done))
        if not done.wait(timeout):
            # Don't leak bookkeeping for a request that never resolved.
            rid = getattr(done, "_rid", None)
            if rid is not None:
                self._futures.pop(rid, None)
                self._results.pop(rid, None)
            raise TimeoutError("request timed out")
        # rid is stashed on the event by the drive thread.
        rid = done._rid  # type: ignore[attr-defined]
        if (err := self.cond.errors.pop(rid, None)) is not None:
            self._results.pop(rid, None)
            raise RuntimeError(f"request failed in worker: {err}")
        return self._results.pop(rid)

    def _drive(self) -> None:
        while not self._stop.is_set():
            # Ingest any newly-submitted requests (runtime touched here only).
            while True:
                try:
                    request, done = self._submit_q.get_nowait()
                except queue.Empty:
                    break
                rid = self.cond.submit(request)
                self._futures[rid] = done
                done._rid = rid  # type: ignore[attr-defined]
            # Drive one continuous-batching step across all active requests.
            self.cond.poll(timeout_ms=5)
            # Resolve any requests that just finished.
            for rid in list(self.cond.finished):
                self.cond.finished.discard(rid)
                self._results[rid] = self.cond.results.pop(rid, [])
                if (ev := self._futures.pop(rid, None)) is not None:
                    ev.set()


def build_app(engine: ServingEngine):
    """A minimal FastAPI app over the serving engine (call `engine.start()`
    before serving). The default handler passes the JSON body straight to
    the model and returns the postprocessed result; swap the marshalling for
    a model's real request/response schema."""
    from fastapi import FastAPI

    app = FastAPI(title="mstar-rs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "inflight": engine.cond._inflight}

    @app.post("/generate")
    async def generate(body: dict) -> dict:
        # Run the blocking submit in a threadpool so the event loop stays free.
        import anyio

        result = await anyio.to_thread.run_sync(engine.submit, body)
        return {"result": [_jsonable(x) for x in result]}

    return app


def _jsonable(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.tolist()
    except ImportError:
        pass
    return x
