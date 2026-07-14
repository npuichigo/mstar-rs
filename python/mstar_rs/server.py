"""The serving drive-loop: `ServingEngine`.

`ServingEngine` owns a **background drive thread** that holds the `Conductor`
(and thus the Rust `Runtime` — all runtime access stays on one thread, as PyO3
requires) and loops `conductor.poll()`, continuously-batching across every
in-flight request. Callers submit and block for the result. It's the reusable
request-driver, independent of any HTTP layer: `verify_serve_local.py` drives an
in-process `Driver` through it with no server at all.

The production **HTTP surface is the Rust axum frontend** (`crates/mstar-server`),
started together with the workers by `mstar_rs.launch`. The frontend runs HTTP +
adapters + media + SSE off the GIL in its own process and talks to the Python
conductor over the `mstar-comm` ZmqCommunicator (submit text/media, stream
`ResultChunk`s back) — the same split vLLM (`rust/vllm-server`) and SGLang
(`sgl-router`) use. This module carries no HTTP server of its own; the earlier
in-process FastAPI dev-path was removed once the axum frontend + launcher
covered the whole surface.
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
