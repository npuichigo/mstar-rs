"""Multi-process runtime: a **conductor** process drives one or more
**worker** processes over the ZeroMQ control mesh (`ZmqCommunicator`), with tensors
moving through shared memory (`ShmArena`).

Split of responsibilities (mirroring mstar's conductor/worker):

- **Conductor** — owns the Rust `Runtime` (scheduling, walk state, routing)
  and the model *policy* (`walks` / `initial_walks` / `next_forward`). It
  works with tensor *descriptors* only; it never runs the model. Per batch it
  resolves the input descriptors and sends an `execute` message to the node's
  worker; on `done` it mints output uuids, records their SHM location, and
  routes via `complete_batch`.
- **Worker** — stateless per batch: reads each input tensor from SHM by
  descriptor, runs `model.execute`, stages each output back into its own SHM
  arena, and replies with the output descriptors. One worker per GPU.

Every tensor edge crosses via SHM (one D2H on the producer, one H2D on the
consumer) — uniform, and exactly the cross-worker path. Intermediates that
happen to be produced and consumed on the same worker still round-trip
through SHM here; keeping same-worker edges on-device is a later optimization.

The control protocol is msgpack (a real, safe wire format — not pickle);
tensor bytes never travel over it, only `(arena, offset, nbytes, dims,
dtype)` descriptors.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import msgpack
import torch

_DISAGG_DEBUG = bool(os.environ.get("MSTAR_DISAGG_DEBUG"))


def _dbg(who: str, msg: str) -> None:
    if _DISAGG_DEBUG:
        print(f"[{who}] {msg}", flush=True)


class _IncrementalDetok:
    """Decode a stream of text token ids into cleanly-splittable pieces.

    mstar detokenizes in the data-worker; the Rust frontend carries no
    tokenizer, so the conductor detokenizes at the frontend seam. Decoding the
    whole prefix each step (then emitting the new suffix) keeps multi-token
    characters and BPE merges intact — a partial code point never ships as a
    broken UTF-8 fragment."""

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer
        self._ids: list[int] = []
        self._text = ""

    def step(self, token_id: int) -> str:
        self._ids.append(token_id)
        full = self._tok.decode(self._ids, skip_special_tokens=True)
        piece = full[len(self._text):]
        self._text = full
        return piece


class _FrontendStream:
    """Encode postprocessed emissions as frontend `ResultChunk`s and ship them.

    The seam both serving backends share (centralized `Conductor` and
    decentralized `DisaggCoordinator`): text emissions (token ids) are
    detokenized here with an incremental decoder per request — mstar
    detokenizes in the data-worker; the Rust frontend carries no tokenizer —
    audio emissions ship as int16 PCM at the model rate (the frontend wraps
    them in a WAV container), anything else as raw bytes."""

    def __init__(self, mbox, tokenizer) -> None:
        self._mbox = mbox
        self._tokenizer = tokenizer
        self._detoks: dict[Any, _IncrementalDetok] = {}
        # Requests whose FRONTEND detokenizes (it submitted token ids): text
        # emissions ship as raw ids (modality "token", 4-byte LE) instead of
        # detokenized pieces — the Rust side holds the incremental decoder.
        self._raw: set = set()

    def set_raw(self, front_rid) -> None:
        self._raw.add(front_rid)

    def _send(self, front_rid, modality: str, data: bytes) -> None:
        self._mbox.send("frontend", msgpack.packb(
            {"t": "chunk", "rid": front_rid, "modality": modality,
             "data": data, "metadata": {}}))

    def emit(self, front_rid, modality: str, value) -> None:
        if modality == "text":
            if front_rid in self._raw:
                self._send(front_rid, "token", int(value).to_bytes(4, "little"))
                return
            detok = self._detoks.get(front_rid)
            if detok is None:
                detok = self._detoks[front_rid] = _IncrementalDetok(self._tokenizer)
            piece = detok.step(int(value))
            if piece:
                self._send(front_rid, "text", piece.encode("utf-8"))
        elif modality == "audio":
            import numpy as np
            pcm = np.ascontiguousarray(value.numpy()).astype("<i2").tobytes()
            self._send(front_rid, "audio", pcm)
        else:
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self._send(front_rid, modality, bytes(data))

    def done(self, front_rid) -> None:
        self._detoks.pop(front_rid, None)
        self._raw.discard(front_rid)
        self._mbox.send("frontend", msgpack.packb({"t": "done", "rid": front_rid}))

    def error(self, front_rid, msg: str) -> None:
        """Terminal backend failure — the frontend surfaces it as an HTTP 500
        (non-streaming) or an SSE/NDJSON error event (streaming)."""
        self._detoks.pop(front_rid, None)
        self._raw.discard(front_rid)
        self._mbox.send("frontend", msgpack.packb(
            {"t": "err", "rid": front_rid, "msg": str(msg)}))

from mstar_rs._core import Runtime, SegmentedShmArena, ShmArena, ZmqCommunicator

from .driver import Driver

# ---- tensor <-> shared-memory codec -----------------------------------------

_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_str(dtype: torch.dtype) -> str:
    """Map a torch dtype to its wire tag, with a clear error for unsupported
    dtypes (e.g. fp8/float64) instead of a bare KeyError that would kill the
    batch — and, in the worker, the process."""
    try:
        return _DTYPE_TO_STR[dtype]
    except KeyError:
        raise TypeError(
            f"dtype {dtype} cannot cross the SHM seam; add it to _DTYPE_TO_STR"
        ) from None


class ShmPool:
    """This entity's producer buffer — a grow-by-segments arena — plus a cache
    of opened peer arenas for reading. A descriptor is
    ``[arena_name, offset, nbytes, dims, dtype_str]``; with segmentation the
    arena_name is a segment name (``mstar_rs_{entity}.seg{i}``), which is an
    ordinary named arena, so the consumer side is unchanged.

    Growth replaces the old fixed 1 GB arena's hard `Full` error: a new
    fixed-size segment is added under pressure (a dedicated one for oversized
    tensors), up to ``max_segments``. Segments never move once created —
    `on_new_segment(name, segment)` fires exactly once per segment, which is
    where a CUDA host-registration (`cudaHostRegister(ptr, len)`) belongs so
    D2H/H2D copies through the segment can run truly async on side streams."""

    def __init__(self, entity_id: str, segment_size: int = 1 << 28,
                 max_segments: int = 32, on_new_segment=None) -> None:
        self.entity_id = entity_id
        self.base = f"mstar_rs_{entity_id}"
        self.arena = SegmentedShmArena.create(self.base, segment_size, max_segments)
        self.on_new_segment = on_new_segment
        self._views: list = []                 # segment idx -> memoryview
        self._own: dict[str, int] = {}          # own segment name -> idx
        self._peers: dict[str, ShmArena] = {}   # peer arena/segment name -> handle
        self._sync_segments()

    def _sync_segments(self) -> None:
        while len(self._views) < self.arena.num_segments:
            i = len(self._views)
            seg = self.arena.segment(i)
            self._views.append(memoryview(seg))
            self._own[self.arena.segment_name(i)] = i
            if self.on_new_segment is not None:
                self.on_new_segment(self.arena.segment_name(i), seg)

    @staticmethod
    def entity_of(name: str) -> str:
        """The owning entity id of an arena/segment name."""
        return name[len("mstar_rs_"):].split(".seg")[0]

    def stage(self, t: torch.Tensor) -> list:
        """Copy a tensor's bytes into our arena; return its descriptor."""
        t = t.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        seg, off = self.arena.reserve(max(nbytes, 1))
        self._sync_segments()  # growth may have added a segment
        if nbytes:  # torch.frombuffer rejects an empty buffer
            mv = self._views[seg][off:off + nbytes]
            torch.frombuffer(mv, dtype=t.dtype).copy_(t.flatten())
        return [self.arena.segment_name(seg), off, nbytes, list(t.shape),
                _dtype_str(t.dtype)]

    def read(self, desc: list) -> torch.Tensor:
        name, off, nbytes, dims, dtype_str = desc
        dtype = _STR_TO_DTYPE[dtype_str]
        if nbytes == 0:  # empty tensor: nothing to map, reconstruct from dims
            return torch.empty(dims, dtype=dtype)
        if (i := self._own.get(name)) is not None:
            mv = self._views[i][off:off + nbytes]
        else:
            arena = self._peers.get(name)
            if arena is None:
                arena = ShmArena.open(name)
                self._peers[name] = arena
            mv = memoryview(arena)[off:off + nbytes]
        flat = torch.frombuffer(mv, dtype=dtype).clone()
        return flat.reshape(dims)

    def owns(self, name: str) -> bool:
        return name in self._own

    def free_by_name(self, name: str, off: int) -> bool:
        """Release an offset in one of OUR segments (by descriptor name)."""
        i = self._own.get(name)
        return self.arena.free(i, off) if i is not None else False

    def free(self, desc: list) -> None:
        self.free_by_name(desc[0], desc[1])


# ---- worker -----------------------------------------------------------------


class Worker:
    """Runs on one GPU. Loops on the mailbox: execute node batches, reply.

    Takes a `ModelEngine` (the data-plane half — this is where model weights
    live and torch compute runs). A full `Model` satisfies `ModelEngine`, so
    either can be passed."""

    def __init__(self, worker_id: str, engine, socket_dir: str, device: str = "cpu") -> None:
        self.worker_id = worker_id
        self.engine = engine
        self.device = device
        self.mbox = ZmqCommunicator(worker_id, socket_dir)
        self.shm = ShmPool(worker_id)

    def run(self) -> None:
        while True:
            raw = self.mbox.recv_timeout(1000)
            if raw is None:
                continue
            msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            t = msg["t"]
            if t == "shutdown":
                return
            if t == "execute":
                self._handle_execute(msg)
            elif t == "register":
                # Per-request engine config (arrives before the rid's first
                # execute — the conductor sends it at ingest, ordered per peer).
                self.engine.register_request(int(msg["rid"]), dict(msg["kwargs"]))
            elif t == "release":
                self.engine.release_request(int(msg["rid"]))
            elif t == "free":
                # Conductor reclaims a finished request's tensors: release our
                # (segment, offset)s — all were read/cloned by the conductor.
                for name, off in msg["items"]:
                    self.shm.free_by_name(name, off)

    def _handle_execute(self, msg: dict) -> None:
        batch_id = msg["batch_id"]
        rids = [int(rid) for rid in msg["inputs"].keys()]
        # TP: a partition sharded across N ranks runs the SAME execute on the
        # SAME inputs in NCCL lockstep (the collectives inside the model forward
        # synchronize them). Every rank reads inputs from the conductor's SHM
        # arena (host SHM, readable by all processes) — no NCCL input-broadcast
        # needed. The residual stream is replicated, so all ranks produce
        # identical outputs; only rank 0 stages + replies (the replicas discard
        # theirs). Freeing inputs on rank 0's `done` is safe: rank 0 can only
        # finish execute after every replica passed the shared all-reduces,
        # which come *after* the input read at execute's start.
        tp_rank = msg.get("tp_rank", 0)
        try:
            # inputs: {rid: {name: [descriptor, ...]}}
            inputs = {
                int(rid): {
                    name: [self.shm.read(d).to(self.device) for d in descs]
                    for name, descs in named.items()
                }
                for rid, named in msg["inputs"].items()
            }
            # KV models: {rid: {label, pages, seq_pos, append_len, ...}}. Keys
            # come back from msgpack as ints; normalize so kv[rid] matches the
            # int rids the engine iterates. None for stateless models.
            kv = msg.get("kv")
            if kv is not None:
                kv = {int(rid): view for rid, view in kv.items()}
            outputs = self.engine.execute(msg["node"], msg["walk"], inputs, kv=kv)
            if tp_rank != 0:
                return  # replica: outputs are replicated on rank 0; stay silent
            # check_stop: report loops the model wants terminated this pass
            # (the conductor replays them before complete_batch).
            stops = self.engine.loops_to_finish()
            # Stage every output tensor into our arena; reply with descriptors.
            out_desc = {
                rid: {
                    name: [self.shm.stage(t) for t in tensors]
                    for name, tensors in named.items()
                }
                for rid, named in outputs.items()
            }
        except Exception as e:  # never let one bad batch kill the worker
            import traceback

            traceback.print_exc()
            self.mbox.send(
                "conductor",
                msgpack.packb({"t": "error", "batch_id": batch_id, "rids": rids, "msg": repr(e)}),
            )
            return
        self.mbox.send(
            "conductor",
            msgpack.packb(
                {"t": "done", "batch_id": batch_id, "outputs": out_desc, "stops": stops}
            ),
        )


# ---- conductor --------------------------------------------------------------


class Conductor:
    """Drives the Runtime + model policy; dispatches batches to workers.

    Takes a `ModelPolicy` (the control-plane half — graph, walk seeding,
    continuation policy, postprocess). It never runs the model, so it holds no
    weights: pass a weightless `ModelPolicy` and only the workers load weights
    (mirrors mstar, whose conductor holds a policy-only `Model` while weights
    materialize lazily in the worker). A full `Model` also satisfies
    `ModelPolicy`, so it can be passed for models not yet split."""

    def __init__(
        self,
        policy,
        node_to_worker: dict[str, str],
        socket_dir: str,
        max_batch_size: int = 8,
    ) -> None:
        self.policy = policy
        # A node maps to a LIST of worker ids — its TP ranks, rank 0 first. A
        # bare string is a single-GPU node (a 1-element group), kept for
        # back-compat with non-TP callers.
        self.node_to_workers = {
            n: ([w] if isinstance(w, str) else list(w))
            for n, w in node_to_worker.items()
        }
        self.max_batch_size = max_batch_size
        self.runtime = Runtime(_spec_json(policy))
        if (kv := policy.kv_config()) is not None:
            self.runtime.configure_kv(*kv)
        if unbatchable := policy.unbatchable():
            self.runtime.configure_unbatchable([tuple(p) for p in unbatchable])
        self.mbox = ZmqCommunicator("conductor", socket_dir)
        self.shm = ShmPool("conductor")
        # uuid -> descriptor for every tensor the runtime is routing.
        self.desc: dict[int, list] = {}
        # request id -> uuids minted for it, so SHM/descriptors can be reclaimed
        # when the request finishes (mirrors the single-process store's
        # free_request; without it the arenas and `desc` grow without bound).
        self._req_uuids: dict[int, list[int]] = {}
        self.results: dict[int, list[Any]] = {}
        self.errors: dict[int, str] = {}  # request id -> worker error message
        self._inflight = 0        # batches dispatched but not yet completed
        self.finished: set[int] = set()  # request ids whose request completed
        # Serving hooks (set by a frontend bridge): stream each emission and
        # signal completion, keyed by the *frontend's* request id.
        self.on_token = None      # callable(front_rid, value)
        self.on_emit = None       # callable(front_rid, modality, value) — multimodal frontend
        self.on_done = None       # callable(front_rid)
        self.on_error = None      # callable(front_rid, msg) — terminal failure
        self._stream: _FrontendStream | None = None  # set by serve_frontend
        self._front_rid: dict[int, int] = {}  # runtime rid -> frontend rid
        self._stop = threading.Event()  # graceful shutdown for serve_frontend

    # -- request ingestion --

    def submit(self, request: dict[str, Any]) -> int:
        rid = self.runtime.add_request()
        request = dict(request, request_id=rid)
        self.results[rid] = []
        self._req_uuids[rid] = []
        # Per-request engine config (sampling knobs, voice, seed) to every
        # worker BEFORE any execute can be dispatched for this rid (the mbox is
        # ordered per peer) — mstar's model_kwargs-at-ingest, worker side.
        if mk := request.get("model_kwargs"):
            reg = msgpack.packb({"t": "register", "rid": rid, "kwargs": dict(mk)})
            for worker in {w for ws in self.node_to_workers.values() for w in ws}:
                self.mbox.send(worker, reg)
        for nxt in self.policy.initial_walks(request):
            self._start_walk(rid, nxt)
        return rid

    def _start_walk(self, rid: int, next_walk) -> None:
        walk, inputs, *rest = next_walk
        kv_appends = rest[0] if len(rest) > 0 else None
        kv_scratch = rest[1] if len(rest) > 1 else None
        seeded = []
        for node, name, tensors in inputs:
            refs = []
            for t in tensors:
                uuid = self.runtime.new_uuid()
                self.desc[uuid] = self.shm.stage(t)  # seed -> SHM
                self._req_uuids.setdefault(rid, []).append(uuid)
                refs.append((uuid, list(t.shape), _dtype_str(t.dtype)))
            seeded.append((node, name, refs))
        self.runtime.start_walk(rid, walk, seeded, kv_appends, kv_scratch)

    # -- the drive loop --

    def poll(self, timeout_ms: int = 5) -> bool:
        """One non-blocking drive step across ALL active requests (continuous
        batching): dispatch every ready batch, then drain whatever has arrived
        on the inbox. The inbox carries two kinds of message — worker `done`
        (a batch completed) and, when a frontend is attached, `submit` (a new
        request). Both are demuxed here by `t`. Returns True if anything
        progressed. This is the serving primitive — the HTTP layer runs it in
        a background thread while requests stream in and out concurrently."""
        did = False
        while (batch := self.runtime.next_batch(self.max_batch_size)) is not None:
            self._dispatch(batch)
            self._inflight += 1
            did = True
        # Block up to `timeout_ms` for the first message, then drain whatever
        # else is queued non-blocking. Blocking (rather than a bare try_recv)
        # is what keeps an idle conductor asleep instead of spinning a core —
        # recv_timeout wakes on either a worker `done` or a new `submit`.
        raw = self.mbox.recv_timeout(timeout_ms)
        while raw is not None:
            did = True
            self._handle_message(raw)
            raw = self.mbox.try_recv()
        return did

    def _handle_message(self, raw: bytes) -> None:
        """Demux an inbox message by type: worker completion, worker error, or
        a new request from a frontend."""
        msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        t = msg["t"]
        if t == "done":
            self._process_done(msg)
            self._inflight -= 1
        elif t == "error":
            self._process_error(msg)
            self._inflight -= 1
        elif t == "submit":
            self._ingest_submit(msg)
        elif t == "abort":
            self._abort_front(msg["rid"])

    def _abort_front(self, front_rid) -> None:
        """The frontend gave up on a request (client disconnect / timeout):
        release its engine + runtime state. Idempotent — an already-finished
        request is a no-op (mstar's abort_request)."""
        rid = next((r for r, f in self._front_rid.items() if f == front_rid), None)
        if rid is None:
            return
        self._front_rid.pop(rid, None)  # no done/err back; the frontend is gone
        try:
            self.runtime.finish_request(rid)
        except Exception:
            pass  # in-flight batches for it complete as no-ops
        self._finish_request(rid)
        self.results.pop(rid, None)

    def _ingest_submit(self, msg: dict) -> None:
        """A frontend submitted a flattened request (text + media file paths +
        in/out modalities + model_kwargs, mirroring mstar's PreprocessInput):
        start it, and remember the frontend's request id so emissions stream
        back. The policy tokenizes the text (and loads any media) when it builds
        the initial walks. Ingest failures (bad media, empty prompt) go back as
        a terminal error instead of killing the serve loop."""
        try:
            rid = self._do_submit(msg)
        except Exception as e:  # noqa: BLE001 — any ingest failure fails ONE request
            if self.on_error is not None:
                self.on_error(msg["rid"], repr(e))
            return
        self._front_rid[rid] = msg["rid"]
        if msg.get("frontend_detok") and self._stream is not None:
            # The frontend holds the incremental decoder for this request:
            # ship its text emissions as raw token ids.
            self._stream.set_raw(msg["rid"])

    def _do_submit(self, msg: dict) -> int:
        req = {
            "text": msg.get("text"),
            "file_paths": msg.get("file_paths") or {},
            "input_modalities": list(msg.get("input_modalities") or []),
            "output_modalities": list(msg.get("output_modalities") or ["text"]),
            "model_kwargs": dict(msg.get("model_kwargs") or {}),
        }
        if msg.get("tokens") is not None:
            req["tokens"] = list(msg["tokens"])  # frontend-tokenized prompt
        return self.submit(req)

    def run_until_idle(self) -> dict[int, list[Any]]:
        """Drive until no request has work left (used by the batch demo)."""
        while True:
            did = self.poll(timeout_ms=50)
            if not did and self._inflight == 0:
                break
        return self.results

    def _dispatch(self, batch) -> None:
        workers = self.node_to_workers[batch.node]
        # inputs: {rid: {name: [descriptor]}} resolved from uuids.
        inputs = {
            rid: {
                name: [self.desc[uuid] for (uuid, _dims, _dt) in refs]
                for name, refs in named.items()
            }
            for rid, named in batch.inputs.items()
        }
        # Send the same execute to every TP rank (rank 0 first). All ranks read
        # inputs from our SHM arena and run in NCCL lockstep; only rank 0 (the
        # leader) replies with outputs, so `_inflight` counts one per batch.
        # `batch.kv` carries the runtime's per-request KV view (label, pages,
        # seq_pos, append_len, scratch_len) for KV models — plain ints/lists, so
        # it rides the msgpack control wire (never tensor bytes).
        for tp_rank, worker in enumerate(workers):
            self.mbox.send(
                worker,
                msgpack.packb(
                    {"t": "execute", "batch_id": batch.batch_id, "node": batch.node,
                     "walk": batch.walk, "inputs": inputs, "kv": batch.kv,
                     "tp_rank": tp_rank}
                ),
            )

    def _process_done(self, msg: dict) -> None:
        """Route a worker's `done` message: mint output uuids, record their
        SHM location, replay any loop-finish signals the worker detected during
        execute, and route via complete_batch."""
        # Mint uuids for the worker's outputs + record their SHM location.
        outputs = {}
        for rid, named in msg["outputs"].items():
            rid = int(rid)
            out = {}
            for name, descs in named.items():
                refs = []
                for d in descs:
                    uuid = self.runtime.new_uuid()
                    self.desc[uuid] = d
                    self._req_uuids.setdefault(rid, []).append(uuid)
                    refs.append((uuid, d[3], d[4]))  # (uuid, dims, dtype)
                out[name] = refs
            outputs[rid] = out
        # check_stop -> STOP_LOOPS: the worker ran execute (where a model
        # detects EOS), so it reports the loops to finish; replay them here
        # BEFORE complete_batch so the loop terminates this iteration rather
        # than advancing (mirrors the single-process Driver ordering). Without
        # this, a looping model served through the conductor never stops early.
        for rid, loop_name in msg.get("stops", []):
            self.runtime.signal_loop_finish(int(rid), loop_name)
        for event in self.runtime.complete_batch(msg["batch_id"], outputs):
            self._handle_event(event)

    def _process_error(self, msg: dict) -> None:
        """A worker raised during execute: fail every request in that batch so
        the serving layer resolves (instead of hanging on a dead worker) and
        its SHM is reclaimed."""
        detail = msg.get("msg", "worker error")
        for rid in msg.get("rids", []):
            rid = int(rid)
            self.errors[rid] = detail
            # Frontend-streamed request: surface the failure as a terminal
            # error (HTTP 500 / SSE error event), not a clean `done`.
            front = self._front_rid.pop(rid, None)
            if front is not None and self.on_error is not None:
                self.on_error(front, detail)
            try:
                self.runtime.finish_request(rid)
            except Exception:
                pass  # request may already be gone; failing it is best-effort
            self._finish_request(rid)

    def _handle_event(self, event: dict) -> None:
        rid = event["request_id"]
        if event["type"] == "emission":
            tensors = [self.shm.read(self.desc[uuid]) for (uuid, _d, _t) in event["tensors"]]
            value = self.policy.postprocess(event["name"], event["modality"], tensors)
            self.results[rid].append(value)
            # Stream this emission to the frontend, if one is attached. A
            # multimodal frontend (on_emit) gets the modality so it can build a
            # ResultChunk; the legacy token-only hook (on_token) still works.
            if rid in self._front_rid:
                front = self._front_rid[rid]
                if self.on_emit is not None:
                    self.on_emit(front, event["modality"], value)
                elif self.on_token is not None:
                    self.on_token(front, value)
        elif event["type"] == "free":
            # Per-tensor reclaim: free SHM for tensors the runtime reports
            # unreachable. Emitted after this batch's emission/walk_done events,
            # so any read of these buffers already happened.
            self._free_uuids(event["uuids"])
        elif event["type"] == "walk_done":
            persist = {
                name: [self.shm.read(self.desc[uuid]) for (uuid, _d, _t) in refs]
                for name, refs in event["persist"].items()
            }
            nxt = self.policy.next_forward(
                rid, event["partition"], event["walk"], event["fwd_index"],
                persist, event["stream_done"],
            )
            if nxt is None:
                if self.runtime.finish_partition(rid, event["partition"]):
                    self.runtime.finish_request(rid)
                    self._finish_request(rid)
            else:
                self._start_walk(rid, nxt)

    def _finish_request(self, rid: int) -> None:
        """A request completed (or failed): reclaim its SHM + descriptors, then
        do path-appropriate bookkeeping. A frontend bridge already streamed the
        output, so drop the server-side buffers and fire on_done; otherwise
        retain `results`/`finished` for a drive loop (ServingEngine) to consume."""
        self._reclaim(rid)
        # Free per-request engine state (sampler slots, voice) on every worker.
        rel = msgpack.packb({"t": "release", "rid": rid})
        for worker in {w for ws in self.node_to_workers.values() for w in ws}:
            self.mbox.send(worker, rel)
        front = self._front_rid.pop(rid, None)
        if self.on_done is not None:
            if front is not None:
                self.on_done(front)
            self.results.pop(rid, None)  # streamed already; don't accumulate
        else:
            self.finished.add(rid)  # drive loop (ServingEngine) consumes + pops

    def _free_uuids(self, uuids) -> None:
        """Free SHM for a set of uuids: conductor-owned offsets locally,
        worker-owned offsets via a `free` message to the owning worker.
        Idempotent — a uuid already freed is no longer in `desc` and is
        skipped (so the request-finish backstop and the incremental Free
        events compose without double-freeing)."""
        by_worker: dict[str, list] = {}
        for uuid in uuids:
            desc = self.desc.pop(uuid, None)
            if desc is None:
                continue
            arena_name, off = desc[0], desc[1]
            if self.shm.owns(arena_name):
                self.shm.free_by_name(arena_name, off)
            else:  # segment name is f"mstar_rs_{worker_id}.seg{i}"
                by_worker.setdefault(ShmPool.entity_of(arena_name), []).append(
                    (arena_name, off))
        for worker_id, items in by_worker.items():
            self.mbox.send(worker_id, msgpack.packb({"t": "free", "items": items}))

    def _reclaim(self, rid: int) -> None:
        """Backstop at request finish: free whatever this request still holds.
        The incremental Free events reclaim most tensors mid-request (per
        mstar's refcount reclaim); this sweeps the remainder (persisted /
        cross-partition tensors), mirroring mstar's cleanup_request."""
        self._free_uuids(self._req_uuids.pop(rid, []))

    # -- frontend serving --

    def serve_frontend(self) -> None:
        """Run forever as a streaming backend for the Rust axum frontend.

        The frontend sends `submit` messages (text + media file paths + in/out
        modalities + model_kwargs) on our inbox; we wire each emission back as a
        `ResultChunk` ({modality, data, metadata}) and signal completion, over
        the same `ZmqCommunicator` mesh as msgpack. `submit`/`chunk`/`done` are the
        only messages that cross this seam — tensor bytes never do (those move
        worker<->conductor through SHM)."""
        stream = _FrontendStream(self.mbox, getattr(self.policy, "tokenizer", None))
        self._stream = stream
        self.on_emit = stream.emit
        self.on_done = stream.done
        self.on_error = stream.error
        # Short poll timeout bounds shutdown latency: after stop() the loop
        # returns from the blocking recv within this window and exits, so the
        # thread is joinable before interpreter teardown (a daemon thread stuck
        # in a native recv at finalization cores the process).
        while not self._stop.is_set():
            self.poll(timeout_ms=20)

    def stop(self) -> None:
        """Signal serve_frontend to exit; join the serving thread after this."""
        self._stop.set()

    def shutdown_workers(self) -> None:
        for worker in {w for ws in self.node_to_workers.values() for w in ws}:
            self.mbox.send(worker, msgpack.packb({"t": "shutdown"}))


def _spec_json(policy) -> str:
    import json

    if (topo := policy.partitions()) is not None:
        partition_specs, connection_specs = topo
        return json.dumps(
            {"walks": policy.walks(), "partitions": partition_specs, "connections": connection_specs}
        )
    return json.dumps(policy.walks())


# ---- decentralized (in-worker scheduler) multi-process runtime --------------
#
# The DECENTRALIZED alternative to the Conductor above. Each worker owns a
# partition, holds its OWN scheduler (a `Driver` scoped via local_partitions),
# and drives its own loop — so the per-step decode loop pays NO conductor round
# trip (mstar's in-worker MicroScheduler model). Cross-partition tensors stream
# worker->worker peer-to-peer (SHM descriptor + a ZMQ `stream` notify), never
# through a scheduler. A thin coordinator only ingests requests (seeds each
# partition's worker once), relays emissions, and finishes a request when every
# partition reports done. Phase-2 of the per-step-dispatch work; see
# examples/verify_disagg.py.


class DisaggWorker:
    """Self-driving worker for one (or more) partition(s). Runs its own Driver
    loop; ships stream outputs to peer workers; injects peer chunks locally."""

    def __init__(self, worker_id, engine, local_partitions, partition_to_worker,
                 socket_dir, device: str = "cpu", io_leader: bool = True,
                 coordinator_id: str = "coordinator",
                 tp_nodes: list | None = None,
                 tp_followers: list | None = None,
                 tp_follow_nodes: list | None = None) -> None:
        self.worker_id = worker_id
        # Tensor-parallel batch coordination (mstar's ScheduleTPNode):
        #   leader rank:   tp_nodes = the TP nodes it initiates; tp_followers =
        #                  the follower worker ids to ship each batch decision to
        #                  (BEFORE executing), so every rank runs identical
        #                  batches in identical order under concurrency.
        #   follower rank: tp_follow_nodes = nodes it never self-initiates; it
        #                  replays the leader's shipped decisions instead.
        self.tp_nodes = set(tp_nodes or [])
        self.tp_followers = list(tp_followers or [])
        self.driver = Driver(engine, local_partitions=local_partitions,
                             tp_follow_nodes=tp_follow_nodes)
        if self.tp_nodes and self.tp_followers:
            self.driver.on_batch = self._broadcast_batch
        # follower: leader decisions whose gids aren't all mapped locally yet
        # (the tp_sched can overtake the seed — different senders).
        self._tp_pending: list[tuple] = []
        self.mbox = ZmqCommunicator(worker_id, socket_dir)
        self.shm = ShmPool(worker_id)
        self.partition_to_worker = partition_to_worker
        # Where emissions/partition-done go. "coordinator" for the standalone
        # DisaggCoordinator; the serving launcher binds the coordinator as
        # "conductor" so the Rust frontend's submits land in the same inbox.
        self.coordinator_id = coordinator_id
        self.device = device
        # io_leader=False: a tensor-parallel FOLLOWER rank. It runs the SAME
        # deterministic schedule + compute as the leader (so the per-layer NCCL
        # all-reduces stay in lockstep), but suppresses all outward I/O —
        # shipping stream chunks, emissions, and partition-done — because its
        # outputs are replicas of the leader's. Only rank 0 of a TP group ships.
        self.io_leader = io_leader
        self.device = device
        self._g2l: dict[int, int] = {}   # global request id -> local runtime id
        self._l2g: dict[int, int] = {}
        self._stop = False
        # producer-done notifications to ship AFTER this tick's stream chunks
        # (order: chunks first, then done) — (gid, from, edge, to).
        self._pending_done: list[tuple] = []
        # emissions + partition-done fire from inside driver.poll(); relay them
        # (on_emit carries the modality, which the frontend seam needs).
        self.driver.on_emit = self._emit
        self.driver.on_partition_done = self._partition_done

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 — report the crash, then die loudly
            # Tell the coordinator which requests die with us so it fails them
            # (frontend 500 / errors dict) instead of hanging forever on a
            # worker that will never report partition_done. Best-effort — the
            # exception still propagates (the process exits as before).
            try:
                if self.io_leader:
                    self.mbox.send(self.coordinator_id, msgpack.packb(
                        {"t": "worker_error", "worker": self.worker_id,
                         "gids": list(self._g2l.keys()), "msg": repr(e)}))
            except Exception:
                pass
            raise

    def _run(self) -> None:
        while not self._stop:
            raw = self.mbox.recv_timeout(50)   # block when idle (no busy spin)
            # Drain a BOUNDED number of queued messages, then always poll — an
            # unbounded drain would starve poll() when a peer floods this worker
            # with stream chunks faster than it can consume them (the consumer
            # would buffer forever and never drive its own partition).
            n = 0
            while raw is not None and n < 16:
                self._handle(raw)
                if self._stop:
                    return
                n += 1
                raw = self.mbox.try_recv()
            self._drain_tp_pending()
            did = self.driver.poll()           # one in-process step, no round-trip
            if did:
                _dbg(self.worker_id, "poll ran a batch")
            self._ship_outbox()

    def _broadcast_batch(self, node: str, walk: str, rids: list) -> None:
        """TP leader: ship this batch decision to follower ranks BEFORE
        executing (mstar sends ScheduleTPNode before GPU submission), so the
        followers replay identical batches in identical order."""
        if node not in self.tp_nodes:
            return
        gids = [self._l2g[r] for r in rids]
        msg = msgpack.packb({"t": "tp_sched", "node": node, "walk": walk, "gids": gids})
        for follower in self.tp_followers:
            self.mbox.send(follower, msg)

    def _drain_tp_pending(self) -> None:
        """Follower: register leader decisions whose gids are all mapped
        locally (the tp_sched can overtake the seed; FIFO order preserved)."""
        while self._tp_pending:
            node, walk, gids = self._tp_pending[0]
            if any(g not in self._g2l for g in gids):
                return  # head's seed(s) not ingested yet; keep order, retry
            self.driver.register_tp_follow(
                node, walk, [self._g2l[g] for g in gids])
            self._tp_pending.pop(0)

    def _flush_follows_for(self, local: int, gid: int, timeout_s: float = 5.0) -> None:
        """Run the driver until no queued leader decision references this
        request (bounded). The referenced batches are executable by
        construction — the leader only ships batches it scheduled from the
        same deterministic state."""
        def pending() -> bool:
            return (any(gid in gs for _n, _w, gs in self._tp_pending)
                    or any(local in rs for _n, _w, rs in self.driver._tp_follow))

        deadline = time.time() + timeout_s
        while pending():
            self._drain_tp_pending()
            if not self.driver.poll():
                if time.time() > deadline:
                    _dbg(self.worker_id, f"flush_follows timeout gid={gid}")
                    return
                time.sleep(0.001)

    def _handle(self, raw: bytes) -> None:
        msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        t = msg["t"]
        if t == "shutdown":
            # TP leader: forward on the ordered tp_sched connection so a
            # follower never sees shutdown overtake a queued batch decision
            # (the coordinator shuts down only leaders).
            for follower in self.tp_followers:
                self.mbox.send(follower, raw)
            self._stop = True
        elif t == "seed":
            gid = int(msg["gid"])
            _dbg(self.worker_id, f"seed gid={gid} walk={msg['walk']}")
            # A worker owning several partitions gets ONE seed per partition for
            # the same request — they all share a single local request, so only
            # create it on the first seed and reuse it thereafter (else each
            # partition's walk lands in a separate request and cross-partition
            # streams never meet their consumer).
            local = self._g2l.get(gid)
            if local is None:
                local = self.driver.new_request()
                self._g2l[gid] = local
                self._l2g[local] = gid
                self.driver._front_rid[local] = gid  # so on_emit fires with gid
                # per-request engine config (sampling, voice) — once per gid
                self.driver.model.register_request(local, dict(msg.get("mk") or {}))
            inputs = [
                (node, name, [self.shm.read(d).to(self.device) for d in descs])
                for node, name, descs in msg["inputs"]
            ]
            self.driver.start(local, msg["walk"], inputs, *msg["kv"])
        elif t == "stream":
            _dbg(self.worker_id, f"stream gid={msg['gid']} {msg['from']}->{msg['to']}")
            local = self._g2l[int(msg["gid"])]
            tensors = [self.shm.read(d).to(self.device) for d in msg["descs"]]
            self.driver.inject(local, msg["from"], msg["edge"], msg["to"], tensors)
        elif t == "stream_done":
            _dbg(self.worker_id, f"stream_done gid={msg['gid']} {msg['from']}->{msg['to']}")
            local = self._g2l[int(msg["gid"])]
            self.driver.signal_stream_done(local, msg["from"], msg["edge"], msg["to"])
        elif t == "tp_sched":
            # leader's batch decision (TP follow) — replay after gid mapping
            _dbg(self.worker_id, f"tp_sched {msg['node']} gids={msg['gids']}")
            self._tp_pending.append((msg["node"], msg["walk"], [int(g) for g in msg["gids"]]))
        elif t == "finish":
            gid = int(msg["gid"])
            # TP leader: forward the finish to follower ranks over the SAME
            # ordered connection that carries tp_sched, so a follower can never
            # see finish overtake a batch decision (the coordinator finishes
            # only leaders — ordering by construction, not by luck).
            for follower in self.tp_followers:
                self.mbox.send(follower, raw)
            local = self._g2l.get(gid)
            if local is not None:
                # Replay any queued leader decisions that still reference this
                # request before teardown (the gid must stay mapped while the
                # flush translates + runs them).
                self._flush_follows_for(local, gid)
                self._g2l.pop(gid, None)
                self._l2g.pop(local, None)
                self.driver._front_rid.pop(local, None)
                self.driver.runtime.finish_request(local)
                self.driver.store.free_request(local)
                self.driver.model.release_request(local)

    def _ship_outbox(self) -> None:
        if not self.io_leader:
            self.driver.outbox.clear()   # follower: replica output, don't ship
            return
        for frm, edge, to, tensors, local in self.driver.outbox:
            gid = self._l2g[local]
            _dbg(self.worker_id, f"ship {frm}->{to} gid={gid} -> {self.partition_to_worker[to]}")
            descs = [self.shm.stage(t) for t in tensors]
            self.mbox.send(
                self.partition_to_worker[to],
                msgpack.packb({"t": "stream", "gid": gid, "from": frm,
                               "edge": edge, "to": to, "descs": descs}),
            )
        self.driver.outbox.clear()
        # ship queued producer-done AFTER the chunks (chunks first, then done)
        for gid, frm, edge, to in self._pending_done:
            self.mbox.send(
                self.partition_to_worker[to],
                msgpack.packb({"t": "stream_done", "gid": gid, "from": frm,
                               "edge": edge, "to": to}),
            )
        self._pending_done.clear()

    def _emit(self, gid: int, modality: str, value) -> None:
        if not self.io_leader:
            return   # follower: replica emission, leader already sent it
        if isinstance(value, torch.Tensor):
            self.mbox.send(self.coordinator_id, msgpack.packb(
                {"t": "emission", "gid": gid, "modality": modality,
                 "desc": self.shm.stage(value)}))
        else:
            self.mbox.send(self.coordinator_id, msgpack.packb(
                {"t": "emission", "gid": gid, "modality": modality, "value": value}))

    def _partition_done(self, local: int, partition: str) -> None:
        if not self.io_leader:
            return   # follower: leader reports the partition done
        gid = self._l2g.get(local)
        if gid is None:
            return
        # Tell each consumer worker this producer partition is done, so its
        # continue_after_done streams switch to yielding empties (keeping its own
        # AR loop alive). Queued to ship AFTER this tick's stream chunks.
        for edge, to in self.driver.outgoing_cross_worker(partition):
            self._pending_done.append((gid, partition, edge, to))
        self.mbox.send(self.coordinator_id, msgpack.packb(
            {"t": "partition_done", "gid": gid, "partition": partition}))


class DisaggCoordinator:
    """Per-request ingest + egress for the decentralized workers. NOT per-step:
    it seeds each partition's worker once, relays emissions to the caller, and
    completes a request when all partitions report done."""

    def __init__(self, policy, partition_to_worker, socket_dir,
                 my_id: str = "coordinator") -> None:
        self.policy = policy
        # A partition maps to a LIST of workers (its TP ranks; a bare string is
        # a 1-rank group). Seeds go to ALL ranks (they run in lockstep); the
        # first is the I/O leader (peers ship stream chunks to it).
        self.partition_to_workers = {
            p: ([w] if isinstance(w, str) else list(w))
            for p, w in partition_to_worker.items()
        }
        self.walk_partition = {
            w: p["name"] for p in policy.partitions()[0] for w in p["walks"]
        }
        self._all_partitions = [p["name"] for p in policy.partitions()[0]]
        # `my_id` is this coordinator's inbox name. The serving launcher binds
        # it as "conductor" (the peer the Rust frontend submits to) and points
        # the workers' coordinator_id here.
        self.mbox = ZmqCommunicator(my_id, socket_dir)
        self.shm = ShmPool(my_id)
        self._gid = 0
        self.results: dict[int, list[Any]] = {}
        self._pending: dict[int, set] = {}   # gid -> partitions not yet done
        self.finished: set[int] = set()
        self.errors: dict[int, str] = {}     # gid -> failure detail
        # Workers that reported a crash: their requests were failed, and new
        # submits that would seed them are refused up front.
        self._dead_workers: set[str] = set()
        # Frontend serving (mirrors Conductor.serve_frontend).
        self._front_rid: dict[int, Any] = {}   # gid -> frontend request id
        self._stream: _FrontendStream | None = None
        self._stop = threading.Event()

    def submit(self, request: dict[str, Any]) -> int:
        if dead := (self._dead_workers & self._all_workers()):
            raise RuntimeError(f"worker(s) {sorted(dead)} died; restart the stack")
        gid = self._gid
        self._gid += 1
        self.results[gid] = []
        self._pending[gid] = set(self._all_partitions)
        req = dict(request, request_id=gid)
        mk = dict(request.get("model_kwargs") or {})
        for nxt in self.policy.initial_walks(req):
            walk, inputs, *kv = nxt
            seeded = [[node, name, [self.shm.stage(t) for t in tensors]]
                      for node, name, tensors in inputs]
            msg = msgpack.packb(
                {"t": "seed", "gid": gid, "walk": walk, "inputs": seeded,
                 "kv": list(kv), "mk": mk})
            # seed every rank of the partition (TP group runs in lockstep)
            for worker in self.partition_to_workers[self.walk_partition[walk]]:
                self.mbox.send(worker, msg)
        return gid

    def poll(self, timeout_ms: int = 20) -> None:
        raw = self.mbox.recv_timeout(timeout_ms)
        while raw is not None:
            msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            t = msg["t"]
            if t == "emission":
                gid = int(msg["gid"])
                value = self.shm.read(msg["desc"]) if "desc" in msg else msg["value"]
                if gid in self._front_rid:
                    # Stream to the frontend (worker already postprocessed);
                    # don't accumulate server-side.
                    self._stream.emit(self._front_rid[gid], msg.get("modality", ""), value)
                elif gid in self.results:
                    self.results[gid].append(value)
                # else: late emission for an aborted request — drop it
            elif t == "partition_done":
                gid = int(msg["gid"])
                if gid not in self._pending:
                    raw = self.mbox.try_recv()
                    continue  # late report for an aborted request
                self._pending[gid].discard(msg["partition"])
                if not self._pending[gid]:
                    self.finished.add(gid)
                    # Finish only each partition's LEADER rank (workers[0]);
                    # leaders forward it to their TP followers on the same
                    # ordered connection as tp_sched, so a follower never sees
                    # finish overtake a batch decision.
                    fin = msgpack.packb({"t": "finish", "gid": gid})
                    for worker in {ws[0] for ws in self.partition_to_workers.values()}:
                        self.mbox.send(worker, fin)
                    if (front := self._front_rid.pop(gid, None)) is not None:
                        self._stream.done(front)
                        self.results.pop(gid, None)  # streamed already
            elif t == "submit":
                # A frontend submitted a flattened request (same wire shape the
                # centralized Conductor ingests). Ingest failures fail ONE
                # request instead of killing the serve loop.
                try:
                    req = {
                        "text": msg.get("text"),
                        "file_paths": msg.get("file_paths") or {},
                        "input_modalities": list(msg.get("input_modalities") or []),
                        "output_modalities": list(msg.get("output_modalities") or ["text"]),
                        "model_kwargs": dict(msg.get("model_kwargs") or {}),
                    }
                    if msg.get("tokens") is not None:
                        req["tokens"] = list(msg["tokens"])
                    gid = self.submit(req)
                except Exception as e:  # noqa: BLE001
                    if self._stream is not None:
                        self._stream.error(msg["rid"], repr(e))
                else:
                    self._front_rid[gid] = msg["rid"]
                    if msg.get("frontend_detok") and self._stream is not None:
                        self._stream.set_raw(msg["rid"])
            elif t == "abort":
                # Frontend gave up (disconnect/timeout): finish the request on
                # every leader (they forward to TP followers) — idempotent.
                front = msg["rid"]
                gid = next((g for g, f in self._front_rid.items() if f == front), None)
                if gid is not None:
                    self._front_rid.pop(gid, None)
                    self._pending.pop(gid, None)
                    self.results.pop(gid, None)
                    self.finished.add(gid)
                    fin = msgpack.packb({"t": "finish", "gid": gid})
                    for worker in {ws[0] for ws in self.partition_to_workers.values()}:
                        self.mbox.send(worker, fin)
            elif t == "worker_error":
                # A worker crashed: fail every request it carried (frontend
                # error / errors dict) so nothing hangs on a partition that
                # will never report done, tell the surviving leaders to drop
                # them, and refuse new submits that would seed the dead worker.
                detail = msg.get("msg", "worker died")
                dead = msg.get("worker", "")
                self._dead_workers.add(dead)
                for gid in msg.get("gids", []):
                    gid = int(gid)
                    if gid in self.finished:
                        continue
                    self._pending.pop(gid, None)
                    self.results.pop(gid, None)
                    self.errors[gid] = detail
                    self.finished.add(gid)
                    if (front := self._front_rid.pop(gid, None)) is not None \
                            and self._stream is not None:
                        self._stream.error(front, detail)
                    fin = msgpack.packb({"t": "finish", "gid": gid})
                    for worker in {ws[0] for ws in self.partition_to_workers.values()}:
                        if worker != dead:
                            self.mbox.send(worker, fin)
            raw = self.mbox.try_recv()

    def run_until_idle(self) -> dict[int, list[Any]]:
        while len(self.finished) < self._gid:
            self.poll()
        return self.results

    def serve_frontend(self) -> None:
        """Run forever as a streaming backend for the Rust axum frontend — the
        decentralized counterpart of `Conductor.serve_frontend`. Same seam
        (`submit` in; `chunk`/`done` out via `_FrontendStream`), but per-request
        only: workers self-schedule their decode loops, so nothing crosses this
        loop per step — it just ingests requests and relays emissions."""
        self._stream = _FrontendStream(self.mbox, self.policy.tokenizer)
        while not self._stop.is_set():
            self.poll(timeout_ms=20)

    def stop(self) -> None:
        """Signal serve_frontend to exit; join the serving thread after this."""
        self._stop.set()

    def _all_workers(self) -> set:
        return {w for ws in self.partition_to_workers.values() for w in ws}

    def shutdown_workers(self) -> None:
        # Shut down only each partition's LEADER; leaders forward to their TP
        # followers on the ordered tp_sched connection (same reasoning as
        # `finish`: shutdown must not overtake a queued batch decision).
        for worker in {ws[0] for ws in self.partition_to_workers.values()}:
            self.mbox.send(worker, msgpack.packb({"t": "shutdown"}))
