"""Multi-process runtime: a **conductor** process drives one or more
**worker** processes over the UDS control mesh (`Mailbox`), with tensors
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

import threading
from typing import Any

import msgpack
import torch

from mstar_rs._core import Mailbox, Runtime, ShmArena

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


class ShmPool:
    """One owned arena (this entity's producer buffer) + a cache of opened
    peer arenas for reading. A descriptor is
    ``[arena_name, offset, nbytes, dims, dtype_str]``."""

    def __init__(self, entity_id: str, size: int = 1 << 30) -> None:
        self.name = f"mstar_rs_{entity_id}"
        self.arena = ShmArena.create(self.name, size)
        self._peers: dict[str, ShmArena] = {self.name: self.arena}

    def stage(self, t: torch.Tensor) -> list:
        """Copy a tensor's bytes into our arena; return its descriptor."""
        t = t.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        off = self.arena.reserve(max(nbytes, 1))
        if nbytes:  # torch.frombuffer rejects an empty buffer
            mv = memoryview(self.arena)[off:off + nbytes]
            torch.frombuffer(mv, dtype=t.dtype).copy_(t.flatten())
        return [self.name, off, nbytes, list(t.shape), _DTYPE_TO_STR[t.dtype]]

    def read(self, desc: list) -> torch.Tensor:
        name, off, nbytes, dims, dtype_str = desc
        dtype = _STR_TO_DTYPE[dtype_str]
        if nbytes == 0:  # empty tensor: nothing to map, reconstruct from dims
            return torch.empty(dims, dtype=dtype)
        arena = self._peers.get(name)
        if arena is None:
            arena = ShmArena.open(name)
            self._peers[name] = arena
        mv = memoryview(arena)[off:off + nbytes]
        flat = torch.frombuffer(mv, dtype=dtype).clone()
        return flat.reshape(dims)

    def free(self, desc: list) -> None:
        if desc[0] == self.name:
            self.arena.free(desc[1])


# ---- worker -----------------------------------------------------------------


class Worker:
    """Runs on one GPU. Loops on the mailbox: execute node batches, reply."""

    def __init__(self, worker_id: str, model, socket_dir: str, device: str = "cpu") -> None:
        self.worker_id = worker_id
        self.model = model
        self.device = device
        self.mbox = Mailbox(worker_id, socket_dir)
        self.shm = ShmPool(worker_id)

    def run(self) -> None:
        while True:
            raw = self.mbox.recv_timeout(1000)
            if raw is None:
                continue
            msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            if msg["t"] == "shutdown":
                return
            if msg["t"] == "execute":
                self._handle_execute(msg)

    def _handle_execute(self, msg: dict) -> None:
        # inputs: {rid: {name: [descriptor, ...]}}
        inputs = {
            int(rid): {
                name: [self.shm.read(d).to(self.device) for d in descs]
                for name, descs in named.items()
            }
            for rid, named in msg["inputs"].items()
        }
        kv = msg.get("kv")  # reserved for KV models; None for stateless
        outputs = self.model.execute(msg["node"], msg["walk"], inputs, kv=kv)
        # Stage every output tensor into our arena; reply with descriptors.
        out_desc = {
            rid: {
                name: [self.shm.stage(t) for t in tensors]
                for name, tensors in named.items()
            }
            for rid, named in outputs.items()
        }
        self.mbox.send(
            "conductor",
            msgpack.packb({"t": "done", "batch_id": msg["batch_id"], "outputs": out_desc}),
        )


# ---- conductor --------------------------------------------------------------


class Conductor:
    """Drives the Runtime + model policy; dispatches batches to workers."""

    def __init__(
        self,
        model,
        node_to_worker: dict[str, str],
        socket_dir: str,
        max_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.node_to_worker = node_to_worker
        self.max_batch_size = max_batch_size
        self.runtime = Runtime(_spec_json(model))
        if (kv := model.kv_config()) is not None:
            self.runtime.configure_kv(*kv)
        self.mbox = Mailbox("conductor", socket_dir)
        self.shm = ShmPool("conductor")
        # uuid -> descriptor for every tensor the runtime is routing.
        self.desc: dict[int, list] = {}
        self.results: dict[int, list[Any]] = {}
        self._inflight = 0        # batches dispatched but not yet completed
        self.finished: set[int] = set()  # request ids whose request completed
        # Serving hooks (set by a frontend bridge): stream each emission and
        # signal completion, keyed by the *frontend's* request id.
        self.on_token = None      # callable(front_rid, value)
        self.on_done = None       # callable(front_rid)
        self._front_rid: dict[int, int] = {}  # runtime rid -> frontend rid
        self._stop = threading.Event()  # graceful shutdown for serve_frontend

    # -- request ingestion --

    def submit(self, request: dict[str, Any]) -> int:
        rid = self.runtime.add_request()
        request = dict(request, request_id=rid)
        for nxt in self.model.initial_walks(request):
            self._start_walk(rid, nxt)
        self.results[rid] = []
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
                refs.append((uuid, list(t.shape), _DTYPE_TO_STR[t.dtype]))
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
        """Demux an inbox message by type: worker completion vs new request."""
        msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        if msg["t"] == "done":
            self._process_done(msg)
            self._inflight -= 1
        elif msg["t"] == "submit":
            self._ingest_submit(msg)

    def _ingest_submit(self, msg: dict) -> None:
        """A frontend submitted token-ids: start a request, and remember the
        frontend's request id so emissions can be streamed back to it."""
        rid = self.submit({"tokens": list(msg["tokens"]), "max_tokens": msg["max_tokens"]})
        self._front_rid[rid] = msg["rid"]

    def run_until_idle(self) -> dict[int, list[Any]]:
        """Drive until no request has work left (used by the batch demo)."""
        while True:
            did = self.poll(timeout_ms=50)
            if not did and self._inflight == 0:
                break
        return self.results

    def _dispatch(self, batch) -> None:
        worker = self.node_to_worker[batch.node]
        # inputs: {rid: {name: [descriptor]}} resolved from uuids.
        inputs = {
            rid: {
                name: [self.desc[uuid] for (uuid, _dims, _dt) in refs]
                for name, refs in named.items()
            }
            for rid, named in batch.inputs.items()
        }
        self.mbox.send(
            worker,
            msgpack.packb(
                {"t": "execute", "batch_id": batch.batch_id, "node": batch.node,
                 "walk": batch.walk, "inputs": inputs}
            ),
        )

    def _process_done(self, msg: dict) -> None:
        """Route a worker's `done` message: mint output uuids, record their
        SHM location, and route via complete_batch."""
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
                    refs.append((uuid, d[3], d[4]))  # (uuid, dims, dtype)
                out[name] = refs
            outputs[rid] = out
        for event in self.runtime.complete_batch(msg["batch_id"], outputs):
            self._handle_event(event)

    def _handle_event(self, event: dict) -> None:
        rid = event["request_id"]
        if event["type"] == "emission":
            tensors = [self.shm.read(self.desc[uuid]) for (uuid, _d, _t) in event["tensors"]]
            value = self.model.postprocess(event["name"], event["modality"], tensors)
            self.results[rid].append(value)
            # Stream this emission to the frontend, if one is attached.
            if self.on_token is not None and rid in self._front_rid:
                self.on_token(self._front_rid[rid], value)
        elif event["type"] == "walk_done":
            persist = {
                name: [self.shm.read(self.desc[uuid]) for (uuid, _d, _t) in refs]
                for name, refs in event["persist"].items()
            }
            nxt = self.model.next_forward(
                rid, event["partition"], event["walk"], event["fwd_index"],
                persist, event["stream_done"],
            )
            if nxt is None:
                if self.runtime.finish_partition(rid, event["partition"]):
                    self.runtime.finish_request(rid)
                    self.finished.add(rid)  # signal the serving layer
                    front = self._front_rid.pop(rid, None)
                    if self.on_done is not None and front is not None:
                        self.on_done(front)
            else:
                self._start_walk(rid, nxt)

    # -- frontend serving --

    def serve_frontend(self) -> None:
        """Run forever as a streaming backend for the Rust axum frontend.

        The frontend sends `submit` messages (token-ids) on our inbox; we wire
        each per-token emission and the request-complete signal back to it over
        the same `Mailbox` mesh as msgpack. `submit`/`done`/`token` are the
        only messages that cross this seam — tensor bytes never do (those move
        worker<->conductor through SHM)."""
        self.on_token = lambda front_rid, value: self.mbox.send(
            "frontend",
            msgpack.packb({"t": "token", "rid": front_rid, "id": int(value)}),
        )
        self.on_done = lambda front_rid: self.mbox.send(
            "frontend", msgpack.packb({"t": "done", "rid": front_rid})
        )
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
        for worker in set(self.node_to_worker.values()):
            self.mbox.send(worker, msgpack.packb({"t": "shutdown"}))


def _spec_json(model) -> str:
    import json

    if (topo := model.partitions()) is not None:
        partition_specs, connection_specs = topo
        return json.dumps(
            {"walks": model.walks(), "partitions": partition_specs, "connections": connection_specs}
        )
    return json.dumps(model.walks())
