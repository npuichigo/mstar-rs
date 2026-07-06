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
        mv = memoryview(self.arena)[off:off + nbytes]
        torch.frombuffer(mv, dtype=t.dtype).copy_(t.flatten())
        return [self.name, off, nbytes, list(t.shape), _DTYPE_TO_STR[t.dtype]]

    def read(self, desc: list) -> torch.Tensor:
        name, off, nbytes, dims, dtype_str = desc
        arena = self._peers.get(name)
        if arena is None:
            arena = ShmArena.open(name)
            self._peers[name] = arena
        mv = memoryview(arena)[off:off + nbytes]
        flat = torch.frombuffer(mv, dtype=_STR_TO_DTYPE[dtype_str]).clone()
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
        batching): dispatch every ready batch, then drain the completions
        that have arrived. Returns True if anything progressed. This is the
        serving primitive — the HTTP layer runs it in a background thread
        while requests stream in and out concurrently."""
        did = False
        while (batch := self.runtime.next_batch(self.max_batch_size)) is not None:
            self._dispatch(batch)
            self._inflight += 1
            did = True
        if self._inflight:
            raw = self.mbox.recv_timeout(timeout_ms)
            while raw is not None:
                self._process_done(raw)
                self._inflight -= 1
                did = True
                raw = self.mbox.try_recv()
        return did

    def run_until_idle(self) -> dict[int, list[Any]]:
        """Drive until no request has work left (used by the batch demo)."""
        while self.poll(timeout_ms=2000) or self._inflight:
            pass
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

    def _process_done(self, raw: bytes) -> None:
        """Route a worker's `done` message: mint output uuids, record their
        SHM location, and route via complete_batch."""
        msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        assert msg["t"] == "done"
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
            self.results[rid].append(
                self.model.postprocess(event["name"], event["modality"], tensors)
            )
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
            else:
                self._start_walk(rid, nxt)

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
