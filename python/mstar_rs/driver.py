"""The driver loop: the thin Python shell around the Rust runtime.

This replaces (for the in-process case) mstar's conductor round-trips: the
Rust core picks batches and routes outputs; the driver executes nodes with
torch and answers WalkDone events by consulting the model's policy.

The scheduler (the Rust `Runtime`) and the model compute live in the SAME
process here, so the per-step loop — `next_batch` -> `execute` -> `complete_
batch` -> `next_forward` — is all in-process calls with NO IPC, msgpack, or
SHM round-trip per step. This is the co-located / single-node serving path
(mirrors mstar's in-worker `MicroScheduler`, which each worker drives itself);
the multi-process `Conductor` in `dist.py` — which dispatches a batch to a
worker over ZeroMQ *every step* — is only for genuinely-distributed
deployments (separate GPUs/partitions, multi-node), and pays a per-step round
trip that this path avoids. `poll`/`finished`/`errors`/`shutdown_workers`
mirror the Conductor's surface so `server.ServingEngine` can drive either one.
"""

from __future__ import annotations

import json
from typing import Any

from mstar_rs._core import Runtime

from .model import Model, NextWalk
from .store import TensorStore


class Driver:
    def __init__(self, model: Model, max_batch_size: int = 8,
                 local_partitions: list[str] | None = None) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        # Decentralized / per-worker mode: this driver OWNS only these
        # partitions (schedules + executes them). Stream outputs to other
        # partitions surface as `stream_out` events collected in `self.outbox`
        # for the owning worker to ship; incoming chunks arrive via `inject`.
        # None = own all (single-process). This is the per-partition in-worker
        # scheduler (mstar's MicroScheduler model) — each worker drives its own
        # loop with peer-to-peer streaming, no per-step conductor round-trip.
        self._local_partitions = local_partitions
        # (from_partition, edge, target_partition, [torch tensors], request_id)
        self.outbox: list[tuple] = []
        # A model spec with partitions is passed as {"walks", "partitions",
        # "connections"}; a bare model just as the walk dict.
        self._walk_partition: dict[str, str] = {}
        if (topo := model.partitions()) is not None:
            partition_specs, connection_specs = topo
            for p in partition_specs:
                for w in p["walks"]:
                    self._walk_partition[w] = p["name"]
            spec = {
                "walks": model.walks(),
                "partitions": partition_specs,
                "connections": connection_specs,
            }
            self.runtime = Runtime(json.dumps(spec))
        else:
            self.runtime = Runtime(json.dumps(model.walks()))
        if (kv := model.kv_config()) is not None:
            configs, node_labels = kv
            self.runtime.configure_kv(configs, node_labels)
        if unbatchable := model.unbatchable():
            self.runtime.configure_unbatchable([tuple(p) for p in unbatchable])
        if local_partitions is not None:
            self.runtime.set_local_partitions(local_partitions)
        self.store = TensorStore(self.runtime)
        # request_id -> list of postprocessed emissions
        self.results: dict[int, list[Any]] = {}
        # Serving surface (parity with dist.Conductor, so ServingEngine can
        # drive an in-process Driver with no per-step IPC).
        self.finished: set[int] = set()   # request ids whose request completed
        self.errors: dict[int, str] = {}  # request id -> error message
        self.on_token = None              # callable(front_rid, value) — stream hook
        self.on_done = None               # callable(front_rid)
        # callable(rid, partition) — a locally-owned partition finished. In the
        # decentralized/multi-worker case each worker owns one partition, so
        # finish_partition never returns all-done here; the coordinator collects
        # these to decide request completion across workers.
        self.on_partition_done = None
        self._front_rid: dict[int, int] = {}  # runtime rid -> frontend rid

    def new_request(self) -> int:
        """Create a request slot without seeding (the coordinator drives ingest
        in the decentralized path). Returns the local runtime request id."""
        rid = self.runtime.add_request()
        self.results[rid] = []
        return rid

    def start(self, request_id: int, walk: str, inputs, kv_appends=None,
              kv_scratch=None) -> None:
        """Seed one walk for an existing request (coordinator-driven ingest)."""
        nxt = (walk, inputs) if kv_appends is None else (walk, inputs, kv_appends, kv_scratch)
        self._start_walk(request_id, nxt)

    def _owns(self, walk: str) -> bool:
        """Whether this driver schedules `walk` (its partition is local)."""
        return (self._local_partitions is None
                or self._walk_partition.get(walk) in self._local_partitions)

    def submit(self, request: dict[str, Any]) -> int:
        request_id = self.runtime.add_request()
        request = dict(request, request_id=request_id)
        # Seed only the initial walks for partitions this driver owns; peers
        # seed their own (decentralized ingest). Single-process owns all.
        for nxt in self.model.initial_walks(request):
            if self._owns(nxt[0]):
                self._start_walk(request_id, nxt)
        self.results[request_id] = []
        return request_id

    def _start_walk(self, request_id: int, next_walk: NextWalk) -> None:
        walk, inputs, *rest = next_walk
        kv_appends = rest[0] if len(rest) > 0 else None
        kv_scratch = rest[1] if len(rest) > 1 else None
        seeded = [
            (node, name, [self.store.put(t, request_id) for t in tensors])
            for node, name, tensors in inputs
        ]
        self.runtime.start_walk(request_id, walk, seeded, kv_appends, kv_scratch)

    def poll(self, timeout_ms: int = 0) -> bool:
        """Run the next ready batch IN-PROCESS (no IPC) and route its outputs.
        Returns True if a batch ran. One batch per call so a serving loop can
        ingest new requests between steps (continuous batching emerges from the
        runtime grouping every ready request's same-(node,walk) work into one
        batch). `timeout_ms` is accepted for Conductor-API parity and ignored —
        in-process there is nothing to block on."""
        batch = self.runtime.next_batch(self.max_batch_size)
        if batch is None:
            return False
        inputs = {
            rid: {name: self.store.get_all(refs) for name, refs in named.items()}
            for rid, named in batch.inputs.items()
        }
        outputs = self.model.execute(batch.node, batch.walk, inputs, kv=batch.kv)
        # check_stop -> STOP_LOOPS: must land before complete_batch so the
        # loop terminates on this iteration rather than advancing.
        for rid, loop_name in self.model.loops_to_finish():
            self.runtime.signal_loop_finish(rid, loop_name)
        out_refs = {
            rid: {name: [self.store.put(t, rid) for t in tensors]
                  for name, tensors in named.items()}
            for rid, named in outputs.items()
        }
        for event in self.runtime.complete_batch(batch.batch_id, out_refs):
            self._handle_event(event)
        return True

    def run_until_idle(self) -> dict[int, list[Any]]:
        """Drive the runtime until no request has schedulable work left."""
        while self.poll():
            pass
        return self.results

    def shutdown_workers(self) -> None:
        pass  # in-process: no workers to shut down

    def _handle_event(self, event: dict[str, Any]) -> None:
        rid = event["request_id"]
        if event["type"] == "emission":
            tensors = self.store.get_all(event["tensors"])
            value = self.model.postprocess(event["name"], event["modality"], tensors)
            self.results[rid].append(value)
            if self.on_token is not None and rid in self._front_rid:
                self.on_token(self._front_rid[rid], value)  # stream, no per-step IPC
        elif event["type"] == "walk_done":
            persist = {
                name: self.store.get_all(refs)
                for name, refs in event["persist"].items()
            }
            nxt = self.model.next_forward(
                rid,
                event["partition"],
                event["walk"],
                event["fwd_index"],
                persist,
                event["stream_done"],
            )
            if nxt is None:
                # Partition finished. A local partition-done hook lets the
                # decentralized coordinator track completion across workers
                # (where finish_partition never sees the peers' partitions).
                if self.on_partition_done is not None:
                    self.on_partition_done(rid, event["partition"])
                # The request completes only when every partition is done here
                # (all-owned / single-process); returns all-done then.
                if self.runtime.finish_partition(rid, event["partition"]):
                    self.runtime.finish_request(rid)
                    self.store.free_request(rid)
                    self.finished.add(rid)   # ServingEngine consumes this
                    if self.on_done is not None and rid in self._front_rid:
                        self.on_done(self._front_rid.pop(rid))
            else:
                self._start_walk(rid, nxt)
        elif event["type"] == "stream_out":
            # A stream chunk for a partition another worker owns: read the
            # tensors and stage them for shipping to that worker (which calls
            # `inject`). No conductor per-step round-trip — this is peer-to-peer.
            tensors = self.store.get_all(event["tensors"])
            self.outbox.append((
                event["from_partition"], event["edge"],
                event["target_partition"], tensors, rid,
            ))
        elif event["type"] == "free":
            # Per-tensor reclaim: the runtime says these tensors are now
            # unreachable (consumed + not persisted/buffered). Emitted after
            # this batch's emission/walk_done events, so reads already happened.
            self.store.free(event["uuids"])
        else:
            raise RuntimeError(f"unknown event type: {event['type']}")

    def inject(self, request_id: int, from_partition: str, edge: str,
               target_partition: str, tensors: list) -> None:
        """Consumer side: a peer worker shipped a stream chunk for a partition
        this driver owns. Stage the tensors locally and hand them to the
        runtime, which delivers them to the waiting consumer walk on the next
        poll. Mirrors mstar's peer-to-peer stream delivery into a worker's
        local ready-queue."""
        refs = [self.store.put(t, request_id) for t in tensors]
        self.runtime.inject_stream_chunk(request_id, from_partition, edge,
                                         target_partition, refs)

    def signal_stream_done(self, request_id: int, from_partition: str, edge: str,
                           target_partition: str) -> None:
        """Consumer side: a peer worker's producer partition finished — mark it
        done on the local connection buffer so a continue_after_done stream
        keeps the consumer's own loop alive on empty chunks."""
        self.runtime.signal_stream_done(request_id, from_partition, edge, target_partition)

    def outgoing_cross_worker(self, partition: str):
        """(edge, target_partition) for this partition's connections whose
        consumer is a different worker."""
        return self.runtime.outgoing_cross_worker(partition)
