"""The driver loop: the thin Python shell around the Rust runtime.

This replaces (for the in-process case) mstar's conductor round-trips: the
Rust core picks batches and routes outputs; the driver executes nodes with
torch and answers WalkDone events by consulting the model's policy.
"""

from __future__ import annotations

import json
from typing import Any

from mstar_rs._core import Runtime

from .model import Model, NextWalk
from .store import TensorStore


class Driver:
    def __init__(self, model: Model, max_batch_size: int = 8) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        # A model spec with partitions is passed as {"walks", "partitions",
        # "connections"}; a bare model just as the walk dict.
        if (topo := model.partitions()) is not None:
            partition_specs, connection_specs = topo
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
        self.store = TensorStore(self.runtime)
        # request_id -> list of postprocessed emissions
        self.results: dict[int, list[Any]] = {}

    def submit(self, request: dict[str, Any]) -> int:
        request_id = self.runtime.add_request()
        request = dict(request, request_id=request_id)
        for nxt in self.model.initial_walks(request):
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

    def run_until_idle(self) -> dict[int, list[Any]]:
        """Drive the runtime until no request has schedulable work left."""
        while (batch := self.runtime.next_batch(self.max_batch_size)) is not None:
            inputs = {
                rid: {
                    name: self.store.get_all(refs)
                    for name, refs in named.items()
                }
                for rid, named in batch.inputs.items()
            }
            outputs = self.model.execute(batch.node, batch.walk, inputs, kv=batch.kv)
            # check_stop -> STOP_LOOPS: must land before complete_batch so the
            # loop terminates on this iteration rather than advancing.
            for rid, loop_name in self.model.loops_to_finish():
                self.runtime.signal_loop_finish(rid, loop_name)
            out_refs = {
                rid: {
                    name: [self.store.put(t, rid) for t in tensors]
                    for name, tensors in named.items()
                }
                for rid, named in outputs.items()
            }
            for event in self.runtime.complete_batch(batch.batch_id, out_refs):
                self._handle_event(event)
        return self.results

    def _handle_event(self, event: dict[str, Any]) -> None:
        rid = event["request_id"]
        if event["type"] == "emission":
            tensors = self.store.get_all(event["tensors"])
            self.results[rid].append(
                self.model.postprocess(event["name"], event["modality"], tensors)
            )
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
                # Partition finished; the request completes only when every
                # partition is done (finish_partition returns all-done).
                if self.runtime.finish_partition(rid, event["partition"]):
                    self.runtime.finish_request(rid)
                    self.store.free_request(rid)
            else:
                self._start_walk(rid, nxt)
        elif event["type"] == "free":
            # Per-tensor reclaim: the runtime says these tensors are now
            # unreachable (consumed + not persisted/buffered). Emitted after
            # this batch's emission/walk_done events, so reads already happened.
            self.store.free(event["uuids"])
        else:
            raise RuntimeError(f"unknown event type: {event['type']}")
