"""Decentralized multi-process runtime (mstar's in-worker scheduler model):
**self-driving workers** own their partitions and scheduler loops; a thin
**per-request coordinator** ingests requests and relays emissions. Control
messages ride the ZeroMQ mesh (`ZmqCommunicator`); tensors move through
shared memory (`ShmPool`).

Split of responsibilities:

- **DisaggWorker** — owns its partition(s) and a local `Driver` (scheduler +
  Rust `Runtime` + engine co-located), so the per-token decode loop pays NO
  central round-trip. Cross-partition tensors stream worker→worker
  peer-to-peer (SHM descriptor + a ZMQ `stream` notify), never through a
  scheduler. Under tensor parallelism the leader rank broadcasts each batch
  decision (`tp_sched`) and follower ranks replay identical batches in
  identical order, keeping the NCCL collectives aligned.
- **DisaggCoordinator** — per-request only: seeds each partition's worker
  once at ingest, relays emissions to the caller/frontend, and completes a
  request when all partitions report done. Nothing crosses it per step.

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

    The serving backend's egress seam (used by the decentralized
    `DisaggCoordinator`): text emissions (token ids) are
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

    def __init__(self, entity_id: str, segment_size: int | None = None,
                 max_segments: int | None = None, on_new_segment=None) -> None:
        self.entity_id = entity_id
        self.base = f"mstar_rs_{entity_id}"
        if segment_size is None:
            segment_size = int(
                os.environ.get("MSTAR_RS_SHM_SEGMENT_MB", "256")) << 20
        if max_segments is None:
            max_segments = int(
                os.environ.get("MSTAR_RS_SHM_MAX_SEGMENTS", "32"))
        self.arena = SegmentedShmArena.create(self.base, segment_size, max_segments)
        self._spill_warned = False
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

    def stage(self, t: torch.Tensor):
        """Copy a tensor's bytes into our arena; return its descriptor.

        At the segment cap, degrade to the INLINE form instead of failing:
        the bytes ride the control message itself (the multi-node wire
        shape, which every reader already handles). Slots here are
        request-scoped, so waiting for frees mid-request is futile — the
        overflow tier is the right immediate response; it costs one extra
        copy through the message and needs no reclaim."""
        t = t.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        try:
            seg, off = self.arena.reserve(max(nbytes, 1))
        except RuntimeError:
            total, free, largest = self.arena.stats()
            if not self._spill_warned:
                if free >= nbytes:
                    # Fragmentation signature: total free covers the need,
                    # but no contiguous block does.
                    _log = (f"fragmented: {free} free but largest block "
                            f"{largest}")
                else:
                    _log = f"full: {free} free of {total}"
                print(f"[{self.entity_id}] SHM arena {_log}; staging inline "
                      f"(MSTAR_RS_SHM_SEGMENT_MB/_MAX_SEGMENTS to grow)",
                      flush=True)
                self._spill_warned = True
            return self.inline(t)
        prev = len(self._views)
        self._sync_segments()  # growth may have added a segment
        if len(self._views) > prev:
            total, free, largest = self.arena.stats()
            _dbg(self.entity_id,
                 f"arena grew to {self.arena.num_segments} segments "
                 f"({total >> 20} MiB total, {free >> 20} MiB free, "
                 f"largest block {largest >> 20} MiB)")
        if nbytes:  # torch.frombuffer rejects an empty buffer
            mv = self._views[seg][off:off + nbytes]
            torch.frombuffer(mv, dtype=t.dtype).copy_(t.flatten())
        return [self.arena.segment_name(seg), off, nbytes, list(t.shape),
                _dtype_str(t.dtype)]

    def read(self, desc) -> torch.Tensor:
        if isinstance(desc, dict):  # inline (cross-node): bytes in the message
            dtype = _STR_TO_DTYPE[desc["dtype"]]
            if not desc["b"]:
                return torch.empty(desc["dims"], dtype=dtype)
            return torch.frombuffer(
                bytearray(desc["b"]), dtype=dtype).reshape(desc["dims"])
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

    @staticmethod
    def inline(t: torch.Tensor) -> dict:
        """An INLINE descriptor: the tensor's bytes ride the control message
        itself (msgpack bin) instead of a SHM slot — the cross-NODE form,
        where the consumer cannot map this process's /dev/shm. Same reader
        (`read`) handles both forms; inline slots need no `free`."""
        t = t.detach().to("cpu").contiguous()
        return {"dims": list(t.shape), "dtype": _dtype_str(t.dtype),
                "b": t.numpy().tobytes() if t.numel() else b""}

    def free_by_name(self, name: str, off: int) -> bool:
        """Release an offset in one of OUR segments (by descriptor name)."""
        i = self._own.get(name)
        return self.arena.free(i, off) if i is not None else False

    def free(self, desc: list) -> None:
        self.free_by_name(desc[0], desc[1])


def _bind_mailbox(my_id: str, socket_dir: str, endpoints: dict):
    """Bind this entity's inbox — an explicit endpoint (tcp, for entities
    that receive cross-node traffic) or the default ipc dir scheme — and
    register every explicitly-addressed peer."""
    if my_id in endpoints:
        # tcp inbox, ipc-dir fallback for same-node peers not in `endpoints`
        mbox = ZmqCommunicator.bind_endpoint(my_id, endpoints[my_id], socket_dir)
    else:
        mbox = ZmqCommunicator(my_id, socket_dir)
    for peer, ep in endpoints.items():
        if peer != my_id:
            mbox.register_peer(peer, ep)
    return mbox


# ---- worker -----------------------------------------------------------------


# ---- decentralized (in-worker scheduler) multi-process runtime --------------
#
# Each worker owns a partition, holds its OWN scheduler (a `Driver` scoped via
# local_partitions), and drives its own loop — so the per-step decode loop pays
# NO central round-trip (mstar's in-worker MicroScheduler model).
# Cross-partition tensors stream worker->worker peer-to-peer (SHM descriptor +
# a ZMQ `stream` notify), never through a scheduler. A thin coordinator only
# ingests requests (seeds each partition's worker once), relays emissions, and
# finishes a request when every partition reports done. See
# examples/verify_disagg.py.


class DisaggWorker:
    """Self-driving worker for one (or more) partition(s). Runs its own Driver
    loop; ships stream outputs to peer workers; injects peer chunks locally."""

    def __init__(self, worker_id, engine, local_partitions, partition_to_worker,
                 socket_dir, device: str = "cpu", io_leader: bool = True,
                 coordinator_id: str = "coordinator",
                 tp_nodes: list | None = None,
                 tp_followers: list | None = None,
                 tp_follow_nodes: list | None = None,
                 async_pipeline: bool = False,
                 node_map: dict | None = None,
                 endpoints: dict | None = None) -> None:
        self.worker_id = worker_id
        # Multi-node topology. node_map: entity id -> node name (None = all
        # co-located). Same-node tensor edges use SHM descriptors; cross-node
        # edges ship the bytes INLINE in the control message. endpoints:
        # entity id -> zmq endpoint (e.g. tcp://host:port) for every entity
        # that receives cross-node traffic; entities not listed bind the
        # default ipc scheme in socket_dir.
        self._node_map = node_map or {}
        self._endpoints = endpoints or {}
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
        # async pipeline (mstar's worker run loop): execute on a dedicated
        # thread; host work (messages, streams, outbox, routing) overlaps the
        # in-flight batch, and an eventfd wakes the loop the moment the
        # execute future completes (no poll-timeout latency).
        if async_pipeline and tp_follow_nodes:
            raise ValueError(
                "async_pipeline on a TP follower is not supported yet: the "
                "follow-replay path drives the engine from the message loop")
        self.async_pipeline = async_pipeline
        self.spec_hits = 0   # claimed speculations (pipeline instrumentation)
        # Fairness (mstar's max-consecutive-spec): break the claim chain
        # every N claims with a full scheduler scan, so newly-arrived
        # requests co-batch into the loop instead of starving behind an
        # unbroken speculation chain.
        self.spec_cap = 8
        self._spec_streak = 0
        self.mbox = _bind_mailbox(worker_id, socket_dir, self._endpoints)
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
        if self.async_pipeline:
            return self._run_async()
        busy = False
        while not self._stop:
            # Block only when the LAST tick had no work: a self-driving decode
            # loop must not pay the poll timeout between frames (that tax was
            # measured at ~50 ms/frame -> RTF 0.8 instead of ~0.1).
            raw = self.mbox.recv_timeout(0 if busy else 50)
            # Drain a BOUNDED number of queued messages, then always poll — an
            # unbounded drain would starve poll() when a peer floods this worker
            # with stream chunks faster than it can consume them (the consumer
            # would buffer forever and never drive its own partition).
            n = 0
            while raw is not None:
                self._handle(raw)
                if self._stop:
                    return
                n += 1
                if n >= 16:
                    break      # bound reached; do NOT fetch a frame we
                               # would drop — the next tick resumes the queue
                raw = self.mbox.try_recv()
            self._drain_tp_pending()
            busy = self._poll_contained()      # one in-process step, no round-trip
            if busy:
                _dbg(self.worker_id, "poll ran a batch")
            self._ship_outbox()

    def _run_async(self) -> None:
        """mstar's async worker pipeline, on this runtime: one batch in flight
        on a dedicated execute thread; the host thread overlaps it with
        message handling, stream shipping, and the PREVIOUS batch's routing.
        While batch N runs, its follow-up is SPECULATED (same loop body); the
        moment N completes, the speculation is claimed and N+1 submitted
        before any post-processing — mstar's submit-asap ordering. An eventfd
        registered with the mailbox (the Step-1 wakeup path) wakes the loop
        the instant the future finishes, instead of on the poll timeout.

        Thread discipline: the runtime, the tensor store, and all routing stay
        on THIS thread; the execute thread runs only `model.execute` (torch
        releases the GIL during CUDA waits, which is where the overlap comes
        from on GPU)."""
        from concurrent.futures import ThreadPoolExecutor

        def exec_thread_init() -> None:
            # A fresh thread's CUDA current-device is 0; engines on any other
            # card would launch kernels into the wrong context (observed as
            # cudaErrorIllegalInstruction on the audio worker, cuda:2).
            if str(self.device).startswith("cuda"):
                torch.cuda.set_device(self.device)

        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mstar-exec-{self.worker_id}",
            initializer=exec_thread_init)
        efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.mbox.register_wakeup_fd(efd)
        pending = None   # (batch, future)
        ticket = None    # speculation for the pending batch

        def submit(prepared) -> None:
            nonlocal pending, ticket
            batch, inputs = prepared
            future = executor.submit(
                self.engine_execute, batch.node, batch.walk, inputs, batch.kv)
            future.add_done_callback(lambda _f: os.eventfd_write(efd, 1))
            pending = (batch, future)
            ticket = self.driver.speculate(batch)

        try:
            while not self._stop:
                # Host preamble — overlaps the in-flight batch: drain a
                # bounded number of messages (or block briefly when idle).
                kind, raw = self.mbox.recv_or_wake(50 if pending is None else 0)
                if kind == "wake":
                    try:
                        os.eventfd_read(efd)
                    except BlockingIOError:
                        pass
                n = 0
                while raw is not None:
                    self._handle(raw)
                    if self._stop:
                        return
                    n += 1
                    if n >= 16:
                        break  # bound reached; never fetch-then-drop
                    raw = self.mbox.try_recv()
                self._drain_tp_pending()

                if pending is None:
                    prepared = self.driver.begin_batch()
                    if prepared is not None:
                        submit(prepared)
                    continue

                batch, future = pending
                if not future.done():
                    # Block until the future's eventfd or a message wakes us.
                    kind, raw = self.mbox.recv_or_wake(50)
                    if kind == "wake":
                        try:
                            os.eventfd_read(efd)
                        except BlockingIOError:
                            pass
                    elif kind == "msg" and raw is not None:
                        self._handle(raw)
                        if self._stop:
                            return
                        self._drain_tp_pending()
                    continue

                try:
                    outputs = future.result()
                except Exception as e:  # noqa: BLE001 — contain: fail this
                    # batch's requests, keep the worker serving (engine errors
                    # only; anything else still crashes loudly via _run's
                    # handler on the next unguarded raise)
                    pending = None
                    ticket = None
                    self._fail_requests(list(batch.inputs.keys()), e)
                    continue
                pending = None
                self.driver.finish_batch(batch, outputs)
                # Submit the follow-up ASAP (claimed speculation skips the
                # scheduler scan; fall back to a normal schedule), THEN do
                # post-processing — outbox shipping overlaps GPU(N+1).
                prepared = None
                if ticket is not None and self._spec_streak < self.spec_cap:
                    prepared = self.driver.claim_speculative(ticket)
                ticket = None
                if prepared is not None:
                    self.spec_hits += 1
                    self._spec_streak += 1
                else:
                    prepared = self.driver.begin_batch()
                    self._spec_streak = 0
                if prepared is not None:
                    submit(prepared)
                self._ship_outbox()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            os.close(efd)

    def engine_execute(self, node, walk, inputs, kv):
        """Runs on the execute thread — model compute only, no runtime access."""
        return self.driver.model.execute(node, walk, inputs, kv=kv)

    def _broadcast_batch(self, node: str, walk: str, rids: list) -> None:
        """TP leader: ship this batch decision to follower ranks BEFORE
        executing (mstar sends ScheduleTPNode before GPU submission), so the
        followers replay identical batches in identical order."""
        if node not in self.tp_nodes:
            return
        gids = [self._l2g[r] for r in rids]
        self._bseq = getattr(self, "_bseq", 0) + 1
        msg = msgpack.packb({"t": "tp_sched", "node": node, "walk": walk,
                             "gids": gids, "seq": self._bseq})
        _dbg(self.worker_id, f"broadcast #{self._bseq} {node} gids={gids}")
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
        same deterministic state. A queued decision can also reference OTHER
        requests whose seeds ride the coordinator connection and may still be
        in this worker's mailbox (the forwarded finish travels on the leader
        connection and can overtake them), so the wait must keep consuming
        the mailbox — anything but a shutdown, which is deferred until the
        flush completes so it cannot interrupt it."""
        def pending() -> bool:
            return (any(gid in gs for _n, _w, gs in self._tp_pending)
                    or any(local in rs for _n, _w, rs in self.driver._tp_follow))

        deadline = time.time() + timeout_s
        deferred = []
        while pending():
            self._drain_tp_pending()
            if self.driver.poll():
                continue
            raw = self.mbox.recv_timeout(1)
            if raw is not None:
                if msgpack.unpackb(raw, raw=False,
                                   strict_map_key=False).get("t") == "shutdown":
                    deferred.append(raw)
                else:
                    self._handle(raw)
                continue
            if time.time() > deadline:
                # Never wedge the replay queue on a decision whose requests
                # can no longer all map: purge what references this gid (the
                # replayed sequence has already diverged; keep draining).
                _dbg(self.worker_id, f"flush_follows timeout gid={gid}; purging")
                self._tp_pending = [
                    e for e in self._tp_pending if gid not in e[2]]
                self.driver._tp_follow = type(self.driver._tp_follow)(
                    e for e in self.driver._tp_follow if local not in e[2])
                break
        for raw in deferred:
            self._handle(raw)

    def _same_node(self, peer: str) -> bool:
        if not self._node_map:
            return True
        return self._node_map.get(self.worker_id) == self._node_map.get(peer)

    def _stage_for(self, t: torch.Tensor, peer: str):
        """SHM descriptor for a same-node peer; inline bytes cross-node."""
        return self.shm.stage(t) if self._same_node(peer) else ShmPool.inline(t)

    def _poll_contained(self) -> bool:
        """driver.poll with per-batch error containment (the legacy conductor
        path's semantics, kept for serving parity): an engine exception fails
        the batch's requests — reported to the coordinator so the frontend
        sees an error — and the worker keeps serving. Exceptions without
        batch context (runtime bugs) still crash the worker loudly."""
        try:
            return self.driver.poll()
        except Exception as e:  # noqa: BLE001
            rids = getattr(e, "batch_request_ids", None)
            if rids is None:
                raise
            self._fail_requests(rids, e)
            return True

    def _fail_requests(self, rids: list, exc: Exception) -> None:
        import traceback
        traceback.print_exc()
        gids = [self._l2g[r] for r in rids if r in self._l2g]
        for rid in list(rids):
            gid = self._l2g.get(rid)
            if gid is not None:
                self._g2l.pop(gid, None)
            self._l2g.pop(rid, None)
            self.driver._front_rid.pop(rid, None)
            self.driver.runtime.finish_request(rid)
            self.driver.store.free_request(rid)
            self.driver.model.release_request(rid)
        if self.io_leader and gids:
            self.mbox.send(self.coordinator_id, msgpack.packb(
                {"t": "req_error", "worker": self.worker_id,
                 "gids": gids, "msg": repr(exc)}))

    def _handle(self, raw: bytes) -> None:
        msg = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        t = msg["t"]
        if t == "shutdown":
            # TP leader: forward on the ordered tp_sched connection so a
            # follower never sees shutdown overtake a queued batch decision
            # (the coordinator shuts down only leaders).
            for follower in self.tp_followers:
                self.mbox.send(follower, raw)
            # Follower: drain queued leader decisions BEFORE stopping — the
            # ordered connection guarantees they arrived first, and dropping
            # them would truncate the replayed sequence (the lockstep
            # invariant NCCL depends on; caught by
            # verify_tp_lockstep_concurrent once the poll-tax fix let the
            # leader finish far ahead of the replay).
            deadline = time.time() + 5.0
            quiet = time.time() + 0.2
            while time.time() < deadline:
                # keep consuming the mailbox: replay decisions (and their
                # inputs) can still be in flight when shutdown is handled —
                # exit only after the queues are empty AND the mailbox has
                # been quiet for a grace window.
                queued = self.mbox.try_recv()
                if queued is not None:
                    self._handle(queued)
                    quiet = time.time() + 0.2
                self._drain_tp_pending()
                if self.driver.poll():
                    quiet = time.time() + 0.2
                elif queued is None:
                    if not self._tp_pending and not self.driver._tp_follow \
                            and time.time() >= quiet:
                        break
                    time.sleep(0.001)
            _dbg(self.worker_id, f"shutdown: leftover pending={self._tp_pending} "
                 f"follow={list(self.driver._tp_follow)}")
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
            # The staged seed bytes are copied out above — release the
            # coordinator's arena slots (same protocol as worker-to-worker
            # frees; without this every request leaks its seeds there).
            items = [(d[0], d[1])
                     for _n, _nm, descs in msg["inputs"] for d in descs
                     if isinstance(d, list)]
            if items:
                self.mbox.send(self.coordinator_id, msgpack.packb(
                    {"t": "free", "items": items}))
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
            _dbg(self.worker_id, f"tp_sched #{msg.get('seq')} {msg['node']} gids={msg['gids']}")
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
            consumer = self.partition_to_worker[to]
            _dbg(self.worker_id, f"ship {frm}->{to} gid={gid} -> {consumer}")
            descs = [self._stage_for(t, consumer) for t in tensors]
            self.mbox.send(
                consumer,
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
                 "desc": self._stage_for(value, self.coordinator_id)}))
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
                 my_id: str = "coordinator",
                 node_map: dict | None = None,
                 endpoints: dict | None = None) -> None:
        self.policy = policy
        # Multi-node topology (same contract as DisaggWorker): node_map maps
        # entity ids to node names; seeds to same-node ranks ride SHM
        # descriptors, seeds to cross-node ranks ship their bytes inline.
        # endpoints lists the tcp-bound entities (everyone else = ipc).
        self._node_map = node_map or {}
        self._endpoints = endpoints or {}
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
        self.my_id = my_id
        self.mbox = _bind_mailbox(my_id, socket_dir, self._endpoints)
        self.shm = ShmPool(my_id)
        self._gid = 0
        self.results: dict[int, list[Any]] = {}
        self._pending: dict[int, set] = {}   # gid -> partitions not yet done
        self.finished: set[int] = set()
        self.errors: dict[int, str] = {}     # gid -> failure detail
        # Workers that reported a crash: their requests were failed, and new
        # submits that would seed them are refused up front.
        self._dead_workers: set[str] = set()
        # staged seed slot -> ranks yet to read it (TP seeds are shared)
        self._stage_refs: dict[tuple, int] = {}
        # Frontend serving seam (submit in, chunk/done out).
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
            ranks = self.partition_to_workers[self.walk_partition[walk]]
            local = [w for w in ranks if self._same_node(w)]
            remote = [w for w in ranks if not self._same_node(w)]
            if local:
                # Same-node ranks read the SAME staged SHM slots (TP group
                # runs in lockstep); freed only after ALL of them report
                # their read (refcounted below).
                seeded = [[node, name, [self.shm.stage(t) for t in tensors]]
                          for node, name, tensors in inputs]
                for _node, _name, descs in seeded:
                    for d in descs:
                        if isinstance(d, list):   # inline spills own no slot
                            self._stage_refs[(d[0], d[1])] = len(local)
                msg = msgpack.packb(
                    {"t": "seed", "gid": gid, "walk": walk, "inputs": seeded,
                     "kv": list(kv), "mk": mk})
                for worker in local:
                    self.mbox.send(worker, msg)
            if remote:
                # Cross-node ranks cannot map this process's /dev/shm: the
                # seed bytes ride the message inline (no slot, no free).
                inlined = [[node, name, [ShmPool.inline(t) for t in tensors]]
                           for node, name, tensors in inputs]
                msg = msgpack.packb(
                    {"t": "seed", "gid": gid, "walk": walk, "inputs": inlined,
                     "kv": list(kv), "mk": mk})
                for worker in remote:
                    self.mbox.send(worker, msg)
        return gid

    def _same_node(self, peer: str) -> bool:
        if not self._node_map:
            return True
        return self._node_map.get(self.my_id) == self._node_map.get(peer)

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
                    self._pending.pop(gid)  # bookkeeping done for this gid
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
                # A frontend submitted a flattened request (the multimodal
                # bridge wire shape). Ingest failures fail ONE
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
            elif t == "free":
                # a rank consumed staged seed tensors: release each slot once
                # ALL ranks of its partition have reported their read
                for name, off in msg["items"]:
                    left = self._stage_refs.get((name, off), 1) - 1
                    if left <= 0:
                        self._stage_refs.pop((name, off), None)
                        self.shm.free_by_name(name, off)
                    else:
                        self._stage_refs[(name, off)] = left
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
            elif t == "req_error":
                # A worker contained an engine error: fail ONLY these requests
                # (frontend error / errors dict); the worker keeps serving.
                detail = msg.get("msg", "engine error")
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
        """Run forever as a streaming backend for the Rust axum frontend
        (`submit` in; `chunk`/`done` out via `_FrontendStream`) — per-request
        only: workers self-schedule their decode loops, so nothing crosses this
        loop per step; it just ingests requests and relays emissions."""
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
