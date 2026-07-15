"""Production launcher: the Rust (axum) frontend + Python conductor + worker
processes, started as one command. This is the serving entrypoint that replaces
a Python FastAPI server (mstar's api_server): the axum binary is the HTTP
surface, the conductor drives the model over the ``mstar-comm`` bridge, and the
workers hold the weights.

Layout (see the module boundaries in server.py):

    HTTP ─► mstar-server (Rust) ─ZMQ/msgpack─► conductor (Python) ─ZMQ+SHM─► workers

The conductor's ``serve_frontend`` loop speaks the multimodal bridge protocol
(submit {text, file_paths, modalities, model_kwargs} → ResultChunk stream), so
audio/image/text-in and text/speech-out all flow through this one stack with no
Python HTTP server in the request path.

Run (GPU assignment via CUDA_VISIBLE_DEVICES, as in the verify examples):

    CUDA_VISIBLE_DEVICES=1,6 python -m mstar_rs.launch --model qwen3_omni --tp 2 --port 8000

Ctrl-C shuts the whole stack down cleanly.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_DEFAULT_BINARY = REPO / "target" / "release" / "mstar-server"


# --- worker entrypoints (module-level for spawn picklability) --------------

def _thinker_worker(worker_id: str, tp_rank: int, tp_world: int, socket_dir: str,
                    tp_port: int, max_new: int, audio_output: bool = False) -> None:
    """One TP rank of the Qwen3-Omni Thinker. audio_output=True also streams the
    selected hidden states to the Talker (speech-out stack)."""
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models import Qwen3OmniThinkerEngine

        engine = Qwen3OmniThinkerEngine(
            tp_rank=tp_rank, tp_world=tp_world, max_output_tokens=max_new,
            greedy=True, tp_port=tp_port, audio_output=audio_output,
        )
        Worker(worker_id, engine, socket_dir, device=str(engine.device)).run()
    except Exception:
        import traceback
        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _audio_worker(worker_id: str, socket_dir: str, device: str,
                  voice: str, max_new: int) -> None:
    """Talker + Code2Wav on one GPU (the speech-out stack's rank)."""
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models import Qwen3OmniAudioEngine

        engine = Qwen3OmniAudioEngine(device=device, voice=voice, max_output_tokens=max_new)
        Worker(worker_id, engine, socket_dir, device=device).run()
    except Exception:
        import traceback
        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def _disagg_speech_worker(kind: str, local_partitions: list, ship_to: dict,
                          socket_dir: str, tp_world: int, tp_port: int,
                          max_new: int, voice: str,
                          tp_followers: list | None = None) -> None:
    """A self-driving (decentralized) worker for the speech stack: owns its
    partition(s), runs its own scheduler loop, streams peer-to-peer — no
    per-step conductor round-trip (mstar's in-worker scheduler model). The
    worker composes the full audio graph (policy) with its engine so its local
    Driver can schedule and continue its own walks."""
    try:
        from mstar_rs import Model
        from mstar_rs.dist import DisaggWorker
        from mstar_rs.models import (
            Qwen3OmniAudioEngine,
            Qwen3OmniAudioPolicy,
            Qwen3OmniThinkerEngine,
        )

        policy = Qwen3OmniAudioPolicy(max_output_tokens=max_new, voice=voice)
        if kind.startswith("thinker"):
            engine = Qwen3OmniThinkerEngine(
                tp_rank=int(kind[-1]), tp_world=tp_world, max_output_tokens=max_new,
                greedy=True, tp_port=tp_port, audio_output=True,
            )
        else:
            engine = Qwen3OmniAudioEngine(device=f"cuda:{tp_world}", voice=voice,
                                          max_output_tokens=max_new)

        class _WorkerModel(Model):
            def __init__(self):
                self.device = engine.device
            def walks(self): return policy.walks()
            def partitions(self): return policy.partitions()
            def kv_config(self): return policy.kv_config()
            def unbatchable(self): return policy.unbatchable()
            def initial_walks(self, r): return policy.initial_walks(r)
            def next_forward(self, *a): return policy.next_forward(*a)
            def postprocess(self, *a): return policy.postprocess(*a)
            def execute(self, *a, **k): return engine.execute(*a, **k)
            def loops_to_finish(self): return engine.loops_to_finish()
            def register_request(self, rid, mk): engine.register_request(rid, mk)
            def release_request(self, rid): engine.release_request(rid)

        model = _WorkerModel()
        # Only rank 0 of the TP Thinker is the I/O leader; followers compute
        # for the NCCL all-reduces but suppress outward I/O. Under concurrency
        # the leader also broadcasts each Thinker batch decision (tp_sched) so
        # every rank runs identical batches in identical order — follower ranks
        # never self-initiate Thinker batches (tp_follow_nodes).
        io_leader = kind == "thinker0" or not kind.startswith("thinker")
        is_tp_rank = kind.startswith("thinker")
        # MSTAR_ASYNC_PIPELINE=1: overlap host work with the in-flight batch
        # (mstar's async worker loop). TP followers stay sync — their replay
        # path drives the engine from the message loop.
        use_async = (os.environ.get("MSTAR_ASYNC_PIPELINE") == "1"
                     and (io_leader or not is_tp_rank))
        DisaggWorker(
            kind, model, local_partitions, ship_to, socket_dir,
            device=str(model.device), io_leader=io_leader,
            coordinator_id="conductor",
            tp_nodes=["Thinker"] if (is_tp_rank and io_leader) else None,
            tp_followers=tp_followers if (is_tp_rank and io_leader) else None,
            tp_follow_nodes=["Thinker"] if (is_tp_rank and not io_leader) else None,
            async_pipeline=use_async,
        ).run()
    except Exception:
        import traceback
        with open(f"{socket_dir}/{kind}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


# --- generic supervisor ----------------------------------------------------

def serve(*, model_name: str, policy_factory=None, worker_specs, node_to_workers=None,
          port: int = 8000, binary: Path | None = None, frontend: bool = True,
          socket_dir: str | None = None, load_timeout: float = 300.0,
          backend_factory=None) -> int:
    """Spawn workers, run the conductor's frontend loop, and (unless
    ``frontend=False``) supervise the axum binary. Blocks until interrupted.

    worker_specs: list of (worker_id, target, args) for the model workers.
    policy_factory: () -> weightless ModelPolicy for the conductor.
    node_to_workers: graph-node -> worker-id (or list of TP rank ids).
    backend_factory: optional () -> serving backend (an object with
        serve_frontend/stop/shutdown_workers). Overrides the default
        centralized Conductor — the decentralized path passes a
        DisaggCoordinator factory here.

    ``frontend=False`` (the ``--no-frontend`` deploy) starts only the Python
    backend — workers + the conductor's bridge loop bound to ``socket_dir`` — and
    prints the exact command to run ``mstar-server`` as a separate service (or on
    another host) against that same socket dir. This decouples frontend and
    backend lifecycles (restart/scale the Rust frontend without bouncing the GPU
    workers), matching mstar's separate-process api_server and vLLM's standalone
    mode.
    """
    binary = Path(binary) if binary else _DEFAULT_BINARY
    if frontend and not binary.exists():
        print(f"missing frontend binary {binary}; run: cargo build --release -p mstar-server",
              file=sys.stderr)
        return 2

    ctx = mp.get_context("spawn")
    socket_dir = socket_dir or tempfile.mkdtemp(prefix="mstar_rs_serve_")
    worker_ids = [wid for (wid, _t, _a) in worker_specs]

    procs = []
    for wid, target, args in worker_specs:
        p = ctx.Process(target=target, args=args, daemon=True)
        p.start()
        procs.append(p)

    print(f"waiting for {len(procs)} worker(s) to init + load weights...", flush=True)
    deadline = time.time() + load_timeout
    need = {f"{socket_dir}/{w}.ipc" for w in worker_ids}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{w}.err").exists() for w in worker_ids):
            break
        time.sleep(0.5)
    bound = [w for w in worker_ids if Path(f"{socket_dir}/{w}.ipc").exists()]
    if len(bound) != len(worker_ids):
        for wid in worker_ids:
            errf = Path(f"{socket_dir}/{wid}.err")
            if errf.exists():
                print(f"--- {wid} failed ---\n{errf.read_text()}", file=sys.stderr)
        print(f"only {bound} of {worker_ids} bound; aborting", file=sys.stderr)
        return 1
    print(f"workers bound: {bound}", flush=True)

    if backend_factory is not None:
        cond = backend_factory()
    else:
        from mstar_rs.dist import Conductor

        cond = Conductor(policy_factory(), node_to_worker=node_to_workers,
                         socket_dir=socket_dir)
    serve_thread = threading.Thread(target=cond.serve_frontend, daemon=True)
    serve_thread.start()
    print("conductor serving (frontend bridge up)", flush=True)

    server = None
    if frontend:
        server = subprocess.Popen([str(binary), model_name, str(port), socket_dir])
        print(f"mstar-server (Rust) on http://127.0.0.1:{port} for model {model_name!r}", flush=True)
    else:
        print("backend only (--no-frontend). Start the Rust frontend separately with:\n"
              f"    {binary} {model_name} {port} {socket_dir}\n"
              "(same host or another — it only needs this socket dir)", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            if server is not None and server.poll() is not None:
                print("frontend exited; shutting down", file=sys.stderr)
                break
            for p in procs:
                if not p.is_alive():
                    print("a worker died; shutting down", file=sys.stderr)
                    stop.set()
                    break
            time.sleep(0.5)
    finally:
        print("\nshutting down...", flush=True)
        if server is not None:
            server.terminate()
        cond.stop()
        serve_thread.join(timeout=3)
        cond.shutdown_workers()
        for p in procs:
            p.join(timeout=5)
    return 0


# --- qwen3-omni text-out specialization ------------------------------------

def serve_qwen3_omni_text(*, tp_world: int = 2, port: int = 8000,
                          max_new: int = 256, tp_port: int = 29730,
                          binary: Path | None = None, frontend: bool = True) -> int:
    """Serve Qwen3-Omni chat (audio/image/text-in → text-out), Thinker sharded
    across ``tp_world`` GPUs (from CUDA_VISIBLE_DEVICES)."""
    from mstar_rs.models import Qwen3OmniThinkerPolicy

    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_qwen3_")
    worker_ids = [f"thinker_r{r}" for r in range(tp_world)]
    worker_specs = [
        (wid, _thinker_worker, (wid, rank, tp_world, socket_dir, tp_port, max_new))
        for rank, wid in enumerate(worker_ids)
    ]
    return serve(
        model_name="qwen3_omni",
        policy_factory=lambda: Qwen3OmniThinkerPolicy(max_output_tokens=max_new, greedy=True),
        worker_specs=worker_specs,
        node_to_workers={"Thinker": worker_ids},
        port=port,
        binary=binary,
        frontend=frontend,
        socket_dir=socket_dir,
    )


def serve_qwen3_omni_speech(*, tp_world: int = 2, port: int = 8000,
                            max_new: int = 2048, voice: str = "chelsie",
                            tp_port: int = 29732, binary: Path | None = None,
                            frontend: bool = True) -> int:
    """Serve Qwen3-Omni speech-out (audio/image/text-in → text + 24 kHz speech):
    Thinker sharded across ``tp_world`` GPUs + Talker/Code2Wav on the next GPU
    (all from CUDA_VISIBLE_DEVICES, so use e.g. CVD=1,6,7 for tp_world=2)."""
    from mstar_rs.models import Qwen3OmniAudioPolicy

    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_qwen3_speech_")
    thinker_ids = [f"thinker_r{r}" for r in range(tp_world)]
    worker_specs = [
        (wid, _thinker_worker, (wid, rank, tp_world, socket_dir, tp_port, max_new, True))
        for rank, wid in enumerate(thinker_ids)
    ]
    # The audio worker takes the GPU after the Thinker's TP ranks (cuda:tp_world).
    worker_specs.append(
        ("audio", _audio_worker, ("audio", socket_dir, f"cuda:{tp_world}", voice, max_new)))
    return serve(
        model_name="qwen3_omni",
        policy_factory=lambda: Qwen3OmniAudioPolicy(max_output_tokens=max_new, voice=voice),
        worker_specs=worker_specs,
        node_to_workers={"Thinker": thinker_ids, "Talker": "audio", "Code2Wav": "audio"},
        port=port,
        binary=binary,
        frontend=frontend,
        socket_dir=socket_dir,
    )


def serve_qwen3_omni_speech_decentralized(*, tp_world: int = 2, port: int = 8000,
                                          max_new: int = 2048, voice: str = "chelsie",
                                          tp_port: int = 29734, binary: Path | None = None,
                                          frontend: bool = True) -> int:
    """Serve the speech stack on the DECENTRALIZED runtime: each worker owns its
    partition and drives its own scheduler loop (mstar's in-worker model), so
    the per-token decode pays no conductor round-trip — the coordinator is
    per-request only (ingest + emission relay). This removes the per-step
    dispatch overhead of the centralized path. Concurrency-safe under TP:
    thinker0 broadcasts each batch decision to the follower ranks (tp_sched,
    mstar's ScheduleTPNode), which replay identical batches in identical
    order, keeping the NCCL collectives aligned."""
    from mstar_rs.dist import DisaggCoordinator
    from mstar_rs.models import Qwen3OmniAudioPolicy

    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_qwen3_disagg_")
    thinker_ids = [f"thinker{r}" for r in range(tp_world)]
    partition_to_workers = {"Thinker": thinker_ids, "Talker": ["audio"], "Code2Wav": ["audio"]}
    # a worker shipping a stream chunk sends it to the consumer's I/O leader
    ship_to = {"Thinker": thinker_ids[0], "Talker": "audio", "Code2Wav": "audio"}

    followers = thinker_ids[1:]
    worker_specs = [
        (wid, _disagg_speech_worker,
         (wid, ["Thinker"], ship_to, socket_dir, tp_world, tp_port, max_new, voice,
          followers))
        for wid in thinker_ids
    ]
    worker_specs.append(
        ("audio", _disagg_speech_worker,
         ("audio", ["Talker", "Code2Wav"], ship_to, socket_dir, tp_world, tp_port,
          max_new, voice, None)))

    return serve(
        model_name="qwen3_omni",
        worker_specs=worker_specs,
        port=port,
        binary=binary,
        frontend=frontend,
        socket_dir=socket_dir,
        # Bind the coordinator as "conductor": that's the peer the Rust
        # frontend submits to (workers' coordinator_id points here too).
        backend_factory=lambda: DisaggCoordinator(
            Qwen3OmniAudioPolicy(max_output_tokens=max_new, voice=voice),
            partition_to_workers, socket_dir, my_id="conductor"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mstar-rs serving launcher (axum + conductor + workers)")
    ap.add_argument("--model", default="qwen3_omni", help="model name (adapter key)")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel world size (Thinker)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speech", action="store_true",
                    help="serve the speech-out stack (adds Talker+Code2Wav on cuda:tp); "
                         "default is text-out only")
    ap.add_argument("--decentralized", action="store_true",
                    help="speech stack on the decentralized runtime: in-worker "
                         "schedulers, per-request coordinator, no per-step conductor "
                         "dispatch (single request per TP group for now)")
    ap.add_argument("--voice", default="chelsie")
    ap.add_argument("--max-new", type=int, default=None,
                    help="max output tokens (default 256 text / 2048 speech)")
    ap.add_argument("--no-frontend", action="store_true",
                    help="start only the Python backend (workers + conductor bridge); "
                         "run mstar-server separately against the printed socket dir")
    ap.add_argument("--binary", default=None, help="path to the mstar-server release binary")
    args = ap.parse_args(argv)

    if args.model != "qwen3_omni":
        print(f"launcher currently supports --model qwen3_omni; got {args.model!r}", file=sys.stderr)
        return 2
    frontend = not args.no_frontend
    if args.decentralized:
        if not args.speech:
            print("--decentralized currently requires --speech (the multi-partition "
                  "stack is where in-worker scheduling pays)", file=sys.stderr)
            return 2
        return serve_qwen3_omni_speech_decentralized(
            tp_world=args.tp, port=args.port, voice=args.voice,
            max_new=args.max_new or 2048, binary=args.binary, frontend=frontend,
        )
    if args.speech:
        return serve_qwen3_omni_speech(
            tp_world=args.tp, port=args.port, voice=args.voice,
            max_new=args.max_new or 2048, binary=args.binary, frontend=frontend,
        )
    return serve_qwen3_omni_text(
        tp_world=args.tp, port=args.port, max_new=args.max_new or 256,
        binary=args.binary, frontend=frontend,
    )


if __name__ == "__main__":
    sys.exit(main())
