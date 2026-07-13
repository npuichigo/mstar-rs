"""End-to-end proof: the Qwen3-Omni Thinker runs text-AR through the FULL
mstar-rs multi-process runtime, tensor-parallel across 2 GPUs.

This closes the loop opened by the two half-proofs:
  * tp_thinker_forward.py — a real sharded Thinker runs under NCCL (standalone),
  * verify_dist_tp.py    — the conductor dispatches/dedupes TP correctly (toy).
Here the REAL sharded Thinker is driven by the REAL conductor: a conductor
process (weightless Qwen3OmniThinkerPolicy) + two TP worker processes (rank 0
on GPU1, rank 1 on GPU6, each a ~31 GB shard of Qwen3OmniThinkerEngine). The
conductor sends every prefill/decode batch to both ranks; they run in NCCL
lockstep; only rank 0's tokens are routed. Success = coherent text ("Paris").

    CUDA_VISIBLE_DEVICES=1,6 python examples/verify_thinker_tp_dist.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

WORLD = 2
TP_PORT = 29713
PROMPT = "What is the capital of France? Answer in one sentence."
MAX_NEW = 32


def worker_main(worker_id: str, tp_rank: int, socket_dir: str) -> None:
    try:
        from mstar_rs.dist import Worker
        from mstar_rs.models import Qwen3OmniThinkerEngine

        engine = Qwen3OmniThinkerEngine(
            tp_rank=tp_rank, tp_world=WORLD, max_output_tokens=MAX_NEW,
            greedy=True, tp_port=TP_PORT,
        )
        # device lives on the engine (cuda:tp_rank, set by init_dist).
        Worker(worker_id, engine, socket_dir, device=str(engine.device)).run()
    except Exception:
        import traceback

        with open(f"{socket_dir}/{worker_id}.err", "w") as f:
            traceback.print_exc(file=f)
        raise


def main() -> int:
    ctx = mp.get_context("spawn")
    socket_dir = tempfile.mkdtemp(prefix="mstar_rs_thinker_tp_")
    ids = [f"thinker_r{r}" for r in range(WORLD)]

    workers = []
    for rank, wid in enumerate(ids):
        p = ctx.Process(target=worker_main, args=(wid, rank, socket_dir), daemon=True)
        p.start()
        workers.append(p)

    # Each worker rendezvous-inits NCCL then loads its ~31 GB shard (~35 s)
    # before binding its ZMQ inbox — wait generously for the .ipc files.
    print("waiting for TP workers to init + load shards (~40 s)...", flush=True)
    deadline = time.time() + 180
    need = {f"{socket_dir}/{w}.ipc" for w in ids}
    while time.time() < deadline and not all(Path(p).exists() for p in need):
        if any(Path(f"{socket_dir}/{w}.err").exists() for w in ids):
            break
        time.sleep(0.5)
    bound = [w for w in ids if Path(f"{socket_dir}/{w}.ipc").exists()]
    print(f"workers bound: {bound}", flush=True)

    from mstar_rs.dist import Conductor
    from mstar_rs.models import Qwen3OmniThinkerPolicy

    policy = Qwen3OmniThinkerPolicy(max_output_tokens=MAX_NEW, greedy=True)
    cond = Conductor(policy, node_to_worker={"Thinker": ids}, socket_dir=socket_dir)

    rid = cond.submit({"prompt": PROMPT})
    results = cond.run_until_idle()
    toks = results.get(rid, [])
    text = policy.tokenizer.decode(toks, skip_special_tokens=True)

    cond.shutdown_workers()
    for p in workers:
        p.join(timeout=5)
    for wid in ids:
        errf = Path(f"{socket_dir}/{wid}.err")
        if errf.exists():
            print(f"--- {wid} error ---\n{errf.read_text()}")

    print(f"\n[gen] {len(toks)} tokens: {text!r}")
    ok = "Paris" in text
    print(f"[verify] coherent (contains 'Paris'): {'PASS' if ok else 'FAIL'}")
    print(f"\nTHINKER TP E2E {'OK' if ok else 'FAILED'} "
          f"(conductor -> 2 TP workers -> sharded Thinker)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
