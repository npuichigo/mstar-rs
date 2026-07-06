# mstar-rs

A Rust re-implementation of the [M*](../mstar) serving runtime, designed from the position of
"if we were the initial implementers and chose Rust for the runtime": **Rust owns the control
plane, Python owns the model compute.**

## Why this split

Profiling of the Python mstar runtime showed that tensor *transport* is a small fraction of
end-to-end time, while the control plane — graph-walk bookkeeping, scheduling, routing,
fan-out — is pure Python running under the GIL, interleaved with GPU driving code. The durable
win from Rust is a **GIL-free, allocation-light control plane** with a thin, well-defined
boundary to Python for everything that touches a `torch.Tensor`.

M*'s own architecture makes this split unusually clean, because its core contract already
separates the two planes:

- The **control plane** carries only *descriptors* (`TensorPointerInfo` in Python mstar,
  `TensorRef` here): uuid, dims, dtype. Never tensor bytes.
- The **data plane** moves bytes out-of-band (in-process object store, SHM ring, RDMA).

So the Rust/Python seam is exactly the seam mstar already has between its runtime and its
engines (`BaseEngine`: `NodeBatch` in → `NodeOutput` out).

| Plane | Owner | Contents |
|---|---|---|
| Control (Rust) | `mstar-core`, `mstar-sched`, `mstar-runtime` | graph IR, walk state machine, readiness/loop bookkeeping, continuous-batching scheduler, request lifecycle, tensor refcounts |
| Boundary (PyO3) | `mstar-py` | `Runtime` class: `add_request` / `start_walk` / `next_batch` / `complete_batch` → typed events |
| Data (Python) | `python/mstar_rs/` | torch submodules (engines), uuid→Tensor object store, media loading, tokenization, postprocess, walk-transition policy |

**Inversion of control at the boundary.** Python mstar's conductor calls
`model.get_partition_forward_pass_args(...)` (Python calling Python). Here the Rust runtime
never calls into Python; it *returns events* (`Emission`, `WalkDone`, `RequestDone`) from
`complete_batch()`, and a thin Python driver loop feeds them to the model's policy, which
answers by calling `start_walk()` / `finish_request()`. The GIL is held only while torch runs.

## Crate layout & implementation order

Order is dependency order — each tier is usable and testable without the tiers after it.
"Core" = the essence of M* (a model is a dataflow graph; a request is a Walk over it).
"Dependent" = deployment machinery that scales the core out.

| Tier | Crate / dir | Status | What it is |
|---|---|---|---|
| **T0 core** | `crates/mstar-core` | ✅ implemented | Graph IR (`NodeSpec`, `EdgeSpec`, `Sequential`/`Parallel`/`Loop`), compiled walk graphs, per-request **walk state machine** (readiness signals, next-iter buffering, loop advance/terminate, emissions/persist routing). Pure Rust, zero I/O, heavily unit-tested. |
| **T1 core** | `crates/mstar-sched` | ✅ implemented | Continuous-batching **micro-scheduler**: scans ready nodes across requests, groups by (node, walk), round-robin fairness, batch-size clamp. Mirrors `worker/micro_scheduler.py`. |
| **T2 core** | `crates/mstar-runtime` | ✅ implemented | Request lifecycle (the conductor logic, in-process): walk seeding, batch issue/complete, event stream, tensor uuid registry. Mirrors `conductor/conductor.py` + `worker/node_manager_utils.py` for the single-worker case. |
| **T3 boundary** | `crates/mstar-py` + `python/mstar_rs` | ✅ implemented | PyO3/maturin extension `mstar_rs._core` + Python package: object store, executor protocol, driver loop, model registry. |
| **T3 model** | `python/mstar_rs/models/vjepa2.py` | ✅ first model | **V-JEPA 2** (`prefill_video`: `video_encoder → predictor → EMIT_TO_CLIENT`) — all-stateless, no KV cache: the cleanest end-to-end proof of the runtime. |
| **T4** | KV-cache tier (`mstar-runtime::kv` + `python/mstar_rs/fi.py`) | ✅ implemented | Rust: paged-KV **allocator** (free-list, per-(request,label) page tables + seq positions), schedule-time reservation with hold-on-OOM, scratch-page reservation for transient suffixes, page-table views in `Batch.kv`, advance-on-complete, free-on-finish. Python: KV tensors in mstar's layout + a cache handle implementing mstar's cache-manager interface (`apply_rope`/`run_attention`/…) over **FlashInfer paged attention in bf16** — mstar transformer modules run unmodified, on the same kernels mstar's engine uses. |
| **T4 model** | `python/mstar_rs/models/pi05.py` | ✅ second model | **Pi0.5**: prefill (SigLIP fp32 → PaliGemma writes paged prefix KV, bf16) + 10-step flow-matching `action_gen` loop as **one CUDA-graph-captured euler step replayed 10×** (frozen prefix, `kv_appends=0` + suffix scratch). Reuses mstar's pi05 nn modules + weight remapping wholesale; the single code path matches mstar's engine config exactly. |
| **T4** | Streaming tier (`mstar-runtime::stream` + partitions) | ✅ implemented | Rust: **`ChunkPolicy`** (sliding-window / ramp / left-context / fixed) + **`StreamBuffer`** (in-order windowing, overlap, producer-done flush, `continue_after_done`); the runtime runs **concurrent per-partition walk slots** per request, delivers ready windows into consumer walks (`pump_streams`), and marks partition-done on the pass that consumes the final chunk. Python driver seeds every partition at ingest, `finish_partition` completes the request when all are done. Unit-tested + toy e2e. |
| **T4 model** | `python/mstar_rs/models/orpheus.py` | ✅ third model | **Orpheus** speech LM: `LLM` partition (prefill + AR decode loop, KV append 1, EOS-stop) streaming `new_token` under a SlidingWindow(28,7) into a self-triggered `SNAC` partition (vendored SNAC decoder → 24 kHz audio). Reuses mstar's `OrpheusForCausalLM` behind a causal FlashInfer handle (llama3 RoPE); the decode step is **CUDA-graph-captured with the sampler in-graph** (mstar's `CudaGraphableSampler`) and the SNAC decoder is graphed too — matching mstar's decode path. LLM data plane **bit-exact 81/81** vs a dense reference; **RTF 0.349, matched to mstar's 0.345** on 1× H100. |
| **T5 core** | Async execution pipeline (`mstar-runtime` + driver) | ▢ planned | The runtime capability, not a per-model trick: **overlap host-side scheduling/routing/postprocess with GPU compute** across every walk, mirroring mstar's worker `run()` (async 1-deep speculative pipeline). Runtime side: speculative `next_batch` (predict the next-ready nodes from the walk state machine — pure Rust control plane) + decoupled async `complete_batch`; driver side: a GPU stream/thread that launches batch N+1 while the host routes/postprocesses N, CUDA-event-gated, with off-stream stop checks (no per-token `.item()` sync). Model code is unchanged — `execute` stays synchronous. Benefits every model (vjepa2 sequential, pi05 flow loop, orpheus AR decode); mstar-rs can do the speculation GIL-free. |
| **T5** | Multi-process transport (`crates/mstar-comm`) | ✅ implemented | Both planes. **Control**: a PUSH/PULL message `Mailbox` (bind one inbox, fire-and-forget sends by id, ordered delivery, reconnect-on-restart via socket-inode check) over Unix-domain sockets + length-prefixed bincode frames — a real wire format replacing pickle, no libzmq dependency. **Data**: `ShmArena` — a persistent `/dev/shm` mmap + first-fit free-list allocator, exposed to Python via the buffer protocol so `torch.frombuffer(memoryview(arena)[off:off+n])` moves tensors cross-process zero-copy. |
| **T5** | Runtime split (`python/mstar_rs/dist.py`) | ✅ implemented | **Conductor + worker processes.** The conductor owns the Rust `Runtime` + model policy (descriptor-only, never runs the model) and dispatches each batch to the node's worker over the `Mailbox` (msgpack control); a **stateless worker** reads inputs from SHM, runs `model.execute`, stages outputs back to SHM, and replies with descriptors. Proven by `examples/run_dist.py`: a disaggregated 2-node model with node `a` on worker_0 and node `b` on worker_1, output crossing worker→worker via `/dev/shm` — the real multi-GPU path — with only descriptors on the wire. |
| T5 dependent | `crates/mstar-server` | ▢ planned | axum HTTP shell (`/generate`, OpenAI routes). Thin; can equally stay FastAPI. |

Model roadmap (runtime features each one forces): **vjepa2** ✅ (stateless walks — T0-T3) →
**pi05** ✅ (paged KV cache + `Loop` flow-matching — T4) → **orpheus** ✅ (streaming
partitions + AR decode — T4; RTF-matched to mstar) → **bagel / qwen3_omni** (CFG-parallel,
MoE).

## Core semantics (faithful to Python mstar)

- A model declares named **walks** (`prefill`, `decode`, …), each a `Section` tree over
  `NodeSpec`s. Edges carry an input *name*; a node is ready when `input_names ⊆ received`.
- `EMIT_TO_CLIENT` / `EMPTY_DESTINATION` are special edge destinations; `persist: true` edges
  become walk outputs handed to the policy (mstar's "persist signals to conductor").
- **Next-iter buffering**: an input arriving for a node that already has that input (or already
  ran this iteration) buffers into the node's next-iteration slot — this is how loop-back edges
  work, mirroring `ready_next_iter` in `graph/base.py`.
- **Loops**: `complete_iter()` terminates on `curr_iter+1 == max_iters` or an explicit finish
  signal (`signal_loop_finish`, mstar's `check_stop → STOP_LOOPS`); `outputs` snapshot the last
  iteration, `accumulated_outputs` append across iterations. External inputs are re-injected on
  advance. (v0: single-level loops; nesting is documented-deferred.)
- A walk is **done** when every node has completed and all loops have terminated; the policy
  then picks the next walk (or finishes the request) — one forward pass per round, exactly like
  the conductor.

## Quickstart

```bash
cd mstar-rs
cargo test --workspace                       # T0-T2 unit tests
cd crates/mstar-py
maturin develop --release                    # builds mstar_rs into the active venv
python ../../examples/run_vjepa2.py --video <path>   # first end-to-end model
```

## Measured performance vs Python mstar (2026-07, 1× H100)

Every row is **matched-config** — mstar-rs and Python mstar (`mstar serve`) run the same
compute (same precision, same FlashInfer/CUDA-graph path). All figures are warm (post-JIT/
capture), on the same otherwise-idle GPU.

| Model | matched config | Python mstar | mstar-rs | |
|---|---|---|---|---|
| **V-JEPA 2** ViT-L | bf16 autocast + `torch.compile` | 1172 ms | **380 ms** | mstar-rs **3.1× faster** |
| **Pi0.5** | bf16 + FlashInfer paged attn + CUDA-graph decode | 74.8 ms | **57.6 ms** | mstar-rs **1.3× faster** |
| **Orpheus** (RTF) | bf16 + FlashInfer + CUDA-graph decode **incl. in-graph sampler + SNAC** | 0.345 | **0.349** | **matched** (≈1%) |
| **control plane** (no-op, 2-node walk) | like-for-like | 184 µs¹ | **79 µs** | the layer mstar-rs rewrites |

vjepa2/pi05 are e2e request latency (20 iters); orpheus is RTF = wall/audio-seconds
(length-normalized). ¹ Python mstar's `WorkerGraphIO` layer *alone* — excludes its
scheduler, tensor store, ZMQ, pickle; the Rust number is the whole runtime.

**Reading the results.** mstar-rs wins decisively when per-request serving overhead
dominates (vjepa2, pi05: HTTP + ZMQ + pickle + SHM hops that the in-process runtime
eliminates — part of the gap is architecture, not language, but removing those hops for
single-GPU deployment is the design). As the autoregressive decode loop comes to dominate
wall-clock, that advantage amortizes away and it comes down to graphing the same kernels
mstar graphs. **Orpheus reaches parity once the whole decode step is captured to match
mstar** — the LLM forward, the sampler (`CudaGraphableSampler`, in-graph seeded sampling),
and the SNAC decoder. Profiling was decisive here: the eager sampler cost ~4.4 ms/token —
*more than the entire 28-layer LLM decode* — and folding it into the graph (as mstar does)
took RTF 0.408 → 0.349. The residual ~1% is the synchronous driver: mstar's worker overlaps
host-side scheduling/routing/postprocess with GPU compute (async 1-deep speculative
pipeline), and the mstar-rs v0 driver does not yet — this is the **T5 async-execution-pipeline
tier** (a runtime capability that benefits every walk, not a per-model tweak), not a language
gap: the control-plane row shows the Rust runtime is **>2× faster** than even mstar's
graph-IO layer in isolation, and its share of any real request is ~0.1-0.3%. Speculative
scheduling is pure control-plane logic — exactly what the Rust walk state machine does — so
mstar-rs can pipeline GIL-free.

**Numerical fidelity** (all verified vs independent dense references sharing no runtime
code): pi05 actions max_abs_diff 7.0e-3 / cosine 0.999986; vjepa2 predictor bit-exact;
orpheus LLM data plane (KV / attention / RoPE) **bit-exact 81/81** greedy tokens via the
eager-argmax path, and the in-graph sampler is separately verified capture-faithful
(identical token eager-vs-graphed on identical inputs — greedy isn't argmax-reproducible
because mstar encodes it as `top_k=1`, which breaks flat-codebook ties differently). The
engine (`python/mstar_rs/fi.py`)
mirrors mstar's recipe exactly — same KV layout `[L, pages, 2, page_size, kvh, hd]`,
`BatchPrefillWithPagedKVCacheWrapper("NHD")` planned from CPU int32 page tables,
fancy-indexed K/V writes, FlashInfer RoPE — so mstar's transformer modules run unmodified.

## The driver loop (what replaces conductor+worker in-process)

```python
rt = mstar_rs.Runtime(walks_json)                 # Rust
rid = rt.add_request()
rt.start_walk(rid, "prefill_video", model.initial_inputs(rid, media))   # policy (Python)
while (batch := rt.next_batch(max_batch_size)) is not None:             # Rust picks
    outs = executor.execute(batch, store)                               # Python/torch
    for ev in rt.complete_batch(batch.batch_id, outs):                  # Rust routes
        model.on_event(ev, rt, store)             # Emission → postprocess; WalkDone → next walk
```
