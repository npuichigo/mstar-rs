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
| **T4 model** | `python/mstar_rs/models/orpheus.py` | ✅ third model | **Orpheus** speech LM: `LLM` partition (prefill + AR decode loop, KV append 1, mstar's `Sampler`, EOS-stop) streaming `new_token` under a SlidingWindow(28,7) into a self-triggered `SNAC` partition (vendored SNAC decoder → 24 kHz audio). Reuses mstar's `OrpheusForCausalLM` behind a causal FlashInfer handle (llama3 RoPE), with the single-token decode step **CUDA-graph-captured** (scratch-page warmup so AR writes don't corrupt the cache). Greedy token stream **bit-exact (81/81)** vs an independent dense-attention reference; warm RTF **0.41** (4.8 ms/token) on 1× H100. |
| T5 dependent | `crates/mstar-comm` | ▢ planned | ZMQ PUSH/PULL control mesh (serde/msgpack envelopes — replaces pickle) + SHM-ring tensor transport. Only needed for multi-process / disaggregated deployment. |
| T5 dependent | `crates/mstar-server` | ▢ planned | axum HTTP shell (`/generate`, OpenAI routes). Thin; can equally stay FastAPI. |

Model roadmap (runtime features each one forces): **vjepa2** ✅ (stateless walks — T0-T3) →
**pi05** ✅ (paged KV cache + `Loop` flow-matching — T4) → **orpheus** ◐ (streaming
partitions — tier done, model awaiting weight access) → **bagel / qwen3_omni** (CFG-parallel,
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
| **Orpheus** (RTF) | bf16 + FlashInfer + CUDA-graph decode | **0.345** | 0.409 | mstar **1.2× faster** (why below) |
| **control plane** (no-op, 2-node walk) | like-for-like | 184 µs¹ | **79 µs** | the layer mstar-rs rewrites |

vjepa2/pi05 are e2e request latency (20 iters); orpheus is RTF = wall/audio-seconds
(length-normalized). ¹ Python mstar's `WorkerGraphIO` layer *alone* — excludes its
scheduler, tensor store, ZMQ, pickle; the Rust number is the whole runtime.

**Reading the results.** mstar-rs wins decisively when per-request serving overhead
dominates (vjepa2, pi05: HTTP + ZMQ + pickle + SHM hops that the in-process runtime
eliminates — part of the gap is architecture, not language, but removing those hops for
single-GPU deployment is the design). The advantage shrinks as the autoregressive decode
loop comes to dominate wall-clock and amortizes the serving overhead — and **orpheus is
where mstar's more mature engine edges ahead**: it CUDA-graphs the SNAC decoder (5 shape
buckets, ~40 invocations/request) and the sampler, while mstar-rs graphs only the LLM
decode step and runs SNAC + sampling eager. That's the entire ~1.2× gap and it's a known,
closable optimization — at the cost of mstar's ~10-minute warmup (37 LLM + 5 SNAC graph
captures across buckets) versus mstar-rs's ~1 second (one decode graph per shape). The
control-plane row is the pure like-for-like: the Rust runtime is **>2× faster** than even
mstar's graph-IO layer in isolation, and its share of any real request is ~0.1-0.3%.

**Numerical fidelity** (all verified vs independent dense references sharing no runtime
code): pi05 actions max_abs_diff 7.0e-3 / cosine 0.999986; orpheus greedy tokens
**bit-exact 81/81**; vjepa2 predictor bit-exact. The engine (`python/mstar_rs/fi.py`)
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
