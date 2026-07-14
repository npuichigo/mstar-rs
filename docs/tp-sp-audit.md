# TP/SP audit: mstar's `CommGroup` / sharding path vs the Rust runtime

The Step-5 audit item from RFC #130 ("maybe TP/SP logic … I haven't audited it
yet against our `CommGroup` / Ulysses-SP path"). Audited: `distributed/base.py`
(`ShardingGroup`/`ShardingConfig`/`compute_fanout`), `distributed/
communication.py` (`TPCommGroup`/`WorkerTPGroups`), and every model's
`get_sharding_config`.

## What mstar actually has today

**1. `TPCommGroup` — the collective layer.** `all_gather` / `all_reduce` /
`reduce_scatter` / `broadcast` over a NCCL group. There is **no `all_to_all`**
— the primitive Ulysses sequence-parallel attention is built on. No
`sp_size`, no sequence-dimension scatter/gather, no SP attention path exists
anywhere in the tree. **Ulysses-SP is roadmap, not code** — nothing to audit
against yet, and nothing for the scheduler port to carry for it.

*Runtime interaction: none.* mstar-rs already reuses `TPCommGroup` wholesale
inside engines (the collectives run inside `execute`, invisible to
scheduling). When Ulysses lands it adds `all_to_all` to this same layer —
engine-level, scheduler-agnostic.

**2. `ShardingGroup` / `ShardingConfig` — conductor-side shard routing.**
Per-`(node, graph_walk)` worker groups, per-signal `shard_dim` declarations,
and `compute_fanout`: for a tensor crossing an edge between groups of
different `tp_size`, interval arithmetic on the leading shard dim decides
which slice goes to which destination rank (vLLM-style divisibility
enforced); `shard_dim=None` means replicated (rank 0 broadcasts to workers
that lack the tensor).

*Currently exercised by zero models.* Every registered model returns
`ShardingConfig(groups=[], shard_dim={})` — all signals replicated, groups
degenerate to per-node singletons. The machinery is real but dormant.

## Delta vs the Rust runtime

| mstar capability | mstar-rs today | Step-5 relevance |
|---|---|---|
| TP execution (collectives in the forward) | same — reuses `TPCommGroup` in engines | none: below the scheduler |
| TP batch agreement across ranks | `ScheduleTPNode` follow path | **ported** (`set_tp_follower_nodes` + `next_batch_for`, leader-broadcast, ordering fixes) |
| Replicated-signal fanout | equivalent by construction: seeds/inputs fan out through host SHM (readable by all ranks); rank-0-only output routing | none |
| **Sharded-signal fanout** (`shard_dim` + `compute_fanout`) | **not implemented** — full tensors only | portable 1:1 whenever needed: `compute_fanout` is a pure function (interval overlap), no scheduler coupling; it belongs wherever routing lives (Step 4's conductor). Dormant upstream too, so not a blocker |
| **Per-`(node, graph_walk)` worker groups** (e.g. prefill and decode on different rank sets) | **not supported** — `node → workers` is constant across walks | the one real generalization for the Step-5 scheduler port: batch routing would key on `(node, walk)` instead of `node`. Mechanical, but touches dispatch tables |
| Streaming-consumer constraint (consumers must have walk-independent groups) | same constraint, implicitly (a `DisaggWorker` owns its partition for all walks) | none — both sides agree |

## Conclusions for Step 5

1. **No SP work is needed in the scheduler port.** Ulysses-SP doesn't exist
   upstream yet; when it does, its collectives live in the engine layer and
   its ranks behave exactly like TP ranks from the scheduler's point of view —
   the leader-broadcast batch contract already covers any SPMD group whose
   ranks must run identical batches.
2. **Sharded fanout can be ported on demand.** `compute_fanout` is
   self-contained interval math; since no model declares a `shard_dim`, port
   it when the first model does (or keep it Python — it runs per-edge at
   routing time, not per-step).
3. **The one genuine gap to schedule into Step 5:** per-`(node, graph_walk)`
   worker groups. The Rust runtime and both serving paths assume a node's
   worker set is walk-independent; mstar's config permits per-walk groups
   (unused today, but a supported shape). Keying dispatch on `(node, walk)`
   closes it.
