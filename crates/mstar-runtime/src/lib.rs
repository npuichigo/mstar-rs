//! mstar-runtime: request lifecycle over walk state machines — the
//! single-process equivalent of Python mstar's conductor + worker routing.
//!
//! Control is *inverted* relative to Python mstar: instead of the conductor
//! calling into the model (`get_partition_forward_pass_args`), this runtime
//! returns typed [`Event`]s from [`Runtime::complete_batch`], and the driver
//! (Python) answers by calling [`Runtime::start_walk`] /
//! [`Runtime::finish_request`]. The runtime therefore never holds the GIL
//! and never calls Python.
//!
//! Tensors exist here only as [`TensorRef`] descriptors; the driver owns the
//! `uuid -> torch.Tensor` object store.

pub mod kv;
pub mod stream;

use std::collections::BTreeMap;

use mstar_core::{
    CompiledWalk, CoreError, IncomingInput, RouteEvent, TensorRef, WalkSet, WalkState,
};
use mstar_sched::{MicroScheduler, ReadyEntry};
use serde::Deserialize;
use thiserror::Error;

pub use kv::{KvCacheConfig, KvManager, KvRequestState, KvView};
pub use stream::{ChunkPolicySpec, StreamBuffer, StreamChunk};

/// The partition every walk belongs to when a model declares none.
pub const DEFAULT_PARTITION: &str = "default";

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("unknown request {0}")]
    UnknownRequest(u64),
    #[error("unknown batch {0}")]
    UnknownBatch(u64),
    #[error("partition '{1}' of request {0} already has an active walk")]
    WalkActive(u64, String),
    #[error("partition '{1}' of request {0} is done")]
    PartitionDone(u64, String),
    #[error("unknown partition '{0}'")]
    UnknownPartition(String),
    #[error("request {0} is finished")]
    RequestFinished(u64),
    #[error("batch {batch}: missing outputs for request {request}")]
    MissingOutputs { batch: u64, request: u64 },
    #[error("no connection from partition '{from}' with edge '{edge}' into '{to}'")]
    UnknownConnection {
        from: String,
        edge: String,
        to: String,
    },
    #[error("invalid model spec: {0}")]
    InvalidSpec(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Declares which walks belong to a partition (mstar `PartitionDefinition`,
/// minus `initial_walk` — the policy seeds initial walks explicitly).
#[derive(Debug, Clone, Deserialize)]
pub struct PartitionSpec {
    pub name: String,
    pub walks: Vec<String>,
}

/// A streaming connection between two partitions (mstar `Connection`).
#[derive(Debug, Clone, Deserialize)]
pub struct ConnectionSpec {
    pub from: String,
    pub to: String,
    pub edge_name: String,
    pub policy: ChunkPolicySpec,
}

/// Full model spec: walks plus optional partition topology.
#[derive(Debug, Deserialize)]
struct ModelSpec {
    walks: serde_json::Value,
    #[serde(default)]
    partitions: Vec<PartitionSpec>,
    #[serde(default)]
    connections: Vec<ConnectionSpec>,
}

/// What the runtime tells the driver after routing a completed batch.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A tensor reached `EMIT_TO_CLIENT` — postprocess and stream it out.
    Emission {
        request_id: u64,
        partition: String,
        name: String,
        modality: Option<String>,
        tensors: Vec<TensorRef>,
    },
    /// A forward pass (one walk) finished for one partition. `persist`
    /// carries the request's accumulated persist signals (latest value per
    /// name — mstar's conductor-side `persist_signals`). `stream_done` is
    /// true when this walk consumed the FINAL chunk of an incoming stream —
    /// mstar's partition-done-on-the-consuming-pass invariant. The policy
    /// must now call `start_walk` or `finish_partition`.
    WalkDone {
        request_id: u64,
        partition: String,
        walk: String,
        fwd_index: u64,
        persist: Vec<(String, Vec<TensorRef>)>,
        stream_done: bool,
    },
}

/// A scheduled batch: one node, one walk, with per-request input tensors for
/// the data plane to execute.
#[derive(Debug, Clone)]
pub struct Batch {
    pub batch_id: u64,
    pub node: String,
    pub walk: String,
    /// request_id -> input name -> tensors
    pub inputs: BTreeMap<u64, BTreeMap<String, Vec<TensorRef>>>,
    /// For KV-cache nodes: request_id -> page table / position view.
    pub kv: BTreeMap<u64, KvView>,
}

#[derive(Debug, Default)]
struct PartitionRuntimeState {
    walk: Option<WalkState>,
    fwd_index: u64,
    done: bool,
    /// A final stream chunk was ingested into the active walk; its WalkDone
    /// reports `stream_done` (partition-done rides the consuming pass).
    final_chunk_pending: bool,
    /// label -> tokens appended per execution of that label's KV node in the
    /// current walk (0 = read-only). Declared by the policy at start_walk.
    kv_appends: BTreeMap<String, u64>,
    /// label -> transient scratch tokens past the committed sequence that
    /// each execution also needs pages for (pi05's flow-loop suffix).
    kv_scratch: BTreeMap<String, u64>,
}

#[derive(Debug)]
struct RequestState {
    /// One slot per partition, all running concurrently.
    partitions: BTreeMap<String, PartitionRuntimeState>,
    /// Stream buffers, one per connection (indexed like `connections`).
    buffers: Vec<StreamBuffer>,
    persist: BTreeMap<String, Vec<TensorRef>>,
    finished: bool,
}

#[derive(Debug)]
struct InflightBatch {
    node: String,
    partition: String,
    request_ids: Vec<u64>,
    /// (request, label, append) to advance when the batch completes.
    kv_advances: Vec<(u64, String, u64)>,
}

/// The in-process M* runtime: walk registry + per-request state + scheduler.
#[derive(Debug)]
pub struct Runtime {
    walks: WalkSet,
    /// walk name -> partition name (DEFAULT_PARTITION when undeclared).
    walk_partition: BTreeMap<String, String>,
    partition_names: Vec<String>,
    connections: Vec<ConnectionSpec>,
    requests: BTreeMap<u64, RequestState>,
    scheduler: MicroScheduler,
    inflight: BTreeMap<u64, InflightBatch>,
    kv: Option<KvManager>,
    next_request_id: u64,
    next_batch_id: u64,
    next_uuid: u64,
}

impl Runtime {
    /// Build from the walk-set JSON (`{"walk_name": <Section>, ...}`) with a
    /// single default partition (non-streaming models).
    pub fn from_walks_json(json: &str) -> Result<Self> {
        let walks = WalkSet::from_json(json)?;
        let walk_partition = walks
            .walks
            .keys()
            .map(|w| (w.clone(), DEFAULT_PARTITION.to_string()))
            .collect();
        Ok(Self::build(
            walks,
            walk_partition,
            vec![DEFAULT_PARTITION.to_string()],
            Vec::new(),
        ))
    }

    /// Build from a full model spec:
    /// `{"walks": {...}, "partitions": [{name, walks}], "connections":
    ///   [{from, to, edge_name, policy}]}`.
    pub fn from_spec_json(json: &str) -> Result<Self> {
        let spec: ModelSpec = serde_json::from_str(json)
            .map_err(|e| RuntimeError::InvalidSpec(format!("bad model-spec JSON: {e}")))?;
        let walks = WalkSet::from_json(&spec.walks.to_string())?;
        if spec.partitions.is_empty() {
            return Self::from_walks_json(&spec.walks.to_string());
        }
        let mut walk_partition = BTreeMap::new();
        let mut partition_names = Vec::new();
        for p in &spec.partitions {
            partition_names.push(p.name.clone());
            for w in &p.walks {
                if !walks.walks.contains_key(w) {
                    return Err(RuntimeError::InvalidSpec(format!(
                        "partition '{}' references unknown walk '{}'",
                        p.name, w
                    )));
                }
                walk_partition.insert(w.clone(), p.name.clone());
            }
        }
        for w in walks.walks.keys() {
            if !walk_partition.contains_key(w) {
                return Err(RuntimeError::InvalidSpec(format!(
                    "walk '{w}' belongs to no partition"
                )));
            }
        }
        for c in &spec.connections {
            for p in [&c.from, &c.to] {
                if !partition_names.contains(p) {
                    return Err(RuntimeError::UnknownPartition(p.clone()));
                }
            }
        }
        Ok(Self::build(
            walks,
            walk_partition,
            partition_names,
            spec.connections,
        ))
    }

    fn build(
        walks: WalkSet,
        walk_partition: BTreeMap<String, String>,
        partition_names: Vec<String>,
        connections: Vec<ConnectionSpec>,
    ) -> Self {
        Self {
            walks,
            walk_partition,
            partition_names,
            connections,
            requests: BTreeMap::new(),
            scheduler: MicroScheduler::new(),
            inflight: BTreeMap::new(),
            kv: None,
            next_request_id: 0,
            next_batch_id: 0,
            next_uuid: 0,
        }
    }

    /// Configure paged KV caches: one config per label, plus the KV node ->
    /// label mapping (mstar: `get_kv_cache_config` + `get_node_engine_types`).
    pub fn configure_kv(
        &mut self,
        configs: Vec<KvCacheConfig>,
        node_labels: BTreeMap<String, String>,
    ) {
        self.kv = Some(KvManager::new(configs, node_labels));
    }

    /// Current (pages, seq_pos) for a request's cache label, for the data
    /// plane / debugging.
    pub fn kv_state(&self, request_id: u64, label: &str) -> KvRequestState {
        self.kv
            .as_ref()
            .map(|kv| kv.state(request_id, label))
            .unwrap_or_default()
    }

    /// Mint a fresh tensor uuid. The driver registers the actual torch
    /// tensor under this id in its object store.
    pub fn new_uuid(&mut self) -> u64 {
        self.next_uuid += 1;
        self.next_uuid
    }

    pub fn add_request(&mut self) -> u64 {
        self.next_request_id += 1;
        let rid = self.next_request_id;
        self.requests.insert(
            rid,
            RequestState {
                partitions: self
                    .partition_names
                    .iter()
                    .map(|p| (p.clone(), PartitionRuntimeState::default()))
                    .collect(),
                buffers: self
                    .connections
                    .iter()
                    .map(|c| StreamBuffer::new(c.policy.clone()))
                    .collect(),
                persist: BTreeMap::new(),
                finished: false,
            },
        );
        rid
    }

    /// Seed a walk for a request (the policy's forward-pass args). One walk
    /// may be started at a time per request; walk completion is reported via
    /// [`Event::WalkDone`].
    pub fn start_walk(
        &mut self,
        request_id: u64,
        walk_name: &str,
        inputs: Vec<IncomingInput>,
    ) -> Result<()> {
        self.start_walk_with_kv(request_id, walk_name, inputs, BTreeMap::new(), BTreeMap::new())
    }

    /// `kv_appends`: label -> tokens appended by EACH execution of that
    /// label's KV node during this walk (0 = the node only reads the cache).
    /// `kv_scratch`: label -> transient tokens past the committed sequence
    /// each execution also needs pages for (never committed).
    /// The walk occupies its partition's slot (derived from the walk name).
    pub fn start_walk_with_kv(
        &mut self,
        request_id: u64,
        walk_name: &str,
        inputs: Vec<IncomingInput>,
        kv_appends: BTreeMap<String, u64>,
        kv_scratch: BTreeMap<String, u64>,
    ) -> Result<()> {
        let graph: std::sync::Arc<CompiledWalk> = self.walks.get(walk_name)?;
        let partition = self
            .walk_partition
            .get(walk_name)
            .expect("walk in WalkSet implies partition")
            .clone();
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        if req.finished {
            return Err(RuntimeError::RequestFinished(request_id));
        }
        let pstate = req.partitions.get_mut(&partition).expect("known partition");
        if pstate.done {
            return Err(RuntimeError::PartitionDone(request_id, partition));
        }
        if pstate.walk.is_some() {
            return Err(RuntimeError::WalkActive(request_id, partition));
        }
        let mut walk = WalkState::new(graph);
        walk.seed(inputs)?;
        pstate.walk = Some(walk);
        pstate.kv_appends = kv_appends;
        pstate.kv_scratch = kv_scratch;
        Ok(())
    }

    /// Signal a loop to finish for one request (mstar's `STOP_LOOPS`, e.g.
    /// EOS detected by the data plane's check_stop). Applies to whichever
    /// partition's active walk contains the loop.
    pub fn signal_loop_finish(&mut self, request_id: u64, loop_name: &str) -> Result<()> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        for pstate in req.partitions.values_mut() {
            if let Some(walk) = pstate.walk.as_mut() {
                if walk.signal_loop_finish(loop_name).is_ok() {
                    return Ok(());
                }
            }
        }
        Err(RuntimeError::Core(CoreError::UnknownLoop(
            loop_name.to_string(),
            "<active walks>".to_string(),
        )))
    }

    /// Pick the next batch across all requests (continuous batching), or
    /// None when nothing is ready. Marks the chosen nodes as scheduled and
    /// hands back their input tensors.
    /// A node is schedulable for a request if its walk inputs are ready AND
    /// (for KV nodes) its declared append fits in the remaining pages —
    /// mstar's `engine.check_ready` gate in the micro-scheduler.
    fn ready_entries(&self) -> Vec<ReadyEntry> {
        let mut ready = Vec::new();
        for (&rid, req) in &self.requests {
            if req.finished {
                continue;
            }
            for pstate in req.partitions.values() {
                let Some(walk) = &pstate.walk else { continue };
                for node in walk.ready_nodes() {
                    if let Some(kv) = &self.kv {
                        if let Some(label) = kv.label_for_node(&node) {
                            let append = pstate.kv_appends.get(label).copied().unwrap_or(0);
                            let scratch = pstate.kv_scratch.get(label).copied().unwrap_or(0);
                            if !kv.can_reserve(rid, label, append, scratch) {
                                continue; // held until pages free up
                            }
                        }
                    }
                    ready.push(ReadyEntry {
                        request_id: rid,
                        node,
                        walk: walk.walk_name().to_string(),
                    });
                }
            }
        }
        ready
    }

    /// Deliver ready stream chunks into consumer walks (mstar's
    /// `_poll_stream_buffers`): for each connection whose buffer has a chunk
    /// and whose consumer partition's active walk can accept the edge-name
    /// input, pop and ingest. A final chunk marks the partition's
    /// `final_chunk_pending` so its WalkDone reports `stream_done`.
    fn pump_streams(&mut self) {
        for req in self.requests.values_mut() {
            if req.finished {
                continue;
            }
            for (idx, conn) in self.connections.iter().enumerate() {
                let buffer = &mut req.buffers[idx];
                if !buffer.has_chunk_ready() {
                    continue;
                }
                let pstate = req
                    .partitions
                    .get_mut(&conn.to)
                    .expect("connection partitions validated");
                if pstate.done {
                    continue;
                }
                let Some(walk) = pstate.walk.as_mut() else { continue };
                if !walk.can_accept_input(&conn.edge_name) {
                    continue; // consumer busy; chunk stays buffered
                }
                let chunk = buffer.pop_chunk();
                if chunk.is_final {
                    pstate.final_chunk_pending = true;
                }
                walk.ingest_stream_input(&conn.edge_name, chunk.items);
            }
        }
    }

    pub fn next_batch(&mut self, max_batch_size: usize) -> Result<Option<Batch>> {
        self.pump_streams();
        let ready = self.ready_entries();
        let Some(plan) = self.scheduler.next_batch(&ready, max_batch_size) else {
            return Ok(None);
        };

        let kv_label = self
            .kv
            .as_ref()
            .and_then(|kv| kv.label_for_node(&plan.node))
            .map(str::to_string);
        let partition = self
            .walk_partition
            .get(&plan.walk)
            .expect("scheduled walk is known")
            .clone();

        let mut inputs = BTreeMap::new();
        let mut kv_views = BTreeMap::new();
        let mut kv_advances = Vec::new();
        let mut scheduled_rids = Vec::new();
        for &rid in &plan.request_ids {
            let req = self.requests.get_mut(&rid).expect("rid from ready scan");
            let pstate = req.partitions.get_mut(&partition).expect("known partition");
            // Reserve KV pages FIRST: earlier requests in this same batch may
            // have consumed the pages that made `can_reserve` true at scan
            // time. A request that no longer fits is simply left out of the
            // batch (its node stays ready) — mstar's hold-on-OOM behavior.
            // The first request always fits: nothing allocated since the scan.
            if let Some(label) = &kv_label {
                let append = pstate.kv_appends.get(label).copied().unwrap_or(0);
                let scratch = pstate.kv_scratch.get(label).copied().unwrap_or(0);
                let Some(view) = self
                    .kv
                    .as_mut()
                    .expect("kv_label implies kv")
                    .reserve(rid, label, append, scratch)
                else {
                    continue;
                };
                kv_views.insert(rid, view);
                kv_advances.push((rid, label.clone(), append));
            }
            let req = self.requests.get_mut(&rid).expect("rid from ready scan");
            let pstate = req.partitions.get_mut(&partition).expect("known partition");
            let walk = pstate.walk.as_mut().expect("walk present in ready scan");
            inputs.insert(rid, walk.take_node_inputs(&plan.node)?);
            scheduled_rids.push(rid);
        }
        debug_assert!(!scheduled_rids.is_empty());

        self.next_batch_id += 1;
        let batch_id = self.next_batch_id;
        self.inflight.insert(
            batch_id,
            InflightBatch {
                node: plan.node.clone(),
                partition,
                request_ids: scheduled_rids,
                kv_advances,
            },
        );
        Ok(Some(Batch {
            batch_id,
            node: plan.node,
            walk: plan.walk,
            inputs,
            kv: kv_views,
        }))
    }

    /// The data plane finished a batch: route each request's named outputs,
    /// collect emissions, and report walk completions.
    pub fn complete_batch(
        &mut self,
        batch_id: u64,
        outputs: BTreeMap<u64, BTreeMap<String, Vec<TensorRef>>>,
    ) -> Result<Vec<Event>> {
        let inflight = self
            .inflight
            .remove(&batch_id)
            .ok_or(RuntimeError::UnknownBatch(batch_id))?;
        let partition = inflight.partition.clone();

        // A request may have been finished or aborted while this batch was in
        // flight — `finish_request`/`finish_partition` null `pstate.walk`
        // without removing the batch from `self.inflight`. Completing such a
        // request is a no-op: it is not "live", so we skip its routing AND its
        // KV advance rather than dereferencing a `None` walk (which panicked,
        // surfacing as a PyO3 abort and leaving the runtime inconsistent).
        let live: Vec<u64> = inflight
            .request_ids
            .iter()
            .copied()
            .filter(|rid| {
                self.requests
                    .get(rid)
                    .and_then(|r| r.partitions.get(&partition))
                    .map_or(false, |p| p.walk.is_some())
            })
            .collect();

        // Validate outputs for every live request BEFORE mutating any state,
        // so a missing-output error leaves KV positions untouched (recoverable)
        // rather than advancing seq_pos for a walk that did not progress.
        for &rid in &live {
            if !outputs.contains_key(&rid) {
                return Err(RuntimeError::MissingOutputs {
                    batch: batch_id,
                    request: rid,
                });
            }
        }

        if let Some(kv) = &mut self.kv {
            for (rid, label, append) in &inflight.kv_advances {
                if live.contains(rid) {
                    kv.advance(*rid, label, *append);
                }
            }
        }
        let mut events = Vec::new();
        for &rid in &inflight.request_ids {
            if !live.contains(&rid) {
                continue; // finished/aborted while in flight — no-op
            }
            let out = outputs.get(&rid).expect("validated live above").clone();
            let req = self.requests.get_mut(&rid).expect("live rid");
            let pstate = req.partitions.get_mut(&partition).expect("live partition");
            let walk = pstate.walk.as_mut().expect("live walk");
            let result = walk.complete_node(&inflight.node, out)?;
            for ev in result.events {
                match ev {
                    RouteEvent::Emission {
                        name,
                        modality,
                        tensors,
                    } => events.push(Event::Emission {
                        request_id: rid,
                        partition: partition.clone(),
                        name,
                        modality,
                        tensors,
                    }),
                    RouteEvent::Persist { name, tensors } => {
                        req.persist.insert(name, tensors);
                    }
                    RouteEvent::Stream {
                        name,
                        target_partition,
                        tensors,
                    } => {
                        let idx = self
                            .connections
                            .iter()
                            .position(|c| {
                                c.from == partition
                                    && c.to == target_partition
                                    && c.edge_name == name
                            })
                            .ok_or_else(|| RuntimeError::UnknownConnection {
                                from: partition.clone(),
                                edge: name.clone(),
                                to: target_partition.clone(),
                            })?;
                        req.buffers[idx].push(tensors);
                    }
                }
            }
            if result.walk_done {
                let pstate = req.partitions.get_mut(&partition).expect("known partition");
                let walk_name = pstate.walk.as_ref().expect("walk").walk_name().to_string();
                pstate.walk = None;
                pstate.fwd_index += 1;
                let stream_done = std::mem::take(&mut pstate.final_chunk_pending);
                events.push(Event::WalkDone {
                    request_id: rid,
                    partition: partition.clone(),
                    walk: walk_name,
                    fwd_index: pstate.fwd_index,
                    persist: req
                        .persist
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    stream_done,
                });
            }
        }
        Ok(events)
    }

    /// The policy is done with a partition: mark it, and signal
    /// producer-done on every connection leaving it (flushes buffers).
    /// Returns true when every partition of the request is done.
    pub fn finish_partition(&mut self, request_id: u64, partition: &str) -> Result<bool> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        let pstate = req
            .partitions
            .get_mut(partition)
            .ok_or_else(|| RuntimeError::UnknownPartition(partition.to_string()))?;
        pstate.done = true;
        pstate.walk = None;
        for (idx, conn) in self.connections.iter().enumerate() {
            if conn.from == partition {
                req.buffers[idx].signal_done();
            }
        }
        Ok(req.partitions.values().all(|p| p.done))
    }

    pub fn finish_request(&mut self, request_id: u64) -> Result<()> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        req.finished = true;
        for pstate in req.partitions.values_mut() {
            pstate.done = true;
            pstate.walk = None;
        }
        req.persist.clear();
        if let Some(kv) = &mut self.kv {
            kv.free_request(request_id);
        }
        Ok(())
    }

    /// A chunk exists that `pump_streams` would deliver right now.
    fn has_deliverable_chunk(&self) -> bool {
        self.requests.values().any(|req| {
            !req.finished
                && self.connections.iter().enumerate().any(|(idx, conn)| {
                    req.buffers[idx].has_chunk_ready()
                        && req
                            .partitions
                            .get(&conn.to)
                            .is_some_and(|p| {
                                !p.done
                                    && p.walk
                                        .as_ref()
                                        .is_some_and(|w| w.can_accept_input(&conn.edge_name))
                            })
                })
        })
    }

    /// True when no request has schedulable or in-flight work — the same
    /// gate `next_batch` uses, so a KV-held request counts as idle (blocked
    /// on backpressure; avoiding hold-forever admission is the policy's job,
    /// as with mstar's `max_concurrent_requests` gate). idle ≠ all finished:
    /// a request awaiting a policy decision after WalkDone is also idle.
    pub fn idle(&self) -> bool {
        self.inflight.is_empty()
            && self.ready_entries().is_empty()
            && !self.has_deliverable_chunk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VJEPA2_WALKS: &str = r#"{
        "prefill_video": {
            "kind": "sequential",
            "sections": [
                {"kind": "node", "name": "video_encoder",
                 "input_names": ["video_frames"],
                 "outputs": [{"next_node": "predictor", "name": "encoder_hidden"}]},
                {"kind": "node", "name": "predictor",
                 "input_names": ["encoder_hidden"],
                 "outputs": [{"next_node": "EMIT_TO_CLIENT", "name": "predicted_hidden",
                              "persist": true, "output_modality": "video"}]}
            ]
        }
    }"#;

    fn tref(rt: &mut Runtime) -> TensorRef {
        let id = rt.new_uuid();
        TensorRef::new(id, vec![1], "float32")
    }

    #[test]
    fn complete_batch_after_abort_is_noop_not_panic() {
        // A request aborted while a batch is in flight: finish_request nulls
        // the walk without removing the batch from `inflight`. The late
        // complete_batch must be a no-op, not a `.expect("walk")` panic.
        let mut rt = Runtime::from_walks_json(VJEPA2_WALKS).unwrap();
        let rid = rt.add_request();
        let frames = tref(&mut rt);
        rt.start_walk(
            rid,
            "prefill_video",
            vec![IncomingInput {
                node: "video_encoder".into(),
                name: "video_frames".into(),
                tensors: vec![frames],
            }],
        )
        .unwrap();
        let batch = rt.next_batch(8).unwrap().unwrap(); // batch now in flight

        rt.finish_request(rid).unwrap(); // client abort mid-flight

        let hidden = tref(&mut rt);
        let events = rt
            .complete_batch(
                batch.batch_id,
                BTreeMap::from([(
                    rid,
                    BTreeMap::from([("encoder_hidden".to_string(), vec![hidden])]),
                )]),
            )
            .expect("must not error");
        assert!(events.is_empty(), "aborted request produces no events");
        assert!(rt.idle());
    }

    #[test]
    fn single_request_end_to_end() {
        let mut rt = Runtime::from_walks_json(VJEPA2_WALKS).unwrap();
        let rid = rt.add_request();
        let frames = tref(&mut rt);
        rt.start_walk(
            rid,
            "prefill_video",
            vec![IncomingInput {
                node: "video_encoder".into(),
                name: "video_frames".into(),
                tensors: vec![frames],
            }],
        )
        .unwrap();

        let batch = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(batch.node, "video_encoder");
        assert_eq!(batch.inputs.len(), 1);

        let hidden = tref(&mut rt);
        let events = rt
            .complete_batch(
                batch.batch_id,
                BTreeMap::from([(
                    rid,
                    BTreeMap::from([("encoder_hidden".to_string(), vec![hidden])]),
                )]),
            )
            .unwrap();
        assert!(events.is_empty());

        let batch = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(batch.node, "predictor");
        let pred = tref(&mut rt);
        let events = rt
            .complete_batch(
                batch.batch_id,
                BTreeMap::from([(
                    rid,
                    BTreeMap::from([("predicted_hidden".to_string(), vec![pred.clone()])]),
                )]),
            )
            .unwrap();

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::Emission { request_id, name, .. }
            if *request_id == rid && name == "predicted_hidden"));
        assert!(matches!(&events[1], Event::WalkDone { request_id, walk, fwd_index, persist, partition, stream_done }
            if *request_id == rid && walk == "prefill_video" && *fwd_index == 1
               && persist.len() == 1 && partition == DEFAULT_PARTITION && !stream_done));

        assert!(rt.idle());
        rt.finish_request(rid).unwrap();
        assert!(rt.next_batch(8).unwrap().is_none());
    }

    #[test]
    fn two_requests_batch_together() {
        let mut rt = Runtime::from_walks_json(VJEPA2_WALKS).unwrap();
        let mut rids = Vec::new();
        for _ in 0..2 {
            let rid = rt.add_request();
            let frames = tref(&mut rt);
            rt.start_walk(
                rid,
                "prefill_video",
                vec![IncomingInput {
                    node: "video_encoder".into(),
                    name: "video_frames".into(),
                    tensors: vec![frames],
                }],
            )
            .unwrap();
            rids.push(rid);
        }
        let batch = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(batch.node, "video_encoder");
        assert_eq!(batch.inputs.len(), 2, "continuous batching across requests");

        let outs: BTreeMap<_, _> = rids
            .iter()
            .map(|&rid| {
                let h = tref(&mut rt);
                (
                    rid,
                    BTreeMap::from([("encoder_hidden".to_string(), vec![h])]),
                )
            })
            .collect();
        rt.complete_batch(batch.batch_id, outs).unwrap();

        let batch = rt.next_batch(1).unwrap().unwrap();
        assert_eq!(batch.node, "predictor");
        assert_eq!(batch.inputs.len(), 1, "max_batch_size clamps");
    }

    #[test]
    fn walk_restart_after_walk_done() {
        // Same walk can be re-seeded after completion (decode-style rounds).
        let mut rt = Runtime::from_walks_json(VJEPA2_WALKS).unwrap();
        let rid = rt.add_request();
        for round in 1..=2u64 {
            let frames = tref(&mut rt);
            rt.start_walk(
                rid,
                "prefill_video",
                vec![IncomingInput {
                    node: "video_encoder".into(),
                    name: "video_frames".into(),
                    tensors: vec![frames],
                }],
            )
            .unwrap();
            let b = rt.next_batch(8).unwrap().unwrap();
            let h = tref(&mut rt);
            rt.complete_batch(
                b.batch_id,
                BTreeMap::from([(rid, BTreeMap::from([("encoder_hidden".to_string(), vec![h])]))]),
            )
            .unwrap();
            let b = rt.next_batch(8).unwrap().unwrap();
            let p = tref(&mut rt);
            let events = rt
                .complete_batch(
                    b.batch_id,
                    BTreeMap::from([(
                        rid,
                        BTreeMap::from([("predicted_hidden".to_string(), vec![p])]),
                    )]),
                )
                .unwrap();
            let done = events.iter().find_map(|e| match e {
                Event::WalkDone { fwd_index, .. } => Some(*fwd_index),
                _ => None,
            });
            assert_eq!(done, Some(round));
        }
    }

    const PI05_LIKE_WALKS: &str = r#"{
        "prefill": {
            "kind": "sequential",
            "sections": [
                {"kind": "node", "name": "vit_encoder",
                 "input_names": ["image_inputs"],
                 "outputs": [{"next_node": "LLM", "name": "img_emb"}]},
                {"kind": "node", "name": "LLM",
                 "input_names": ["img_emb", "text_inputs"],
                 "outputs": []}
            ]
        },
        "action_gen": {
            "kind": "loop",
            "name": "flow",
            "max_iters": 3,
            "body": {"kind": "node", "name": "LLM",
                     "input_names": ["noisy_actions", "timestep_index"],
                     "outputs": [{"next_node": "LLM", "name": "noisy_actions"},
                                 {"next_node": "LLM", "name": "timestep_index"}]},
            "outputs": [{"next_node": "EMIT_TO_CLIENT", "name": "noisy_actions",
                         "persist": true, "output_modality": "action"}]
        }
    }"#;

    fn kv_runtime(num_pages: u32) -> Runtime {
        let mut rt = Runtime::from_walks_json(PI05_LIKE_WALKS).unwrap();
        rt.configure_kv(
            vec![KvCacheConfig {
                label: "LLM".into(),
                num_pages,
                page_size: 16,
            }],
            BTreeMap::from([("LLM".to_string(), "LLM".to_string())]),
        );
        rt
    }

    fn drive_prefill(rt: &mut Runtime, rid: u64, prefix_tokens: u64) {
        rt.start_walk_with_kv(
            rid,
            "prefill",
            vec![
                IncomingInput {
                    node: "vit_encoder".into(),
                    name: "image_inputs".into(),
                    tensors: vec![TensorRef::new(rid * 100 + 1, vec![1], "f32")],
                },
                IncomingInput {
                    node: "LLM".into(),
                    name: "text_inputs".into(),
                    tensors: vec![TensorRef::new(rid * 100 + 2, vec![1], "f32")],
                },
            ],
            BTreeMap::from([("LLM".to_string(), prefix_tokens)]),
            BTreeMap::new(),
        )
        .unwrap();
    }

    #[test]
    fn pi05_shaped_prefill_then_action_loop() {
        let mut rt = kv_runtime(8);
        let rid = rt.add_request();
        drive_prefill(&mut rt, rid, 40); // 40 tokens -> 3 pages

        // vit_encoder is not a KV node: no kv view.
        let b = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(b.node, "vit_encoder");
        assert!(b.kv.is_empty());
        rt.complete_batch(
            b.batch_id,
            BTreeMap::from([(
                rid,
                BTreeMap::from([("img_emb".to_string(), vec![TensorRef::new(9, vec![1], "f32")])]),
            )]),
        )
        .unwrap();

        // LLM prefill: kv view with fresh pages, seq_pos 0, append 40.
        let b = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(b.node, "LLM");
        let view = &b.kv[&rid];
        assert_eq!((view.seq_pos, view.append_len, view.pages.len()), (0, 40, 3));
        let events = rt
            .complete_batch(b.batch_id, BTreeMap::from([(rid, BTreeMap::new())]))
            .unwrap();
        assert!(matches!(events.last(), Some(Event::WalkDone { walk, .. }) if walk == "prefill"));
        assert_eq!(rt.kv_state(rid, "LLM").seq_pos, 40);

        // action_gen: read-only loop over the cached prefix.
        rt.start_walk_with_kv(
            rid,
            "action_gen",
            vec![
                IncomingInput {
                    node: "LLM".into(),
                    name: "noisy_actions".into(),
                    tensors: vec![TensorRef::new(11, vec![50, 32], "f32")],
                },
                IncomingInput {
                    node: "LLM".into(),
                    name: "timestep_index".into(),
                    tensors: vec![TensorRef::new(12, vec![1], "i64")],
                },
            ],
            BTreeMap::from([("LLM".to_string(), 0)]),
            BTreeMap::from([("LLM".to_string(), 8)]),
        )
        .unwrap();
        for iter in 0..3u64 {
            let b = rt.next_batch(8).unwrap().unwrap();
            assert_eq!(b.node, "LLM");
            let view = &b.kv[&rid];
            assert_eq!((view.seq_pos, view.append_len), (40, 0), "iter {iter}");
            let events = rt
                .complete_batch(
                    b.batch_id,
                    BTreeMap::from([(
                        rid,
                        BTreeMap::from([
                            (
                                "noisy_actions".to_string(),
                                vec![TensorRef::new(20 + iter, vec![50, 32], "f32")],
                            ),
                            (
                                "timestep_index".to_string(),
                                vec![TensorRef::new(30 + iter, vec![1], "i64")],
                            ),
                        ]),
                    )]),
                )
                .unwrap();
            if iter == 2 {
                assert!(events.iter().any(|e| matches!(e, Event::Emission { .. })));
            }
        }
        assert_eq!(rt.kv_state(rid, "LLM").seq_pos, 40, "read-only loop");
        rt.finish_request(rid).unwrap();
    }

    #[test]
    fn kv_oom_holds_request_until_pages_free() {
        let mut rt = kv_runtime(4); // 64 tokens capacity
        let r1 = rt.add_request();
        let r2 = rt.add_request();
        drive_prefill(&mut rt, r1, 64); // consumes all pages at LLM step
        drive_prefill(&mut rt, r2, 16);

        // Drive both vit_encoders + r1's LLM prefill.
        let mut llm_done = 0;
        while let Some(b) = rt.next_batch(8).unwrap() {
            let outs: BTreeMap<_, _> = b
                .inputs
                .keys()
                .map(|&rid| {
                    let named = if b.node == "vit_encoder" {
                        BTreeMap::from([(
                            "img_emb".to_string(),
                            vec![TensorRef::new(rid * 7, vec![1], "f32")],
                        )])
                    } else {
                        BTreeMap::new()
                    };
                    (rid, named)
                })
                .collect();
            if b.node == "LLM" {
                assert_eq!(
                    b.inputs.keys().copied().collect::<Vec<_>>(),
                    vec![r1],
                    "r2's LLM must be held: r1 holds every page"
                );
                llm_done += 1;
            }
            rt.complete_batch(b.batch_id, outs).unwrap();
        }
        assert_eq!(llm_done, 1);
        assert!(rt.idle(), "r2 held on KV backpressure -> idle");

        // Freeing r1 releases pages; r2 becomes schedulable.
        rt.finish_request(r1).unwrap();
        let b = rt.next_batch(8).unwrap().unwrap();
        assert_eq!(b.node, "LLM");
        assert_eq!(b.kv[&r2].append_len, 16);
    }

    /// Orpheus-shaped spec: LLM partition (prefill + decode loop, each pass
    /// streaming `new_token` into SNAC) + self-triggered SNAC partition
    /// (snac_chunk consumes a 28-token window, stride 7, emits audio).
    const ORPHEUS_LIKE_SPEC: &str = r#"{
        "walks": {
            "prefill": {"kind": "node", "name": "LLM",
                "input_names": ["text_inputs"],
                "outputs": [
                    {"next_node": "EMPTY_DESTINATION", "name": "new_token", "persist": true},
                    {"next_node": "snac_decoder", "name": "new_token",
                     "target_partition": "SNAC"}
                ]},
            "decode": {"kind": "loop", "name": "decode_loop", "max_iters": 100,
                "body": {"kind": "node", "name": "LLM",
                    "input_names": ["text_inputs"],
                    "outputs": [
                        {"next_node": "LLM", "name": "text_inputs"},
                        {"next_node": "snac_decoder", "name": "new_token",
                         "target_partition": "SNAC"}
                    ]}},
            "snac_chunk": {"kind": "node", "name": "snac_decoder",
                "input_names": ["new_token"],
                "outputs": [{"next_node": "EMIT_TO_CLIENT", "name": "audio_chunk",
                             "output_modality": "audio"}]}
        },
        "partitions": [
            {"name": "LLM", "walks": ["prefill", "decode"]},
            {"name": "SNAC", "walks": ["snac_chunk"]}
        ],
        "connections": [
            {"from": "LLM", "to": "SNAC", "edge_name": "new_token",
             "policy": {"kind": "sliding_window", "window": 28, "stride": 7}}
        ]
    }"#;

    /// Drive the orpheus-shaped spec with an in-test executor/policy:
    /// decode emits `total_tokens` tokens then EOS-stops the loop; SNAC
    /// windows must arrive with the right contents; final chunk consumed
    /// before request done.
    #[test]
    fn orpheus_shaped_streaming_end_to_end() {
        let total_tokens: u64 = 40; // 1 prefill + 39 decode
        let mut rt = Runtime::from_spec_json(ORPHEUS_LIKE_SPEC).unwrap();
        rt.configure_kv(
            vec![KvCacheConfig {
                label: "main".into(),
                num_pages: 8,
                page_size: 16,
            }],
            BTreeMap::from([("LLM".to_string(), "main".to_string())]),
        );
        let rid = rt.add_request();
        // Seed both partitions (mstar kicks off all partitions at ingest;
        // SNAC's walk registers with empty inputs and self-triggers).
        rt.start_walk_with_kv(
            rid,
            "prefill",
            vec![IncomingInput {
                node: "LLM".into(),
                name: "text_inputs".into(),
                tensors: vec![TensorRef::new(1000, vec![5], "i64")],
            }],
            BTreeMap::from([("main".to_string(), 5)]),
            BTreeMap::new(),
        )
        .unwrap();
        rt.start_walk(rid, "snac_chunk", vec![]).unwrap();

        let mut next_token: u64 = 0; // uuid == token index
        let mut llm_done = false;
        let mut snac_windows: Vec<Vec<u64>> = Vec::new();
        let mut request_done = false;
        let mut audio_chunks = 0u64;

        while !request_done {
            let Some(batch) = rt.next_batch(8).unwrap() else {
                panic!(
                    "stalled: llm_done={llm_done} windows={} audio={audio_chunks}",
                    snac_windows.len()
                );
            };
            let outs: BTreeMap<u64, BTreeMap<String, Vec<TensorRef>>> = match batch.node.as_str() {
                "LLM" => {
                    let tok = TensorRef::new(next_token, vec![1], "i64");
                    next_token += 1;
                    // EOS: stop the decode loop after emitting the last token.
                    if batch.walk == "decode" && next_token == total_tokens {
                        rt.signal_loop_finish(rid, "decode_loop").unwrap();
                    }
                    let mut named = BTreeMap::new();
                    named.insert("new_token".to_string(), vec![tok.clone()]);
                    if batch.walk == "decode" {
                        named.insert("text_inputs".to_string(), vec![tok]);
                    }
                    BTreeMap::from([(rid, named)])
                }
                "snac_decoder" => {
                    let window: Vec<u64> = batch.inputs[&rid]["new_token"]
                        .iter()
                        .map(|t| t.uuid)
                        .collect();
                    snac_windows.push(window);
                    BTreeMap::from([(
                        rid,
                        BTreeMap::from([(
                            "audio_chunk".to_string(),
                            vec![TensorRef::new(9000 + audio_chunks, vec![2048], "i16")],
                        )]),
                    )])
                }
                other => panic!("unexpected node {other}"),
            };
            for event in rt.complete_batch(batch.batch_id, outs).unwrap() {
                match event {
                    Event::Emission { partition, name, .. } => {
                        assert_eq!((partition.as_str(), name.as_str()), ("SNAC", "audio_chunk"));
                        audio_chunks += 1;
                    }
                    Event::WalkDone {
                        partition,
                        walk,
                        stream_done,
                        ..
                    } => match partition.as_str() {
                        "LLM" => {
                            assert!(!stream_done);
                            if walk == "prefill" {
                                // Policy: prefill -> decode, seeded with the
                                // first sampled token; 1 KV append per step.
                                rt.start_walk_with_kv(
                                    rid,
                                    "decode",
                                    vec![IncomingInput {
                                        node: "LLM".into(),
                                        name: "text_inputs".into(),
                                        tensors: vec![TensorRef::new(0, vec![1], "i64")],
                                    }],
                                    BTreeMap::from([("main".to_string(), 1)]),
                                    BTreeMap::new(),
                                )
                                .unwrap();
                            } else {
                                llm_done = true;
                                rt.finish_partition(rid, "LLM").unwrap();
                            }
                        }
                        "SNAC" => {
                            if stream_done {
                                request_done = rt.finish_partition(rid, "SNAC").unwrap();
                            } else {
                                rt.start_walk(rid, "snac_chunk", vec![]).unwrap();
                            }
                        }
                        other => panic!("unexpected partition {other}"),
                    },
                }
            }
        }
        rt.finish_request(rid).unwrap();

        // 40 tokens, window 28 stride 7: pops at [0..28),[7..35), then
        // producer-done flush [14..40) (26 items, final).
        assert_eq!(snac_windows.len(), 3, "windows: {snac_windows:?}");
        assert_eq!(snac_windows[0], (0..28).collect::<Vec<u64>>());
        assert_eq!(snac_windows[1], (7..35).collect::<Vec<u64>>());
        assert_eq!(snac_windows[2], (14..40).collect::<Vec<u64>>());
        assert_eq!(audio_chunks, 3, "final chunk emitted before teardown");
        assert!(llm_done);
        assert!(rt.idle());
        // 44 tokens of KV committed (5 prefill + 39 decode appends).
        assert_eq!(rt.kv_state(rid, "main").seq_pos, 0, "freed on finish");
    }

    #[test]
    fn error_paths() {
        let mut rt = Runtime::from_walks_json(VJEPA2_WALKS).unwrap();
        assert!(matches!(
            rt.start_walk(99, "prefill_video", vec![]),
            Err(RuntimeError::UnknownRequest(99))
        ));
        let rid = rt.add_request();
        assert!(matches!(
            rt.start_walk(rid, "nope", vec![]),
            Err(RuntimeError::Core(_))
        ));
        rt.start_walk(rid, "prefill_video", vec![]).unwrap();
        assert!(matches!(
            rt.start_walk(rid, "prefill_video", vec![]),
            Err(RuntimeError::WalkActive(_, _))
        ));
        assert!(matches!(
            rt.complete_batch(42, BTreeMap::new()),
            Err(RuntimeError::UnknownBatch(42))
        ));
    }
}
