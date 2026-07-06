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

use std::collections::BTreeMap;

use mstar_core::{
    CompiledWalk, CoreError, IncomingInput, RouteEvent, TensorRef, WalkSet, WalkState,
};
use mstar_sched::{MicroScheduler, ReadyEntry};
use thiserror::Error;

pub use kv::{KvCacheConfig, KvManager, KvRequestState, KvView};

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("unknown request {0}")]
    UnknownRequest(u64),
    #[error("unknown batch {0}")]
    UnknownBatch(u64),
    #[error("request {0} already has an active walk")]
    WalkActive(u64),
    #[error("request {0} is finished")]
    RequestFinished(u64),
    #[error("batch {batch}: missing outputs for request {request}")]
    MissingOutputs { batch: u64, request: u64 },
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// What the runtime tells the driver after routing a completed batch.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A tensor reached `EMIT_TO_CLIENT` — postprocess and stream it out.
    Emission {
        request_id: u64,
        name: String,
        modality: Option<String>,
        tensors: Vec<TensorRef>,
    },
    /// A forward pass (one walk) finished. `persist` carries the request's
    /// accumulated persist signals (latest value per name — mstar's
    /// conductor-side `persist_signals`). The policy must now call
    /// `start_walk` or `finish_request`.
    WalkDone {
        request_id: u64,
        walk: String,
        fwd_index: u64,
        persist: Vec<(String, Vec<TensorRef>)>,
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

#[derive(Debug)]
struct RequestState {
    walk: Option<WalkState>,
    fwd_index: u64,
    persist: BTreeMap<String, Vec<TensorRef>>,
    finished: bool,
    /// label -> tokens appended per execution of that label's KV node in the
    /// current walk (0 = read-only). Declared by the policy at start_walk.
    kv_appends: BTreeMap<String, u64>,
    /// label -> transient scratch tokens past the committed sequence that
    /// each execution also needs pages for (pi05's flow-loop suffix).
    kv_scratch: BTreeMap<String, u64>,
}

#[derive(Debug)]
struct InflightBatch {
    node: String,
    request_ids: Vec<u64>,
    /// (request, label, append) to advance when the batch completes.
    kv_advances: Vec<(u64, String, u64)>,
}

/// The in-process M* runtime: walk registry + per-request state + scheduler.
#[derive(Debug)]
pub struct Runtime {
    walks: WalkSet,
    requests: BTreeMap<u64, RequestState>,
    scheduler: MicroScheduler,
    inflight: BTreeMap<u64, InflightBatch>,
    kv: Option<KvManager>,
    next_request_id: u64,
    next_batch_id: u64,
    next_uuid: u64,
}

impl Runtime {
    /// Build from the walk-set JSON (`{"walk_name": <Section>, ...}`).
    pub fn from_walks_json(json: &str) -> Result<Self> {
        Ok(Self {
            walks: WalkSet::from_json(json)?,
            requests: BTreeMap::new(),
            scheduler: MicroScheduler::new(),
            inflight: BTreeMap::new(),
            kv: None,
            next_request_id: 0,
            next_batch_id: 0,
            next_uuid: 0,
        })
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
                walk: None,
                fwd_index: 0,
                persist: BTreeMap::new(),
                finished: false,
                kv_appends: BTreeMap::new(),
                kv_scratch: BTreeMap::new(),
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
    pub fn start_walk_with_kv(
        &mut self,
        request_id: u64,
        walk_name: &str,
        inputs: Vec<IncomingInput>,
        kv_appends: BTreeMap<String, u64>,
        kv_scratch: BTreeMap<String, u64>,
    ) -> Result<()> {
        let graph: std::sync::Arc<CompiledWalk> = self.walks.get(walk_name)?;
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        if req.finished {
            return Err(RuntimeError::RequestFinished(request_id));
        }
        if req.walk.is_some() {
            return Err(RuntimeError::WalkActive(request_id));
        }
        let mut walk = WalkState::new(graph);
        walk.seed(inputs)?;
        req.walk = Some(walk);
        req.kv_appends = kv_appends;
        req.kv_scratch = kv_scratch;
        Ok(())
    }

    /// Signal a loop to finish for one request (mstar's `STOP_LOOPS`, e.g.
    /// EOS detected by the data plane's check_stop).
    pub fn signal_loop_finish(&mut self, request_id: u64, loop_name: &str) -> Result<()> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        let walk = req.walk.as_mut().ok_or(RuntimeError::UnknownRequest(request_id))?;
        walk.signal_loop_finish(loop_name)?;
        Ok(())
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
            if let Some(walk) = &req.walk {
                for node in walk.ready_nodes() {
                    if let Some(kv) = &self.kv {
                        if let Some(label) = kv.label_for_node(&node) {
                            let append = req.kv_appends.get(label).copied().unwrap_or(0);
                            let scratch = req.kv_scratch.get(label).copied().unwrap_or(0);
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

    pub fn next_batch(&mut self, max_batch_size: usize) -> Result<Option<Batch>> {
        let ready = self.ready_entries();
        let Some(plan) = self.scheduler.next_batch(&ready, max_batch_size) else {
            return Ok(None);
        };

        let kv_label = self
            .kv
            .as_ref()
            .and_then(|kv| kv.label_for_node(&plan.node))
            .map(str::to_string);

        let mut inputs = BTreeMap::new();
        let mut kv_views = BTreeMap::new();
        let mut kv_advances = Vec::new();
        let mut scheduled_rids = Vec::new();
        for &rid in &plan.request_ids {
            let req = self.requests.get_mut(&rid).expect("rid from ready scan");
            // Reserve KV pages FIRST: earlier requests in this same batch may
            // have consumed the pages that made `can_reserve` true at scan
            // time. A request that no longer fits is simply left out of the
            // batch (its node stays ready) — mstar's hold-on-OOM behavior.
            // The first request always fits: nothing allocated since the scan.
            if let Some(label) = &kv_label {
                let append = req.kv_appends.get(label).copied().unwrap_or(0);
                let scratch = req.kv_scratch.get(label).copied().unwrap_or(0);
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
            let walk = req.walk.as_mut().expect("walk present in ready scan");
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
        if let Some(kv) = &mut self.kv {
            for (rid, label, append) in &inflight.kv_advances {
                kv.advance(*rid, label, *append);
            }
        }
        let mut events = Vec::new();
        for &rid in &inflight.request_ids {
            let out = outputs
                .get(&rid)
                .ok_or(RuntimeError::MissingOutputs {
                    batch: batch_id,
                    request: rid,
                })?
                .clone();
            let req = self.requests.get_mut(&rid).expect("inflight rid");
            let walk = req.walk.as_mut().expect("inflight walk");
            let result = walk.complete_node(&inflight.node, out)?;
            for ev in result.events {
                match ev {
                    RouteEvent::Emission {
                        name,
                        modality,
                        tensors,
                    } => events.push(Event::Emission {
                        request_id: rid,
                        name,
                        modality,
                        tensors,
                    }),
                    RouteEvent::Persist { name, tensors } => {
                        req.persist.insert(name, tensors);
                    }
                }
            }
            if result.walk_done {
                let walk_name = walk.walk_name().to_string();
                req.walk = None;
                req.fwd_index += 1;
                events.push(Event::WalkDone {
                    request_id: rid,
                    walk: walk_name,
                    fwd_index: req.fwd_index,
                    persist: req
                        .persist
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                });
            }
        }
        Ok(events)
    }

    pub fn finish_request(&mut self, request_id: u64) -> Result<()> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        req.finished = true;
        req.walk = None;
        req.persist.clear();
        req.kv_appends.clear();
        req.kv_scratch.clear();
        if let Some(kv) = &mut self.kv {
            kv.free_request(request_id);
        }
        Ok(())
    }

    /// True when no request has schedulable or in-flight work — the same
    /// gate `next_batch` uses, so a KV-held request counts as idle (blocked
    /// on backpressure; avoiding hold-forever admission is the policy's job,
    /// as with mstar's `max_concurrent_requests` gate). idle ≠ all finished:
    /// a request awaiting a policy decision after WalkDone is also idle.
    pub fn idle(&self) -> bool {
        self.inflight.is_empty() && self.ready_entries().is_empty()
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
        assert!(matches!(&events[1], Event::WalkDone { request_id, walk, fwd_index, persist }
            if *request_id == rid && walk == "prefill_video" && *fwd_index == 1
               && persist.len() == 1));

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
            Err(RuntimeError::WalkActive(_))
        ));
        assert!(matches!(
            rt.complete_batch(42, BTreeMap::new()),
            Err(RuntimeError::UnknownBatch(42))
        ));
    }
}
