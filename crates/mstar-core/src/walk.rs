use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{CoreError, Result};
use crate::graph::{CompiledWalk, EdgeSpec, EMIT_TO_CLIENT, EMPTY_DESTINATION};
use crate::tensor::TensorRef;

/// An external input injected into a walk (from the policy's forward-pass
/// args, or — in a future multi-worker world — from a peer worker).
#[derive(Debug, Clone)]
pub struct IncomingInput {
    pub node: String,
    pub name: String,
    pub tensors: Vec<TensorRef>,
}

/// Something the walk state machine routed outward while completing a node.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteEvent {
    /// Edge destined for the client (`EMIT_TO_CLIENT`).
    Emission {
        name: String,
        modality: Option<String>,
        tensors: Vec<TensorRef>,
    },
    /// Edge marked `persist: true` — a walk output for the policy.
    Persist {
        name: String,
        tensors: Vec<TensorRef>,
    },
    /// Streaming edge: goes to the stream buffer of the connection into
    /// `target_partition` (the runtime owns buffers; walks don't see them).
    Stream {
        name: String,
        target_partition: String,
        tensors: Vec<TensorRef>,
    },
}

#[derive(Debug, Default)]
pub struct CompletionResult {
    pub events: Vec<RouteEvent>,
    pub walk_done: bool,
}

#[derive(Debug, Default)]
struct NodeState {
    /// Inputs received for the current iteration.
    current: BTreeMap<String, Vec<TensorRef>>,
    /// Inputs buffered for the next loop iteration (mstar's `ready_next_iter`):
    /// filled when an input arrives at a node that already has that input or
    /// already ran this iteration — i.e. loop-back edges.
    next_iter: BTreeMap<String, Vec<TensorRef>>,
    completed: bool,
    scheduled: bool,
}

#[derive(Debug, Default)]
struct LoopState {
    curr_iter: u32,
    finish_signal: bool,
    terminated: bool,
    /// Latest value per output name (snapshot for `outputs`).
    last_values: BTreeMap<String, Vec<TensorRef>>,
    /// Per-iteration values per output name (for `accumulated_outputs`).
    accumulated: BTreeMap<String, Vec<TensorRef>>,
    /// External inputs into loop members, re-injected on every advance
    /// (mstar's `_ingested_external_inputs`).
    external_inputs: Vec<(String, String, Vec<TensorRef>)>,
}

/// Per-request state machine over one compiled walk graph. Ports the runtime
/// behavior of `GraphNode`/`Loop`/`WorkerGraphIO` from `mstar/graph/`.
#[derive(Debug)]
pub struct WalkState {
    graph: Arc<CompiledWalk>,
    nodes: BTreeMap<String, NodeState>,
    loops: Vec<LoopState>,
}

impl WalkState {
    pub fn new(graph: Arc<CompiledWalk>) -> Self {
        let nodes = graph
            .nodes
            .keys()
            .map(|name| (name.clone(), NodeState::default()))
            .collect();
        let loops = graph.loops.iter().map(|_| LoopState::default()).collect();
        Self {
            graph,
            nodes,
            loops,
        }
    }

    pub fn walk_name(&self) -> &str {
        &self.graph.name
    }

    /// Inject external inputs (walk seeding). External inputs into loop
    /// members are recorded for re-injection on each loop advance.
    pub fn seed(&mut self, inputs: Vec<IncomingInput>) -> Result<()> {
        for input in inputs {
            if !self.graph.nodes.contains_key(&input.node) {
                return Err(CoreError::UnknownNode(
                    input.node.clone(),
                    self.graph.name.clone(),
                ));
            }
            if let Some(&loop_idx) = self.graph.node_loop.get(&input.node) {
                self.loops[loop_idx].external_inputs.push((
                    input.node.clone(),
                    input.name.clone(),
                    input.tensors.clone(),
                ));
            }
            self.ingest(&input.node, &input.name, input.tensors);
        }
        Ok(())
    }

    fn ingest(&mut self, node: &str, name: &str, tensors: Vec<TensorRef>) {
        let state = self.nodes.get_mut(node).expect("node validated");
        if state.completed || state.scheduled || state.current.contains_key(name) {
            state.next_iter.insert(name.to_string(), tensors);
        } else {
            state.current.insert(name.to_string(), tensors);
        }
    }

    /// Nodes whose inputs are all present and are not running/finished.
    pub fn ready_nodes(&self) -> Vec<String> {
        self.graph
            .nodes
            .iter()
            .filter(|(name, spec)| {
                let st = &self.nodes[*name];
                !st.completed
                    && !st.scheduled
                    && spec.input_names.iter().all(|i| st.current.contains_key(i))
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Mark a ready node as scheduled and hand back its input tensors for
    /// the data plane to execute with.
    pub fn take_node_inputs(&mut self, node: &str) -> Result<BTreeMap<String, Vec<TensorRef>>> {
        let spec = self
            .graph
            .nodes
            .get(node)
            .ok_or_else(|| CoreError::UnknownNode(node.to_string(), self.graph.name.clone()))?;
        let st = self.nodes.get_mut(node).expect("node validated");
        let missing: Vec<String> = spec
            .input_names
            .iter()
            .filter(|i| !st.current.contains_key(*i))
            .cloned()
            .collect();
        if st.completed || st.scheduled || !missing.is_empty() {
            return Err(CoreError::NodeNotReady {
                node: node.to_string(),
                missing,
            });
        }
        st.scheduled = true;
        Ok(st.current.clone())
    }

    fn stream_target(&self, input_name: &str) -> Option<&str> {
        self.graph.nodes.iter().find_map(|(name, spec)| {
            let st = &self.nodes[name];
            (spec.input_names.contains(input_name)
                && !st.completed
                && !st.scheduled
                && !st.current.contains_key(input_name))
            .then_some(name.as_str())
        })
    }

    /// Can a stream chunk for `input_name` be ingested right now? (mstar's
    /// `process_new_streaming_inputs` with `can_buffer=False` — chunks are
    /// only delivered into a node that is waiting for exactly this input.)
    pub fn can_accept_input(&self, input_name: &str) -> bool {
        self.stream_target(input_name).is_some()
    }

    /// Ingest a stream chunk (the window, in order). Caller must have
    /// checked `can_accept_input`.
    pub fn ingest_stream_input(&mut self, input_name: &str, tensors: Vec<TensorRef>) {
        let node = self
            .stream_target(input_name)
            .expect("caller checked can_accept_input")
            .to_string();
        self.ingest(&node, input_name, tensors);
    }

    /// Record a finish signal for a loop (mstar's `check_stop -> STOP_LOOPS`).
    /// Takes effect when the current iteration completes.
    pub fn signal_loop_finish(&mut self, loop_name: &str) -> Result<()> {
        let idx = self.graph.loop_index(loop_name)?;
        self.loops[idx].finish_signal = true;
        Ok(())
    }

    /// The data plane finished executing `node`; route its named outputs.
    pub fn complete_node(
        &mut self,
        node: &str,
        outputs: BTreeMap<String, Vec<TensorRef>>,
    ) -> Result<CompletionResult> {
        let spec = self
            .graph
            .nodes
            .get(node)
            .cloned()
            .ok_or_else(|| CoreError::UnknownNode(node.to_string(), self.graph.name.clone()))?;
        {
            let st = self.nodes.get_mut(node).expect("node validated");
            if !st.scheduled {
                return Err(CoreError::NodeNotScheduled(node.to_string()));
            }
            st.scheduled = false;
            st.completed = true;
        }

        let mut result = CompletionResult::default();
        let loop_idx = self.graph.node_loop.get(node).copied();

        // Capture loop output values produced by this node.
        if let Some(idx) = loop_idx {
            let loop_spec = &self.graph.loops[idx];
            let capture: Vec<(String, bool)> = loop_spec
                .outputs
                .iter()
                .map(|e| (e.name.clone(), false))
                .chain(
                    loop_spec
                        .accumulated_outputs
                        .iter()
                        .map(|e| (e.name.clone(), true)),
                )
                .collect();
            let lst = &mut self.loops[idx];
            for (name, accumulate) in capture {
                if let Some(tensors) = outputs.get(&name) {
                    if accumulate {
                        lst.accumulated
                            .entry(name.clone())
                            .or_default()
                            .extend(tensors.iter().cloned());
                    }
                    lst.last_values.insert(name, tensors.clone());
                }
            }
        }

        // Route the node's own edges.
        for edge in &spec.outputs {
            let tensors = outputs.get(&edge.name).cloned().unwrap_or_default();
            Self::route_edge(&mut self.nodes, &mut result.events, edge, tensors);
        }

        // Loop iteration bookkeeping.
        if let Some(idx) = loop_idx {
            let all_members_done = self.graph.loops[idx]
                .members
                .iter()
                .all(|m| self.nodes[m].completed);
            if all_members_done {
                self.complete_loop_iter(idx, &mut result.events);
            }
        }

        result.walk_done = self.is_done();
        Ok(result)
    }

    /// mstar `Loop.complete_iter`: terminate on max_iters/finish signal,
    /// otherwise advance one iteration.
    fn complete_loop_iter(&mut self, idx: usize, events: &mut Vec<RouteEvent>) {
        let finished = {
            let lst = &self.loops[idx];
            lst.finish_signal || lst.curr_iter + 1 >= self.graph.loops[idx].max_iters
        };
        if finished {
            let loop_spec = self.graph.loops[idx].clone();
            let (last_values, accumulated) = {
                let lst = &mut self.loops[idx];
                lst.terminated = true;
                (
                    std::mem::take(&mut lst.last_values),
                    std::mem::take(&mut lst.accumulated),
                )
            };
            for edge in &loop_spec.outputs {
                let tensors = last_values.get(&edge.name).cloned().unwrap_or_default();
                Self::route_edge(&mut self.nodes, events, edge, tensors);
            }
            for edge in &loop_spec.accumulated_outputs {
                let tensors = accumulated.get(&edge.name).cloned().unwrap_or_default();
                Self::route_edge(&mut self.nodes, events, edge, tensors);
            }
        } else {
            // Advance: reset members, promote next-iter inputs, re-inject
            // external inputs (which never overwrite fresher loop-back values).
            let members: Vec<String> = self.graph.loops[idx].members.iter().cloned().collect();
            self.loops[idx].curr_iter += 1;
            for member in &members {
                let st = self.nodes.get_mut(member).expect("member validated");
                st.completed = false;
                st.current = std::mem::take(&mut st.next_iter);
            }
            let externals = self.loops[idx].external_inputs.clone();
            for (node, name, tensors) in externals {
                let st = self.nodes.get_mut(&node).expect("member validated");
                st.current.entry(name).or_insert(tensors);
            }
        }
    }

    fn route_edge(
        nodes: &mut BTreeMap<String, NodeState>,
        events: &mut Vec<RouteEvent>,
        edge: &EdgeSpec,
        tensors: Vec<TensorRef>,
    ) {
        if edge.persist {
            events.push(RouteEvent::Persist {
                name: edge.name.clone(),
                tensors: tensors.clone(),
            });
        }
        if let Some(partition) = &edge.target_partition {
            events.push(RouteEvent::Stream {
                name: edge.name.clone(),
                target_partition: partition.clone(),
                tensors,
            });
            return;
        }
        match edge.next_node.as_str() {
            EMIT_TO_CLIENT => events.push(RouteEvent::Emission {
                name: edge.name.clone(),
                modality: edge.output_modality.clone(),
                tensors,
            }),
            EMPTY_DESTINATION => {}
            dest => {
                let state = nodes.get_mut(dest).expect("edges validated at compile");
                if state.completed || state.scheduled || state.current.contains_key(&edge.name) {
                    state.next_iter.insert(edge.name.clone(), tensors);
                } else {
                    state.current.insert(edge.name.clone(), tensors);
                }
            }
        }
    }

    /// A walk is done when every node has completed and every loop has
    /// terminated. One walk completion == one forward pass, after which the
    /// policy picks the next walk (or finishes the request).
    pub fn is_done(&self) -> bool {
        self.graph
            .loops
            .iter()
            .enumerate()
            .all(|(i, _)| self.loops[i].terminated)
            && self
                .nodes
                .iter()
                .all(|(name, st)| match self.graph.node_loop.get(name) {
                    Some(&idx) => self.loops[idx].terminated,
                    None => st.completed,
                })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{LoopSpec, NodeSpec, Section, WalkSet};

    fn tref(uuid: u64) -> TensorRef {
        TensorRef::new(uuid, vec![1], "float32")
    }

    fn node(name: &str, inputs: &[&str], outputs: Vec<EdgeSpec>) -> Section {
        Section::Node(NodeSpec {
            name: name.to_string(),
            input_names: inputs.iter().map(|s| s.to_string()).collect(),
            outputs,
        })
    }

    fn edge(next: &str, name: &str) -> EdgeSpec {
        EdgeSpec {
            next_node: next.to_string(),
            name: name.to_string(),
            persist: false,
            output_modality: None,
            target_partition: None,
        }
    }

    fn emit(name: &str, modality: &str, persist: bool) -> EdgeSpec {
        EdgeSpec {
            next_node: EMIT_TO_CLIENT.to_string(),
            name: name.to_string(),
            persist,
            output_modality: Some(modality.to_string()),
            target_partition: None,
        }
    }

    /// vjepa2 `prefill_video`: video_frames -> [encoder] -> encoder_hidden
    /// -> [predictor] -> predicted_hidden -> EMIT_TO_CLIENT.
    fn vjepa2_like_walk() -> WalkState {
        let section = Section::Sequential {
            sections: vec![
                node(
                    "video_encoder",
                    &["video_frames"],
                    vec![edge("predictor", "encoder_hidden")],
                ),
                node(
                    "predictor",
                    &["encoder_hidden"],
                    vec![emit("predicted_hidden", "video", true)],
                ),
            ],
        };
        WalkState::new(Arc::new(
            CompiledWalk::compile("prefill_video", &section).unwrap(),
        ))
    }

    #[test]
    fn sequential_two_node_walk_end_to_end() {
        let mut walk = vjepa2_like_walk();
        walk.seed(vec![IncomingInput {
            node: "video_encoder".into(),
            name: "video_frames".into(),
            tensors: vec![tref(1)],
        }])
        .unwrap();

        assert_eq!(walk.ready_nodes(), vec!["video_encoder".to_string()]);
        let inputs = walk.take_node_inputs("video_encoder").unwrap();
        assert_eq!(inputs["video_frames"], vec![tref(1)]);
        assert!(walk.ready_nodes().is_empty(), "scheduled node not ready");

        let res = walk
            .complete_node(
                "video_encoder",
                BTreeMap::from([("encoder_hidden".to_string(), vec![tref(2)])]),
            )
            .unwrap();
        assert!(res.events.is_empty());
        assert!(!res.walk_done);

        assert_eq!(walk.ready_nodes(), vec!["predictor".to_string()]);
        walk.take_node_inputs("predictor").unwrap();
        let res = walk
            .complete_node(
                "predictor",
                BTreeMap::from([("predicted_hidden".to_string(), vec![tref(3)])]),
            )
            .unwrap();
        assert!(res.walk_done);
        assert_eq!(res.events.len(), 2); // persist + emission
        assert!(res.events.contains(&RouteEvent::Persist {
            name: "predicted_hidden".into(),
            tensors: vec![tref(3)],
        }));
        assert!(res.events.contains(&RouteEvent::Emission {
            name: "predicted_hidden".into(),
            modality: Some("video".into()),
            tensors: vec![tref(3)],
        }));
    }

    #[test]
    fn take_requires_ready_and_complete_requires_scheduled() {
        let mut walk = vjepa2_like_walk();
        assert!(matches!(
            walk.take_node_inputs("video_encoder"),
            Err(CoreError::NodeNotReady { .. })
        ));
        assert!(matches!(
            walk.complete_node("video_encoder", BTreeMap::new()),
            Err(CoreError::NodeNotScheduled(_))
        ));
    }

    /// pi05-style flow-matching loop: a single LLM node feeding itself
    /// noisy_actions/timestep for max_iters, then emitting the final actions.
    fn flow_loop_walk(max_iters: u32) -> WalkState {
        let body = node(
            "LLM",
            &["noisy_actions", "timestep_index"],
            vec![
                edge("LLM", "noisy_actions"),
                edge("LLM", "timestep_index"),
            ],
        );
        let section = Section::Loop(LoopSpec {
            name: "flow".into(),
            body: Box::new(body),
            max_iters,
            outputs: vec![emit("noisy_actions", "action", true)],
            accumulated_outputs: vec![],
        });
        WalkState::new(Arc::new(CompiledWalk::compile("action_gen", &section).unwrap()))
    }

    fn run_loop_iter(walk: &mut WalkState, out_uuid: u64) -> CompletionResult {
        walk.take_node_inputs("LLM").unwrap();
        walk.complete_node(
            "LLM",
            BTreeMap::from([
                ("noisy_actions".to_string(), vec![tref(out_uuid)]),
                ("timestep_index".to_string(), vec![tref(out_uuid + 1000)]),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn loop_runs_max_iters_and_emits_last_value() {
        let mut walk = flow_loop_walk(3);
        walk.seed(vec![
            IncomingInput {
                node: "LLM".into(),
                name: "noisy_actions".into(),
                tensors: vec![tref(10)],
            },
            IncomingInput {
                node: "LLM".into(),
                name: "timestep_index".into(),
                tensors: vec![tref(11)],
            },
        ])
        .unwrap();

        for iter in 0..3u64 {
            assert_eq!(walk.ready_nodes(), vec!["LLM".to_string()], "iter {iter}");
            let res = run_loop_iter(&mut walk, 100 + iter);
            if iter < 2 {
                assert!(!res.walk_done, "iter {iter} should continue");
                assert!(res.events.is_empty());
            } else {
                assert!(res.walk_done);
                assert!(res.events.contains(&RouteEvent::Emission {
                    name: "noisy_actions".into(),
                    modality: Some("action".into()),
                    tensors: vec![tref(102)], // last iteration's value
                }));
            }
        }
        assert!(walk.ready_nodes().is_empty());
    }

    #[test]
    fn loop_back_values_flow_between_iterations() {
        let mut walk = flow_loop_walk(2);
        walk.seed(vec![
            IncomingInput {
                node: "LLM".into(),
                name: "noisy_actions".into(),
                tensors: vec![tref(10)],
            },
            IncomingInput {
                node: "LLM".into(),
                name: "timestep_index".into(),
                tensors: vec![tref(11)],
            },
        ])
        .unwrap();

        run_loop_iter(&mut walk, 100);
        // Second iteration must see iteration 1's loop-back outputs, not the
        // re-injected seeds (externals must not overwrite loop-back values).
        let inputs = walk.take_node_inputs("LLM").unwrap();
        assert_eq!(inputs["noisy_actions"], vec![tref(100)]);
        assert_eq!(inputs["timestep_index"], vec![tref(1100)]);
    }

    #[test]
    fn finish_signal_terminates_loop_early() {
        let mut walk = flow_loop_walk(10);
        walk.seed(vec![
            IncomingInput {
                node: "LLM".into(),
                name: "noisy_actions".into(),
                tensors: vec![tref(10)],
            },
            IncomingInput {
                node: "LLM".into(),
                name: "timestep_index".into(),
                tensors: vec![tref(11)],
            },
        ])
        .unwrap();

        run_loop_iter(&mut walk, 100);
        walk.signal_loop_finish("flow").unwrap();
        let res = run_loop_iter(&mut walk, 200);
        assert!(res.walk_done, "finish signal should stop before max_iters");
        assert!(res.events.contains(&RouteEvent::Emission {
            name: "noisy_actions".into(),
            modality: Some("action".into()),
            tensors: vec![tref(200)],
        }));
    }

    #[test]
    fn accumulated_outputs_collect_every_iteration() {
        let body = node(
            "gen",
            &["state"],
            vec![edge("gen", "state")],
        );
        let section = Section::Loop(LoopSpec {
            name: "rollout".into(),
            body: Box::new(body),
            max_iters: 3,
            outputs: vec![],
            accumulated_outputs: vec![emit("state", "video", false)],
        });
        let mut walk = WalkState::new(Arc::new(
            CompiledWalk::compile("rollout_walk", &section).unwrap(),
        ));
        walk.seed(vec![IncomingInput {
            node: "gen".into(),
            name: "state".into(),
            tensors: vec![tref(1)],
        }])
        .unwrap();

        let mut last = CompletionResult::default();
        for i in 0..3u64 {
            walk.take_node_inputs("gen").unwrap();
            last = walk
                .complete_node(
                    "gen",
                    BTreeMap::from([("state".to_string(), vec![tref(100 + i)])]),
                )
                .unwrap();
        }
        assert!(last.walk_done);
        assert_eq!(
            last.events,
            vec![RouteEvent::Emission {
                name: "state".into(),
                modality: Some("video".into()),
                tensors: vec![tref(100), tref(101), tref(102)],
            }]
        );
    }

    #[test]
    fn parallel_branches_join() {
        // a -> c, b -> c (fan-in), c emits.
        let section = Section::Sequential {
            sections: vec![
                Section::Parallel {
                    sections: vec![
                        node("a", &["xa"], vec![edge("c", "ya")]),
                        node("b", &["xb"], vec![edge("c", "yb")]),
                    ],
                },
                node("c", &["ya", "yb"], vec![emit("z", "text", false)]),
            ],
        };
        let mut walk = WalkState::new(Arc::new(CompiledWalk::compile("par", &section).unwrap()));
        walk.seed(vec![
            IncomingInput {
                node: "a".into(),
                name: "xa".into(),
                tensors: vec![tref(1)],
            },
            IncomingInput {
                node: "b".into(),
                name: "xb".into(),
                tensors: vec![tref(2)],
            },
        ])
        .unwrap();

        let mut ready = walk.ready_nodes();
        ready.sort();
        assert_eq!(ready, vec!["a".to_string(), "b".to_string()]);

        walk.take_node_inputs("a").unwrap();
        walk.complete_node("a", BTreeMap::from([("ya".to_string(), vec![tref(3)])]))
            .unwrap();
        assert_eq!(
            walk.ready_nodes(),
            vec!["b".to_string()],
            "c waits for both branches; b is still ready"
        );

        walk.take_node_inputs("b").unwrap();
        walk.complete_node("b", BTreeMap::from([("yb".to_string(), vec![tref(4)])]))
            .unwrap();
        assert_eq!(walk.ready_nodes(), vec!["c".to_string()]);

        walk.take_node_inputs("c").unwrap();
        let res = walk
            .complete_node("c", BTreeMap::from([("z".to_string(), vec![tref(5)])]))
            .unwrap();
        assert!(res.walk_done);
    }

    #[test]
    fn walkset_parses_json() {
        let json = r#"{
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
        let set = WalkSet::from_json(json).unwrap();
        let walk = set.get("prefill_video").unwrap();
        assert_eq!(walk.nodes.len(), 2);
        assert!(set.get("nope").is_err());
    }

    #[test]
    fn compile_rejects_bad_specs() {
        // Unknown edge destination.
        let bad = node("a", &["x"], vec![edge("ghost", "y")]);
        assert!(CompiledWalk::compile("w", &bad).is_err());

        // Nested loops unsupported (documented v0 limitation).
        let inner = Section::Loop(LoopSpec {
            name: "inner".into(),
            body: Box::new(node("a", &["x"], vec![edge("a", "x")])),
            max_iters: 2,
            outputs: vec![],
            accumulated_outputs: vec![],
        });
        let outer = Section::Loop(LoopSpec {
            name: "outer".into(),
            body: Box::new(inner),
            max_iters: 2,
            outputs: vec![],
            accumulated_outputs: vec![],
        });
        assert!(matches!(
            CompiledWalk::compile("w", &outer),
            Err(CoreError::NestedLoop(_, _))
        ));
    }
}
