//! mstar-sched: the continuous-batching micro-scheduler.
//!
//! Ports the scheduling core of `mstar/worker/micro_scheduler.py`: every
//! step, scan the ready nodes across all requests, group them by
//! (node, walk) — a batch runs ONE node name under ONE graph walk — pick a
//! group fairly (least-recently scheduled), and clamp to the max batch size.
//!
//! Engine-readiness gating (`engine.check_ready`, e.g. KV-page availability)
//! is a data-plane concern; when it lands (T4), the runtime will filter the
//! ready set before handing it to the scheduler.

use std::collections::BTreeMap;

/// One request's ready node, as reported by its walk state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyEntry {
    pub request_id: u64,
    pub node: String,
    pub walk: String,
}

/// A batch to execute: one node name, one walk, N requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    pub node: String,
    pub walk: String,
    pub request_ids: Vec<u64>,
}

/// Round-robin continuous-batching scheduler. Stateful only for fairness
/// bookkeeping (a monotonic batch counter per (node, walk) group).
#[derive(Debug, Default)]
pub struct MicroScheduler {
    batch_counter: u64,
    last_scheduled: BTreeMap<(String, String), u64>,
}

impl MicroScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pick the next batch from the ready set, or None if nothing is ready.
    /// `max_batch_size = 0` means unbounded.
    pub fn next_batch(&mut self, ready: &[ReadyEntry], max_batch_size: usize) -> Option<BatchPlan> {
        if ready.is_empty() {
            return None;
        }
        // Group by (node, walk); BTreeMap gives deterministic iteration.
        let mut groups: BTreeMap<(String, String), Vec<u64>> = BTreeMap::new();
        for entry in ready {
            groups
                .entry((entry.node.clone(), entry.walk.clone()))
                .or_default()
                .push(entry.request_id);
        }
        // Least-recently-scheduled group first (never-scheduled counts as 0).
        let key = groups
            .keys()
            .min_by_key(|k| self.last_scheduled.get(*k).copied().unwrap_or(0))?
            .clone();
        let mut request_ids = groups.remove(&key).expect("key from groups");
        request_ids.sort_unstable();
        request_ids.dedup();
        if max_batch_size > 0 {
            request_ids.truncate(max_batch_size);
        }

        self.batch_counter += 1;
        self.last_scheduled.insert(key.clone(), self.batch_counter);
        Some(BatchPlan {
            node: key.0,
            walk: key.1,
            request_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(rid: u64, node: &str, walk: &str) -> ReadyEntry {
        ReadyEntry {
            request_id: rid,
            node: node.into(),
            walk: walk.into(),
        }
    }

    #[test]
    fn batches_group_by_node_and_walk() {
        let mut sched = MicroScheduler::new();
        let ready = vec![
            entry(1, "LLM", "decode"),
            entry(2, "LLM", "decode"),
            entry(3, "LLM", "prefill"),
        ];
        let batch = sched.next_batch(&ready, 0).unwrap();
        // One walk per batch: the two decode requests batch together.
        assert_eq!(batch.request_ids.len(), 2);
        assert_eq!(batch.walk, "decode");
    }

    #[test]
    fn round_robin_across_groups() {
        let mut sched = MicroScheduler::new();
        let ready = vec![
            entry(1, "LLM", "decode"),
            entry(2, "snac_decoder", "snac_chunk"),
        ];
        let first = sched.next_batch(&ready, 0).unwrap();
        let second = sched.next_batch(&ready, 0).unwrap();
        assert_ne!(first.node, second.node, "fairness: alternate groups");
        // Third pick returns to the first group.
        let third = sched.next_batch(&ready, 0).unwrap();
        assert_eq!(third.node, first.node);
    }

    #[test]
    fn respects_max_batch_size() {
        let mut sched = MicroScheduler::new();
        let ready: Vec<ReadyEntry> = (0..10).map(|i| entry(i, "LLM", "decode")).collect();
        let batch = sched.next_batch(&ready, 4).unwrap();
        assert_eq!(batch.request_ids.len(), 4);
    }

    #[test]
    fn empty_ready_set() {
        let mut sched = MicroScheduler::new();
        assert!(sched.next_batch(&[], 8).is_none());
    }
}
