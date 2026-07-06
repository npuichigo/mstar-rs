//! Paged KV-cache descriptor management — the control-plane half of mstar's
//! `engine/kv_store.py` (`PageAllocator` + `PagedAllocationManager`).
//!
//! Rust owns pages as *descriptors*: a free-list allocator per cache label
//! and per-(request, label) page tables + sequence positions. The actual KV
//! tensors live in Python (`mstar_rs.kv.PagedKVCache`), indexed by the page
//! tables this module hands out.
//!
//! Contract with the runtime:
//! - The policy declares, per walk, how many tokens each execution of a KV
//!   node appends (`kv_appends` at `start_walk`; mstar ships the same info
//!   as `per_label_seq_info` in its forward-pass args).
//! - Reservation happens at schedule time: a request whose append doesn't
//!   fit is left out of the batch (mstar's `AllocationFailedError` → hold),
//!   and becomes schedulable again when pages free up.
//! - Positions advance when the batch completes; pages free when the
//!   request finishes.

use std::collections::BTreeMap;

/// Static configuration for one KV cache label (usually one per KV node,
/// e.g. "LLM"). `page_size` is in tokens.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub label: String,
    pub num_pages: u32,
    pub page_size: u32,
}

/// Free-list page allocator (mstar `kv_store.py:PageAllocator`).
#[derive(Debug)]
struct PageAllocator {
    free: Vec<u32>,
}

impl PageAllocator {
    fn new(num_pages: u32) -> Self {
        // Reverse so pages are handed out in ascending order (cosmetic).
        Self {
            free: (0..num_pages).rev().collect(),
        }
    }

    fn available(&self) -> u32 {
        self.free.len() as u32
    }

    fn try_alloc(&mut self, n: u32) -> Option<Vec<u32>> {
        if (self.free.len() as u32) < n {
            return None;
        }
        let at = self.free.len() - n as usize;
        Some(self.free.split_off(at))
    }

    fn free_pages(&mut self, pages: Vec<u32>) {
        self.free.extend(pages);
    }
}

/// Per-(request, label) cache state (mstar `kv_store.py:KVRequestState`).
#[derive(Debug, Default, Clone)]
pub struct KvRequestState {
    pub pages: Vec<u32>,
    pub seq_pos: u64,
}

/// The KV view handed to the data plane for one request in a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvView {
    pub label: String,
    pub pages: Vec<u32>,
    /// Sequence position before this execution's append.
    pub seq_pos: u64,
    /// Tokens this execution appends (0 = read-only, e.g. pi05 action_gen).
    pub append_len: u64,
    /// Transient tokens past `seq_pos + append_len` the pages also cover —
    /// scratch K/V (e.g. pi05's 50-token suffix) written each execution but
    /// never committed. Mirrors mstar planning attention for `seq_len +
    /// suffix` while `advance_seq_lens` is never called in the flow loop.
    pub scratch_len: u64,
}

#[derive(Debug)]
pub struct KvManager {
    configs: BTreeMap<String, KvCacheConfig>,
    allocators: BTreeMap<String, PageAllocator>,
    /// KV node name -> cache label.
    node_labels: BTreeMap<String, String>,
    states: BTreeMap<(u64, String), KvRequestState>,
}

impl KvManager {
    pub fn new(configs: Vec<KvCacheConfig>, node_labels: BTreeMap<String, String>) -> Self {
        let allocators = configs
            .iter()
            .map(|c| (c.label.clone(), PageAllocator::new(c.num_pages)))
            .collect();
        Self {
            configs: configs.into_iter().map(|c| (c.label.clone(), c)).collect(),
            allocators,
            node_labels,
            states: BTreeMap::new(),
        }
    }

    pub fn label_for_node(&self, node: &str) -> Option<&str> {
        self.node_labels.get(node).map(String::as_str)
    }

    pub fn state(&self, request_id: u64, label: &str) -> KvRequestState {
        self.states
            .get(&(request_id, label.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn pages_needed(&self, label: &str, state: &KvRequestState, tokens: u64) -> u32 {
        let page_size = self.configs[label].page_size as u64;
        let needed = tokens.div_ceil(page_size) as u32;
        needed.saturating_sub(state.pages.len() as u32)
    }

    /// Would `append + scratch` tokens fit right now (without allocating)?
    pub fn can_reserve(&self, request_id: u64, label: &str, append: u64, scratch: u64) -> bool {
        let state = self.state(request_id, label);
        let needed = self.pages_needed(label, &state, state.seq_pos + append + scratch);
        needed <= self.allocators[label].available()
    }

    /// Allocate pages so the request can hold `seq_pos + append + scratch`
    /// tokens (scratch = transient, never committed). Returns the KV view
    /// for the data plane, or None if pages ran out (callers should have
    /// checked `can_reserve`; None = hold the request).
    pub fn reserve(
        &mut self,
        request_id: u64,
        label: &str,
        append: u64,
        scratch: u64,
    ) -> Option<KvView> {
        let key = (request_id, label.to_string());
        let state = self.states.entry(key).or_default();
        let page_size = self.configs[label].page_size as u64;
        let needed = {
            let total = state.seq_pos + append + scratch;
            (total.div_ceil(page_size) as u32).saturating_sub(state.pages.len() as u32)
        };
        if needed > 0 {
            let new_pages = self.allocators.get_mut(label)?.try_alloc(needed)?;
            state.pages.extend(new_pages);
        }
        Some(KvView {
            label: label.to_string(),
            pages: state.pages.clone(),
            seq_pos: state.seq_pos,
            append_len: append,
            scratch_len: scratch,
        })
    }

    /// The batch completed: the append is now materialized in the cache.
    pub fn advance(&mut self, request_id: u64, label: &str, append: u64) {
        if let Some(state) = self.states.get_mut(&(request_id, label.to_string())) {
            state.seq_pos += append;
        }
    }

    /// Free every label's pages for a finished request.
    pub fn free_request(&mut self, request_id: u64) {
        let keys: Vec<_> = self
            .states
            .range((request_id, String::new())..(request_id + 1, String::new()))
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys {
            if let Some(state) = self.states.remove(&key) {
                if let Some(alloc) = self.allocators.get_mut(&key.1) {
                    alloc.free_pages(state.pages);
                }
            }
        }
    }

    pub fn available_pages(&self, label: &str) -> u32 {
        self.allocators
            .get(label)
            .map(PageAllocator::available)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(num_pages: u32, page_size: u32) -> KvManager {
        KvManager::new(
            vec![KvCacheConfig {
                label: "LLM".into(),
                num_pages,
                page_size,
            }],
            BTreeMap::from([("LLM".to_string(), "LLM".to_string())]),
        )
    }

    #[test]
    fn reserve_advance_and_grow() {
        let mut kv = manager(8, 16);
        // Prefill: 40 tokens -> 3 pages.
        let view = kv.reserve(1, "LLM", 40, 0).unwrap();
        assert_eq!(view.pages.len(), 3);
        assert_eq!(view.seq_pos, 0);
        kv.advance(1, "LLM", 40);
        assert_eq!(kv.state(1, "LLM").seq_pos, 40);
        assert_eq!(kv.available_pages("LLM"), 5);

        // Decode appends within the last page: no new allocation.
        let view = kv.reserve(1, "LLM", 1, 0).unwrap();
        assert_eq!(view.pages.len(), 3);
        assert_eq!(view.seq_pos, 40);
        kv.advance(1, "LLM", 1);

        // Crossing the page boundary (48 tokens) grows by one page.
        for _ in 0..7 {
            kv.reserve(1, "LLM", 1, 0).unwrap();
            kv.advance(1, "LLM", 1);
        }
        assert_eq!(kv.state(1, "LLM").seq_pos, 48);
        let view = kv.reserve(1, "LLM", 1, 0).unwrap();
        assert_eq!(view.pages.len(), 4);
    }

    #[test]
    fn read_only_reserve_allocates_nothing() {
        let mut kv = manager(4, 16);
        kv.reserve(1, "LLM", 20, 0).unwrap();
        kv.advance(1, "LLM", 20);
        let before = kv.available_pages("LLM");
        // pi05 action_gen: reads the prefix cache, appends nothing.
        let view = kv.reserve(1, "LLM", 0, 0).unwrap();
        assert_eq!(view.append_len, 0);
        assert_eq!(view.seq_pos, 20);
        assert_eq!(kv.available_pages("LLM"), before);
    }

    #[test]
    fn scratch_reserves_pages_but_never_commits() {
        let mut kv = manager(8, 16);
        // Prefill 20 tokens -> 2 pages (12 free slots in page 2).
        kv.reserve(1, "LLM", 20, 0).unwrap();
        kv.advance(1, "LLM", 20);
        assert_eq!(kv.available_pages("LLM"), 6);

        // Suffix scratch of 30 tokens: 20+30=50 -> 4 pages total (2 new).
        let view = kv.reserve(1, "LLM", 0, 30).unwrap();
        assert_eq!((view.seq_pos, view.append_len, view.scratch_len), (20, 0, 30));
        assert_eq!(view.pages.len(), 4);
        kv.advance(1, "LLM", 0);
        assert_eq!(kv.state(1, "LLM").seq_pos, 20, "scratch never commits");

        // Next step re-reserves: pages already cover 50 tokens, no growth.
        let before = kv.available_pages("LLM");
        let view = kv.reserve(1, "LLM", 0, 30).unwrap();
        assert_eq!(view.pages.len(), 4);
        assert_eq!(kv.available_pages("LLM"), before);
    }

    #[test]
    fn oom_and_recovery() {
        let mut kv = manager(4, 16);
        kv.reserve(1, "LLM", 64, 0).unwrap(); // all 4 pages
        assert!(!kv.can_reserve(2, "LLM", 1, 0));
        assert!(kv.reserve(2, "LLM", 1, 0).is_none());
        kv.free_request(1);
        assert_eq!(kv.available_pages("LLM"), 4);
        assert!(kv.can_reserve(2, "LLM", 1, 0));
        kv.reserve(2, "LLM", 1, 0).unwrap();
    }
}
