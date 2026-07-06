//! Streaming tier: chunk policies + stream buffers, ported from
//! `mstar/streaming/chunk_policy.py` and `stream_buffer.py`.
//!
//! A stream buffer sits on a connection between two partitions of one
//! request. The producer's streaming edges push tensors (descriptors) in
//! order; the policy decides when a window is ready; `pop_chunk` returns an
//! overlapping window and advances by the stride. The single ordering
//! invariant carried over from mstar: the final (flush) chunk sets
//! `is_final`, and partition-done must ride the walk that *consumes* that
//! chunk — never the ingest.
//!
//! In-process simplification vs mstar: tensors arrive in order (no RDMA
//! two-phase pre_read_register/put), and collation (stacking the window
//! into one tensor) is a data-plane op — the window is delivered as a list
//! of `TensorRef`s and the Python executor stacks.

use mstar_core::TensorRef;
use serde::{Deserialize, Serialize};

/// Chunking policy spec (what a model's connection declares) — serde-able.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChunkPolicySpec {
    /// Overlapping window advancing by `stride` (orpheus SNAC: 28/7).
    SlidingWindow { window: usize, stride: usize },
    /// Smaller first window/stride for low TTFB, then steady-state.
    RampSlidingWindow {
        first_window: usize,
        first_stride: usize,
        window: usize,
        stride: usize,
    },
    /// Causal-vocoder style: `chunk` new items with `left_context` overlap.
    LeftContext { chunk: usize, left_context: usize },
    /// Non-overlapping fixed chunks; optionally keep emitting empty chunks
    /// after the producer finishes (Thinker->Talker).
    Fixed {
        chunk_size: usize,
        #[serde(default)]
        continue_after_done: bool,
    },
}

/// Policy instance with per-request state (mstar's `ChunkPolicy` objects).
#[derive(Debug, Clone)]
pub struct ChunkPolicy {
    spec: ChunkPolicySpec,
    first_chunk_read: bool,
    pub items_consumed: usize,
    past_first: bool, // RampSlidingWindow only
}

impl ChunkPolicy {
    pub fn new(spec: ChunkPolicySpec) -> Self {
        Self {
            spec,
            first_chunk_read: false,
            items_consumed: 0,
            past_first: false,
        }
    }

    pub fn register_chunk(&mut self, chunk_size: usize) {
        self.first_chunk_read = true;
        self.items_consumed += chunk_size;
        self.past_first = true;
    }

    pub fn is_ready(&self, buffer_len: usize) -> bool {
        match &self.spec {
            ChunkPolicySpec::SlidingWindow { window, .. } => buffer_len >= *window,
            ChunkPolicySpec::RampSlidingWindow {
                first_window,
                window,
                ..
            } => buffer_len >= if self.past_first { *window } else { *first_window },
            ChunkPolicySpec::LeftContext {
                chunk,
                left_context,
            } => {
                if self.first_chunk_read {
                    buffer_len >= chunk + left_context
                } else {
                    buffer_len >= *chunk
                }
            }
            ChunkPolicySpec::Fixed { chunk_size, .. } => buffer_len >= *chunk_size,
        }
    }

    /// Items to ADVANCE by (stride). Only meaningful when `is_ready`.
    pub fn next_chunk_size(&self) -> usize {
        match &self.spec {
            ChunkPolicySpec::SlidingWindow { stride, .. } => *stride,
            ChunkPolicySpec::RampSlidingWindow {
                first_stride,
                stride,
                ..
            } => {
                if self.past_first {
                    *stride
                } else {
                    *first_stride
                }
            }
            ChunkPolicySpec::LeftContext {
                chunk,
                left_context,
            } => {
                if self.first_chunk_read {
                    *chunk
                } else {
                    chunk - left_context
                }
            }
            ChunkPolicySpec::Fixed { chunk_size, .. } => *chunk_size,
        }
    }

    /// Full window returned in a chunk (>= stride).
    pub fn window_size(&self) -> usize {
        match &self.spec {
            ChunkPolicySpec::SlidingWindow { window, .. } => *window,
            ChunkPolicySpec::RampSlidingWindow {
                first_window,
                window,
                ..
            } => {
                if self.past_first {
                    *window
                } else {
                    *first_window
                }
            }
            ChunkPolicySpec::LeftContext {
                chunk,
                left_context,
            } => {
                if self.first_chunk_read {
                    chunk + left_context
                } else {
                    *chunk
                }
            }
            ChunkPolicySpec::Fixed { chunk_size, .. } => *chunk_size,
        }
    }

    pub fn continue_after_producer_done(&self) -> bool {
        matches!(
            &self.spec,
            ChunkPolicySpec::Fixed {
                continue_after_done: true,
                ..
            }
        )
    }
}

/// One popped window.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamChunk {
    /// The window, in order (empty = empty chunk for continue-after-done).
    pub items: Vec<TensorRef>,
    pub chunk_index: u64,
    /// Global index of the window's first item.
    pub start_offset: usize,
    pub is_final: bool,
}

/// Per-(request, connection) buffer.
#[derive(Debug)]
pub struct StreamBuffer {
    policy: ChunkPolicy,
    buffer: std::collections::VecDeque<TensorRef>,
    consumed: usize,
    chunks_popped: u64,
    pub producer_done: bool,
    /// Whether the terminal (`is_final`) chunk has already been popped. Guards
    /// the empty-buffer-at-done path so exactly one final chunk is emitted.
    final_emitted: bool,
}

impl StreamBuffer {
    pub fn new(policy_spec: ChunkPolicySpec) -> Self {
        Self {
            policy: ChunkPolicy::new(policy_spec),
            buffer: std::collections::VecDeque::new(),
            consumed: 0,
            chunks_popped: 0,
            producer_done: false,
            final_emitted: false,
        }
    }

    /// Producer streamed tensors (in order; one call per streaming edge).
    pub fn push(&mut self, tensors: Vec<TensorRef>) {
        self.buffer.extend(tensors);
    }

    pub fn signal_done(&mut self) {
        self.producer_done = true;
    }

    /// Total items the consumer has advanced past (mstar's `_consumed`,
    /// reported as `stream_tokens_consumed`).
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    pub fn has_chunk_ready(&self) -> bool {
        let len = self.buffer.len();
        if self.producer_done {
            if len > 0 {
                return true; // final flush of the remainder
            }
            if self.policy.continue_after_producer_done() {
                return true; // keep emitting empty chunks
            }
            // Buffer already emptied (e.g. a non-overlapping policy drained it
            // on an ordinary pop before producer-done). We must still emit ONE
            // terminal chunk so its `is_final` sets `stream_done` — otherwise
            // the consumer partition, which finishes only on `stream_done`,
            // hangs forever. Guarded by `final_emitted` so it fires once.
            if !self.final_emitted {
                return true;
            }
        }
        self.policy.is_ready(len)
    }

    pub fn pop_chunk(&mut self) -> StreamChunk {
        let len = self.buffer.len();
        let start_offset = self.consumed;
        let (items, stride) = if self.producer_done && !self.policy.is_ready(len) {
            // Flush: whole remainder (possibly empty).
            let items: Vec<TensorRef> = self.buffer.drain(..).collect();
            let n = items.len();
            self.consumed += n;
            (items, n)
        } else {
            let window = self.policy.window_size().min(len);
            let stride = self.policy.next_chunk_size();
            let items: Vec<TensorRef> =
                self.buffer.iter().take(window).cloned().collect();
            self.buffer.drain(..stride.min(len));
            self.consumed += stride;
            (items, stride)
        };
        self.policy.register_chunk(stride);
        let is_final = self.producer_done
            && self.buffer.is_empty()
            && !self.policy.continue_after_producer_done();
        if is_final {
            self.final_emitted = true;
        }
        let chunk = StreamChunk {
            items,
            chunk_index: self.chunks_popped,
            start_offset,
            is_final,
        };
        self.chunks_popped += 1;
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(i: u64) -> TensorRef {
        TensorRef::new(i, vec![1], "i64")
    }

    fn ids(chunk: &StreamChunk) -> Vec<u64> {
        chunk.items.iter().map(|x| x.uuid).collect()
    }

    #[test]
    fn sliding_window_orpheus_shape() {
        // window=28, stride=7 — orpheus SNAC.
        let mut buf = StreamBuffer::new(ChunkPolicySpec::SlidingWindow {
            window: 28,
            stride: 7,
        });
        for i in 0..27 {
            buf.push(vec![t(i)]);
        }
        assert!(!buf.has_chunk_ready(), "27 < window");
        buf.push(vec![t(27)]);
        assert!(buf.has_chunk_ready());
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), (0..28).collect::<Vec<_>>());
        assert_eq!((c.chunk_index, c.start_offset, c.is_final), (0, 0, false));
        assert!(!buf.has_chunk_ready(), "21 left < window");

        // 7 more arrive -> next overlapping window [7..35).
        for i in 28..35 {
            buf.push(vec![t(i)]);
        }
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), (7..35).collect::<Vec<_>>());
        assert_eq!(c.start_offset, 7);

        // Producer done with 21 leftover -> final flush, short window.
        buf.signal_done();
        assert!(buf.has_chunk_ready());
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), (14..35).collect::<Vec<_>>());
        assert!(c.is_final);
        assert!(!buf.has_chunk_ready());
        assert_eq!(buf.consumed(), 35);
    }

    #[test]
    fn producer_done_exact_boundary_flush_is_full_window() {
        let mut buf = StreamBuffer::new(ChunkPolicySpec::SlidingWindow {
            window: 4,
            stride: 2,
        });
        for i in 0..4 {
            buf.push(vec![t(i)]);
        }
        buf.signal_done();
        // Still policy-ready: normal windowed pop, NOT flush; not final
        // (2 items remain).
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![0, 1, 2, 3]);
        assert!(!c.is_final);
        // Remainder 2 < window -> flush, final.
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![2, 3]);
        assert!(c.is_final);
    }

    #[test]
    fn fixed_policy_drained_before_done_still_emits_final_chunk() {
        // Regression: a non-overlapping policy can drain the buffer to empty on
        // an ordinary pop BEFORE producer-done. The terminal `is_final` chunk
        // must still fire on signal_done, or the consumer partition hangs.
        let mut buf = StreamBuffer::new(ChunkPolicySpec::Fixed {
            chunk_size: 4,
            continue_after_done: false,
        });
        for i in 0..8 {
            buf.push(vec![t(i)]);
        }
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![0, 1, 2, 3]);
        assert!(!c.is_final);
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![4, 5, 6, 7]);
        assert!(!c.is_final, "producer not done yet");
        assert!(!buf.has_chunk_ready(), "buffer empty, producer not done");

        // Producer signals done with an ALREADY-EMPTY buffer.
        buf.signal_done();
        assert!(
            buf.has_chunk_ready(),
            "must still deliver a terminal chunk so stream_done fires"
        );
        let c = buf.pop_chunk();
        assert!(c.items.is_empty());
        assert!(c.is_final, "terminal chunk carries is_final");
        assert!(!buf.has_chunk_ready(), "exactly one terminal chunk");
        assert_eq!(buf.consumed(), 8);
    }

    #[test]
    fn ramp_policy_first_chunk_smaller() {
        let mut buf = StreamBuffer::new(ChunkPolicySpec::RampSlidingWindow {
            first_window: 2,
            first_stride: 2,
            window: 6,
            stride: 3,
        });
        buf.push(vec![t(0), t(1)]);
        assert!(buf.has_chunk_ready(), "first window is 2");
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![0, 1]);
        for i in 2..8 {
            buf.push(vec![t(i)]);
        }
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), (2..8).collect::<Vec<_>>(), "steady window 6");
    }

    #[test]
    fn left_context_policy() {
        // chunk=4, left_context=2: first pop 4 (advance 2), then windows of
        // 6 advancing 4.
        let mut buf = StreamBuffer::new(ChunkPolicySpec::LeftContext {
            chunk: 4,
            left_context: 2,
        });
        for i in 0..4 {
            buf.push(vec![t(i)]);
        }
        assert!(buf.has_chunk_ready());
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![0, 1, 2, 3]);
        assert!(!buf.has_chunk_ready(), "2 left < window 6");
        for i in 4..8 {
            buf.push(vec![t(i)]);
        }
        let c = buf.pop_chunk();
        assert_eq!(ids(&c), vec![2, 3, 4, 5, 6, 7], "left context overlap");
    }

    #[test]
    fn fixed_continue_after_done_emits_empty_non_final() {
        let mut buf = StreamBuffer::new(ChunkPolicySpec::Fixed {
            chunk_size: 3,
            continue_after_done: true,
        });
        for i in 0..3 {
            buf.push(vec![t(i)]);
        }
        buf.pop_chunk();
        buf.signal_done();
        assert!(buf.has_chunk_ready(), "continue-after-done");
        let c = buf.pop_chunk();
        assert!(c.items.is_empty());
        assert!(!c.is_final, "consumer decides its own done");
    }
}
