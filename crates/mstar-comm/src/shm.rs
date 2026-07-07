//! Shared-memory tensor arena — the data plane for cross-process transport.
//!
//! Ported from `mstar/rust/mstar_shmring` into mstar-rs (self-contained, no
//! PyO3 here — the Python buffer-protocol view lives in `mstar-py`). One
//! persistent `/dev/shm` mmap per producer entity plus an in-process
//! first-fit free-list allocator, replacing the per-tensor file open/write/
//! read/unlink dance in mstar's `SharedMemoryCommunicationManager`.
//!
//! - Producer: [`ShmArena::create`], [`ShmArena::reserve`] -> offset, copy
//!   D2H bytes into `arena[off..off+n]`, send the offset as a descriptor,
//!   [`ShmArena::free`] the offset once the tensor is reclaimed. The conductor
//!   drives reclaim per-tensor: the runtime reports a tensor unreachable
//!   (`Event::Free`) and the conductor frees the owning arena's offset (its
//!   own directly, a worker's via a `free` message), with a per-request sweep
//!   as the backstop.
//! - Consumer: [`ShmArena::open`] the same name, read `arena[off..off+n]`
//!   (H2D). Zero syscalls per tensor, one memcpy each way.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Mutex;

use memmap2::{MmapMut, MmapOptions};
use thiserror::Error;

/// Cache-line-friendly alignment; keeps neighbouring tensors off shared lines.
pub const ALIGN: usize = 256;

#[inline]
fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

#[derive(Debug, Error)]
pub enum ShmError {
    #[error("size must be > 0")]
    ZeroSize,
    #[error("arena full: need {need} B, {free} B free of {total} B")]
    Full {
        need: usize,
        free: usize,
        total: usize,
    },
    #[error("io on {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

/// First-fit free-list allocator over the arena. Coalesces on free.
struct Allocator {
    free: Vec<(usize, usize)>,    // (offset, len), sorted by offset, disjoint
    live: HashMap<usize, usize>,  // offset -> aligned len (free() needs only offset)
}

impl Allocator {
    fn new(size: usize) -> Self {
        Self {
            free: vec![(0, size)],
            live: HashMap::new(),
        }
    }

    fn alloc(&mut self, n: usize) -> Option<usize> {
        let n = align_up(n.max(1));
        for i in 0..self.free.len() {
            let (off, len) = self.free[i];
            if len >= n {
                if len == n {
                    self.free.remove(i);
                } else {
                    self.free[i] = (off + n, len - n);
                }
                self.live.insert(off, n);
                return Some(off);
            }
        }
        None
    }

    fn dealloc(&mut self, off: usize) -> bool {
        let Some(n) = self.live.remove(&off) else {
            return false; // double-free / unknown -> no-op
        };
        self.free.push((off, n));
        self.free.sort_by_key(|b| b.0);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free.len());
        for &(o, l) in &self.free {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == o {
                    last.1 += l;
                    continue;
                }
            }
            merged.push((o, l));
        }
        self.free = merged;
        true
    }

    fn bytes_free(&self) -> usize {
        self.free.iter().map(|b| b.1).sum()
    }
}

/// A named shared-memory arena. The producer owns it (unlinks on drop); a
/// consumer opens the same name read/write without ownership.
pub struct ShmArena {
    mmap: MmapMut,
    len: usize,
    path: String,
    owner: bool,
    alloc: Mutex<Allocator>,
}

// The mmap intentionally aliases across processes; in-process access is
// serialised by the allocator Mutex (and, from Python, the GIL).
unsafe impl Send for ShmArena {}
unsafe impl Sync for ShmArena {}

fn shm_path(name: &str) -> String {
    if Path::new("/dev/shm").is_dir() {
        format!("/dev/shm/{name}")
    } else {
        format!("/tmp/{name}")
    }
}

impl ShmArena {
    /// Producer: create (or replace) the arena and own it (unlink on drop).
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::ZeroSize);
        }
        let path = shm_path(name);
        let io = |source| ShmError::Io {
            path: path.clone(),
            source,
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(io)?;
        file.set_len(size as u64).map_err(io)?;
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file) }.map_err(io)?;
        Ok(Self {
            mmap,
            len: size,
            path,
            owner: true,
            alloc: Mutex::new(Allocator::new(size)),
        })
    }

    /// Consumer: open an existing arena read/write; never unlinks.
    pub fn open(name: &str) -> Result<Self, ShmError> {
        let path = shm_path(name);
        let io = |source| ShmError::Io {
            path: path.clone(),
            source,
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(io)?;
        let size = file.metadata().map_err(io)?.len() as usize;
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file) }.map_err(io)?;
        Ok(Self {
            mmap,
            len: size,
            path,
            owner: false,
            alloc: Mutex::new(Allocator::new(size)), // unused on a consumer
        })
    }

    /// Reserve `nbytes`; returns the byte offset into the arena. Producer only.
    pub fn reserve(&self, nbytes: usize) -> Result<usize, ShmError> {
        let mut a = self.alloc.lock().expect("alloc lock");
        a.alloc(nbytes).ok_or_else(|| ShmError::Full {
            need: nbytes,
            free: a.bytes_free(),
            total: self.len,
        })
    }

    /// Release a reserved offset. Producer only. Idempotent (false = unknown).
    pub fn free(&self, offset: usize) -> bool {
        self.alloc.lock().expect("alloc lock").dealloc(offset)
    }

    pub fn size(&self) -> usize {
        self.len
    }

    pub fn bytes_free(&self) -> usize {
        self.alloc.lock().expect("alloc lock").bytes_free()
    }

    /// Base pointer of the arena — used by the `mstar-py` buffer-protocol view
    /// so torch can `frombuffer(arena[off:off+n])` with no intermediate copy.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mmap.as_ptr() as *mut u8
    }

    /// Byte slice into the arena (bounds-checked; for tests / Rust-side copies).
    pub fn bytes(&self, offset: usize, len: usize) -> &[u8] {
        &self.mmap[offset..offset + len]
    }

    /// Copy `src` into the arena at `offset` (producer D2H staging).
    pub fn write_at(&mut self, offset: usize, src: &[u8]) {
        self.mmap[offset..offset + src.len()].copy_from_slice(src);
    }

    pub fn close(&mut self) {
        if self.owner && !self.path.is_empty() {
            let _ = std::fs::remove_file(&self.path);
            self.path.clear();
        }
    }
}

impl Drop for ShmArena {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(tag: &str) -> String {
        format!("mstar_shm_test_{tag}_{:?}", std::thread::current().id())
    }

    #[test]
    fn alloc_free_coalesce() {
        let mut a = Allocator::new(1024);
        let x = a.alloc(200).unwrap(); // -> 256 aligned
        let y = a.alloc(200).unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 256);
        assert_eq!(a.bytes_free(), 1024 - 512);
        a.dealloc(x);
        a.dealloc(y);
        // Fully coalesced back to one free block.
        assert_eq!(a.free, vec![(0, 1024)]);
        assert!(!a.dealloc(9999)); // unknown offset
    }

    #[test]
    fn producer_consumer_roundtrip_same_process() {
        let n = name("rt");
        let mut prod = ShmArena::create(&n, 4096).unwrap();
        let off = prod.reserve(1500).unwrap();
        let payload: Vec<u8> = (0..1500).map(|i| (i % 251) as u8).collect();
        prod.write_at(off, &payload);

        // Consumer opens the same name and reads at the descriptor offset.
        let cons = ShmArena::open(&n).unwrap();
        assert_eq!(cons.bytes(off, 1500), &payload[..]);

        prod.free(off);
        assert_eq!(prod.bytes_free(), 4096);
    }

    #[test]
    fn reserve_reports_full() {
        let n = name("full");
        let arena = ShmArena::create(&n, 512).unwrap();
        arena.reserve(300).unwrap(); // -> 512 aligned, arena full
        assert!(matches!(arena.reserve(1), Err(ShmError::Full { .. })));
    }

    #[test]
    fn zero_size_rejected() {
        assert!(matches!(
            ShmArena::create(&name("z"), 0),
            Err(ShmError::ZeroSize)
        ));
    }
}
