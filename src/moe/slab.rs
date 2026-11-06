//! One VRAM allocation carved into ranges: the foundation of lending the expert cache's memory to the prefill
//! stream.
//!
//! The cache's banks and the stream's layer buffers are never both live (a chunk runs from its own buffers; the
//! cache is idle until the next decode), so the same bytes can serve both. Doing that needs storages that are a
//! *range* of one allocation - `candle_core::quantized::QCudaStorage::borrow_from`, added for this - because
//! neither crate can otherwise make a storage over someone else's bytes: cudarc's `CudaSlice` owns its pointer
//! and `Drop` frees it, and candle's `QCudaStorage` owns an owned `CudaSlice`.
//!
//! This module will hold the allocator (bump/free over the slab, one range per bank). For now it holds the
//! device test that pins down what the storage guarantees.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use candle_core::quantized::cuda::QCudaStorage;
use candle_core::quantized::GgmlDType;
use candle_core::Device;

/// Set while a chunk is computing inside the slab, i.e. while the cache's own bytes are the chunk's layer buffer.
///
/// Every writer into cache slots consults it. Without that gate the lookahead's fills (and the preload handoff)
/// keep DMA-ing into slots of the borrowed range *during* the chunk, which overwrote the layer buffer and made the
/// chunk's output NaN - measured as 62 non-finite chunk outputs per run at `--pcie-frac 0`, and zero with
/// `--no-lookahead`.
static BORROWED: AtomicBool = AtomicBool::new(false);

/// Serialises "may a fill write into cache memory right now" against "the borrow is taking the range".
///
/// The two sides are *asynchronous*: the lookahead is queued during a decode step and its worker can still be
/// running when the next request's chunk begins - the two phases are mutually exclusive in steady state, but not
/// across that boundary. Checking [`borrowed`] alone was a TOCTOU: a fill could pass the check and queue its DMA
/// after the borrow had taken the range (which is why `WRITE_DURING_BORROW` read 0 while the chunk still came out
/// NaN). A fill now holds this lock from the check until its copies are queued, and the borrow holds it from
/// setting the flag until its range is empty and every queued copy has landed.
static ACCESS: Mutex<()> = Mutex::new(());

/// Held by a fill for the moment it checks the flag and queues its copies.
pub(crate) fn lock_access() -> MutexGuard<'static, ()> {
    ACCESS.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`lock_access`] for a caller on the token thread that must **never** block on it.
///
/// The lookahead holds this lock across `upload_in_place`, which waits for the readers of the slot it is
/// overwriting (`copy_stream::wait_for_readers`, spinning on refcounts that only the token thread can drop). A
/// token-thread waiter would therefore deadlock against it - measured as "a request never finishes" with promotion
/// on. Nothing on the token path needs the lock: it needs to *know* whether a chunk owns the slab, and that is an
/// atomic read (`borrowed()`). So: try, and stand down if it is busy.
pub(crate) fn try_lock_access() -> Option<MutexGuard<'static, ()>> {
    ACCESS.try_lock().ok()
}

/// Mark the slab as borrowed (or not) for the duration of one layer's compute.
pub(crate) fn set_borrowed(on: bool) {
    BORROWED.store(on, Ordering::Release);
}

/// Whether a chunk owns the slab's low end right now: writers into cache slots must stand down.
pub(crate) fn borrowed() -> bool {
    BORROWED.load(Ordering::Acquire)
}

/// One VRAM allocation, handed out as byte ranges.
///
/// Every expert-cache bank is a range of this slab instead of its own `cudaMalloc`, which is what makes the memory
/// *lendable*: a run of the slab can later be re-viewed by the prefill stream, and the banks that overlap it are the
/// ones whose slots get emptied to make room.
///
/// The ranges are a bump allocation, never freed: the cache's shape does not change after load, so there is
/// nothing to reclaim, and a bump cursor keeps a layer's bank contiguous - which is the property the kernels and
/// the borrow both need.
///
/// The layout is recorded as each range is carved ([`Layout`]), because the loan addresses the slab by **byte
/// offset** and has to know which banks a range covers: strata's `slot_offsets`, which its cache keeps for exactly
/// this (`expert_cache.cpp:561-583`).
pub(crate) struct Slab {
    storage: QCudaStorage,
    cursor: Mutex<usize>,
    layout: Mutex<Layout>,
    bytes: usize,
}

/// One carved range: where it starts and how many bytes it took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub offset: usize,
    pub bytes: usize,
}

impl Range {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.bytes
    }
}

/// The slab's map from byte offsets to the ranges carved there.
///
/// Ranges are pushed in allocation order, which is also offset order (a bump allocator), so "the ranges a span
/// covers" is a contiguous slice found by binary search. That is the whole query the loan needs: which banks does
/// `[from, end)` displace, so their slots can be emptied before it and refilled after it.
#[derive(Default, Clone, Debug)]
pub(crate) struct Layout {
    ranges: Vec<Range>,
}

impl Layout {
    fn push(&mut self, offset: usize, bytes: usize) {
        self.ranges.push(Range { offset, bytes });
    }

    pub(crate) fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    /// Bytes carved so far: the end of the last range.
    pub(crate) fn end(&self) -> usize {
        self.ranges.last().map_or(0, |r| r.end())
    }

    pub(crate) fn count(&self) -> usize {
        self.ranges.len()
    }

    /// The ranges a span touches, in offset order.
    pub(crate) fn overlapping(&self, from: usize, to: usize) -> &[Range] {
        let first = self.ranges.partition_point(|r| r.end() <= from);
        let rest = &self.ranges[first..];
        let n = rest.partition_point(|r| r.offset < to);
        &rest[..n]
    }

    /// The offset where a span of `bytes` ending at the end of the carved region starts, or `None` when the slab
    /// has not carved that much yet.
    pub(crate) fn tail_start(&self, bytes: usize) -> Option<usize> {
        self.end().checked_sub(bytes)
    }
}

impl Slab {
    /// Whether the provider carves banks from a slab at all. **On by default**: the cache's bytes are otherwise a
    /// pile of separate `cudaMalloc`s and nothing can be lent from them. `NAIWA_CACHE_SLAB=0` turns it off (probe:
    /// the pre-M33 layout, one allocation per bank).
    pub(crate) fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            !matches!(
                std::env::var("NAIWA_CACHE_SLAB").as_deref(),
                Ok("0") | Ok("off") | Ok("false") | Ok("no")
            )
        })
    }

    /// Allocate `bytes` of VRAM. The slab's own dtype is irrelevant - `F32` has a block size of one, so the element
    /// count is a byte count and every range re-views the bytes with its own quantised type.
    pub(crate) fn new(device: &Device, bytes: usize) -> candle_core::Result<Self> {
        let dev = device
            .as_cuda_device()
            .map_err(|e| candle_core::Error::Msg(format!("slab needs a CUDA device: {e}")))?;
        let storage = QCudaStorage::zeros(dev, bytes / 4, GgmlDType::F32)?;
        let bytes = storage.storage_size_in_bytes();
        Ok(Self {
            storage,
            cursor: Mutex::new(0),
            layout: Mutex::new(Layout::default()),
            bytes,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn used(&self) -> usize {
        *self.cursor.lock().unwrap()
    }

    /// What has been carved so far: the query side of [`Self::alloc`], for the loan's address arithmetic.
    pub(crate) fn layout(&self) -> std::sync::MutexGuard<'_, Layout> {
        self.layout.lock().unwrap()
    }

    /// Whether the prompt path borrows its layer buffers from the pool instead of holding VRAM of its own.
    ///
    /// **Off by default again since M35 §14**: with the loan on, a prompt that fits in a *single* streamed chunk
    /// returns **no token at all** (a 300-token prompt: 3 011 ms and a real answer without the loan, zero SSE events
    /// with it; `NAIWA_PREFILL_BORROW=1` is the arm with it). Prompts spanning two or more chunks are unaffected, which
    /// is why §20.13's A/B - all of it on 2 896- and 3 433-token prompts - saw only the win. The 2x on long prompts is
    /// real and is why the knob stays; the silent empty answer on the common case is why it is no longer the default.
    ///
    /// The loan is what decides the prompt step, and the prompt step is what decides the prompt's wall clock: the
    /// chunk is planned once at warmup from the VRAM that is free, and every chunk streams the whole routed expert set
    /// (≈23 GiB) over PCIe again, so a prompt that needs four chunks pays that four times. With the shipped 2.5 GB
    /// cache the plan lands on 768 tokens and one layer buffer; with the loan it lands on 2 304-2 560 tokens and two,
    /// and a 3 433-token prompt's first token comes at **4 791 ms instead of 10 475**, a 2 896-token one at **3 730
    /// instead of 8 267** (same session, same cache, §20.13).
    ///
    /// It was off because the first measurement said it cost the cache: one loan wipes 719 of 1 474 slots at the
    /// turn-1 chunk and the conversation then ended 39.2 s against 30.2 s (`short`, 3 rounds). That result did not
    /// survive the NaN fix and the once-per-chunk restructure (§20.4): the later A/B on the same corpus is neutral -
    /// 33.5 s against 36.2 s end to end, TPOT 44 ms in both, a tail TTFT of 2 280 against 2 326 - so what is left is a
    /// decision about which case to favour, and the long prompt wins by 2x while the short one does not move.
    ///
    /// Read as a **value**, not as presence: `=0` is the arm without the loan (a presence test made `=0` mean on,
    /// and the first A/B of M33 ran the loan twice).
    pub(crate) fn borrowing() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| match std::env::var("NAIWA_PREFILL_BORROW").as_deref() {
            Ok("1") | Ok("on") | Ok("true") | Ok("yes") => true,
            _ => false,
        })
    }

    /// The slab's device pointer: the borrow ranges are `[base, base + end)`.
    pub(crate) fn base(&self) -> candle_core::Result<u64> {
        Ok(self.storage.device_ptr()? as u64)
    }

    /// A range at `offset` of `el_count` elements at `dtype`, for a buffer that reuses the low end of the slab
    /// instead of taking VRAM of its own. Unlike [`Self::alloc`] this does not advance the cursor: the range is
    /// already owned by whichever cache banks were carved first, and the caller empties those slots.
    pub(crate) fn borrow_at(
        &self,
        offset: usize,
        el_count: usize,
        dtype: GgmlDType,
        device: &Device,
    ) -> candle_core::Result<QCudaStorage> {
        let dev = device
            .as_cuda_device()
            .map_err(|e| candle_core::Error::Msg(format!("slab needs a CUDA device: {e}")))?;
        QCudaStorage::borrow_from(&self.storage, offset, el_count, dtype, dev)
    }

    /// A range of `el_count` elements at `dtype`, or `None` when the slab is full.
    pub(crate) fn alloc(
        &self,
        el_count: usize,
        dtype: GgmlDType,
        device: &Device,
    ) -> candle_core::Result<Option<QCudaStorage>> {
        let dev = device
            .as_cuda_device()
            .map_err(|e| candle_core::Error::Msg(format!("slab needs a CUDA device: {e}")))?;
        let mut cursor = self.cursor.lock().unwrap();
        // `borrow_from` pads the range the same way `zeros` does, so ask it for the range and roll back on failure.
        match QCudaStorage::borrow_from(&self.storage, *cursor, el_count, dtype, dev) {
            Ok(storage) => {
                let offset = *cursor;
                *cursor += storage.allocated_bytes();
                drop(cursor);
                self.layout.lock().unwrap().push(offset, storage.allocated_bytes());
                Ok(Some(storage))
            }
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::cuda::QCudaStorage;
    use candle_core::quantized::{GgmlDType, QStorage};
    use candle_core::Device;

    /// `NAIWA_SLAB_DBG=1 cargo test --release --features cuda -- --ignored --test-threads=1 slab::`
    ///
    /// Checks the three things the borrow relies on: the range starts at its offset, a borrowed range is a *view*
    /// (dropping it does not free the slab), and two ranges can be read as different quantised types.
    #[test]
    #[ignore = "needs a CUDA device; run with --ignored --test-threads=1"]
    fn a_borrowed_range_is_a_view_of_the_slab() {
        let device = Device::new_cuda(0).expect("cuda");
        let dev = device.as_cuda_device().expect("cuda device");
        // 16 MiB of bytes, allocated as F32 elements (block size 1, so el_count == bytes / 4).
        let bytes = 16 << 20;
        let slab = QCudaStorage::zeros(dev, bytes / 4, GgmlDType::F32).expect("slab");
        let base = slab.device_ptr().expect("slab ptr");

        // `el_count` is what fits in the range at that dtype, so 1 MiB of IQ3_S weights is fewer elements than
        // 1 MiB of IQ4_XS. Ask for the range and let the borrow report what does not fit.
        let a = QCudaStorage::borrow_from(&slab, 0, 1 << 20, GgmlDType::IQ3_S, dev).expect("a");
        let b = QCudaStorage::borrow_from(&slab, 1 << 20, 2 << 20, GgmlDType::IQ4XS, dev).expect("b");
        assert_eq!(a.device_ptr().unwrap() as usize, base as usize, "the first range starts at the slab");
        assert_eq!(b.device_ptr().unwrap() as usize, base as usize + (1 << 20), "the second starts at its offset");
        assert!(a.storage_size_in_bytes() <= 1 << 20, "a's bytes stay inside its range");
        assert!(b.storage_size_in_bytes() <= 2 << 20);

        // Dropping a borrowed range must leave the slab intact: its pointer stays valid and a later borrow of the
        // same bytes still lands at the same address.
        drop(a);
        drop(b);
        let c = QCudaStorage::borrow_from(&slab, 1 << 20, 2 << 20, GgmlDType::IQ3_XXS, dev).expect("c");
        assert_eq!(c.device_ptr().unwrap() as usize, base as usize + (1 << 20));
        assert_eq!(slab.storage_size_in_bytes(), bytes);

        // And the slab is still the slab.
        match QStorage::Cuda(slab) {
            QStorage::Cuda(s) => assert_eq!(s.storage_size_in_bytes(), bytes),
            _ => unreachable!(),
        }
    }

    /// The loan addresses the slab by byte offset, so the map from offsets to carved ranges is what says which
    /// banks a span displaces. Pure arithmetic - no device needed.
    #[test]
    fn the_layout_says_which_ranges_a_span_covers() {
        // Three layers' banks, the middle one bigger, exactly as the mixed-quant packs come.
        let mut layout = super::Layout::default();
        for (offset, bytes) in [(0usize, 48usize), (48, 52), (100, 40)] {
            layout.push(offset, bytes);
        }
        assert_eq!(layout.end(), 140);
        assert_eq!(layout.count(), 3);

        // A span inside one range touches only that range.
        let one = layout.overlapping(60, 90);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].offset, 48);
        // A span across a boundary touches both sides, in offset order.
        let two = layout.overlapping(40, 60);
        assert_eq!(two.iter().map(|r| r.offset).collect::<Vec<_>>(), vec![0, 48]);
        // The whole region is all of them.
        assert_eq!(layout.overlapping(0, 140).len(), 3);
        // A span that starts in a gap or past the end touches nothing.
        assert!(layout.overlapping(140, 200).is_empty());

        // The tail: `tail_start` is the offset a `bytes`-long loan ending at the carved end begins at.
        assert_eq!(layout.tail_start(40), Some(100));
        assert_eq!(layout.tail_start(140), Some(0));
        assert_eq!(layout.tail_start(141), None, "a loan bigger than the cache has no start");
    }
}
