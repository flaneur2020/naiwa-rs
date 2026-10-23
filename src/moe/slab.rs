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
/// *lendable*: the low end of the slab can later be re-viewed by the prefill stream, and the banks that overlap it
/// are the ones whose slots get emptied to make room.
///
/// The ranges are a bump allocation, never freed: the cache's shape does not change after load, so there is
/// nothing to reclaim, and a bump cursor keeps a layer's bank contiguous - which is the property the kernels and
/// the eventual borrow both need.
pub(crate) struct Slab {
    storage: QCudaStorage,
    cursor: Mutex<usize>,
    bytes: usize,
}

impl Slab {
    /// `NAIWA_CACHE_SLAB=1` decides whether the provider carves banks from a slab at all.
    pub(crate) fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("NAIWA_CACHE_SLAB").is_some())
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
            bytes,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn used(&self) -> usize {
        *self.cursor.lock().unwrap()
    }

    /// Whether the prefill stream borrows the slab's low end (`NAIWA_PREFILL_BORROW=1`).
    pub(crate) fn borrowing() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("NAIWA_PREFILL_BORROW").is_some())
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
                *cursor += storage.allocated_bytes();
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
}

#[cfg(test)]
mod vmm_tests {
    use candle_core::cuda::cudarc::driver::{result, sys};

    fn gran(dev: i32) -> usize {
        let mut g = 0usize;
        // SAFETY: the option and the out pointer are valid.
        let rc = unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut g,
                &prop(dev),
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "granularity");
        g
    }

    fn prop(dev: i32) -> sys::CUmemAllocationProp {
        let mut p: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        p.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: dev,
        };
        p
    }

    fn free_bytes() -> (usize, usize) {
        let mut f = 0usize;
        let mut t = 0usize;
        // SAFETY: out pointers are valid.
        let rc = unsafe { sys::cuMemGetInfo_v2(&mut f, &mut t) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "mem info");
        (f, t)
    }

    /// `NAIWA_VMM_DBG=1 cargo test --release --features cuda -- --ignored --test-threads=1 vmm`
    ///
    /// The premise of "elastic banks": a reserved *virtual* range can have its physical memory handed back to the
    /// driver and taken again, at the same address, while the driver's stream-ordered pool - which is what every
    /// activation tensor comes from - picks the freed memory up in between. If this holds, the expert cache can be
    /// sized as `free_at_load + scratch` and give the scratch back only while a chunk runs.
    #[test]
    #[ignore = "needs a CUDA device; run with --ignored --test-threads=1"]
    fn a_reserved_range_can_lend_its_physical_memory_to_the_pool() {
        let dev = 0i32;
        // A candle device first: it initialises the context and binds it to this thread, which the raw driver calls
        // below need (cuMemGetInfo without a current context returns CUDA_ERROR_INVALID_CONTEXT).
        let device = candle_core::Device::new_cuda(0).expect("cuda");
        let g = gran(dev);
        let size: usize = 256 << 20; // 256 MiB, a stand-in for the scratch (~866 MiB in the real run)
        let size = size.div_ceil(g) * g;

        // Reserve the virtual range once: this is the address the cache's tensors will live at, forever.
        let mut va: sys::CUdeviceptr = 0;
        // SAFETY: out pointer is valid; no address hint, no flags.
        let rc = unsafe { sys::cuMemAddressReserve(&mut va, size, 0, 0, 0) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "address reserve");

        let map_all = |va: sys::CUdeviceptr| -> sys::CUmemGenericAllocationHandle {
            let mut handle: sys::CUmemGenericAllocationHandle = 0;
            // SAFETY: size is granular; the property is a device-pinned allocation.
            let rc = unsafe { sys::cuMemCreate(&mut handle, size, &prop(dev), 0) };
            assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemCreate");
            // SAFETY: the handle is live and the range is reserved.
            let rc = unsafe { sys::cuMemMap(va, size, 0, handle, 0) };
            assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemMap");
            let mut access: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
            access.location = prop(dev).location;
            access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            // SAFETY: one descriptor, valid for the whole mapped range.
            let rc = unsafe { sys::cuMemSetAccess(va, size, &access, 1) };
            assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemSetAccess");
            handle
        };

        let handle = map_all(va);
        let (free_mapped, _) = free_bytes();

        // Give the physical memory back: unmap the range and release the handle. The *virtual* address stays
        // reserved, so nothing that points into it has to move.
        // SAFETY: the range is mapped and the handle is live.
        assert_eq!(unsafe { sys::cuMemUnmap(va, size) }, sys::CUresult::CUDA_SUCCESS, "unmap");
        assert_eq!(unsafe { sys::cuMemRelease(handle) }, sys::CUresult::CUDA_SUCCESS, "release");
        let (free_released, _) = free_bytes();
        let returned = free_released.saturating_sub(free_mapped);
        println!(
            "VMM: {:.0} MiB reserved; free after map {:.0} MiB, after unmap+release {:.0} MiB (+{:.0})",
            size as f64 / 1048576.0,
            free_mapped as f64 / 1048576.0,
            free_released as f64 / 1048576.0,
            returned as f64 / 1048576.0,
        );
        assert!(
            returned + (32 << 20) >= size,
            "unmapping should return the physical memory to the driver"
        );

        // The stream-ordered pool every activation comes from must be able to take it.
        let stream = device.as_cuda_device().expect("dev").cuda_stream();
        let block = unsafe { stream.alloc::<u8>(size) }.expect("malloc_async from the freed memory");
        println!("VMM: cuMemAllocAsync of {:.0} MiB from the pool: ok", size as f64 / 1048576.0);
        drop(block); // returns to the pool

        // And the cache can have its range back, at the same address.
        let handle = map_all(va);
        // Write through the restored mapping and read it back, so the mapping is proven live, not just returned.
        let host = vec![0x5au8; 4096];
        // SAFETY: 4 KiB at the start of the mapped range.
        let rc = unsafe {
            sys::cuMemcpyHtoD_v2(va, host.as_ptr() as *const _, host.len())
        };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "memcpy H2D into the restored mapping");
        let mut back = vec![0u8; 4096];
        // SAFETY: the same 4 KiB.
        let rc = unsafe { sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut _, va, back.len()) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "memcpy D2H out of the restored mapping");
        assert_eq!(back, host, "the restored mapping must be usable");
        println!("VMM: remapped at the same address and read back {:02x}", back[0]);

        // Clean up.
        // SAFETY: the range is mapped and the handle is live.
        let _ = unsafe { sys::cuMemUnmap(va, size) };
        let _ = unsafe { sys::cuMemRelease(handle) };
        let _ = unsafe { sys::cuMemAddressFree(va, size) };
    }
}
