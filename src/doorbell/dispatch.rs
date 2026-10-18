//! Device-side resident dispatch buffers (module docs: `super`).

use std::ffi::c_void;

use candle_core::{Result, Tensor};

use super::ffi;

/// M6 P2: the device-side dispatch buffers - a per-layer resident bitmap plus the three lists the
/// dispatch kernel writes (resident, staged, CPU) and their counts. All of it is mapped host memory, so
/// the host reads a count and one compact list instead of hashing every selected id itself.
pub struct DispatchBuf {
    host: *mut c_void,
    dev_view: *mut c_void,
    layers: usize,
    max_hits: usize,
}

// SAFETY: a cudaHostAllocMapped allocation; the host side is plain memory and the device side is only
// touched by the kernels launched on the calling stream.
unsafe impl Send for DispatchBuf {}
unsafe impl Sync for DispatchBuf {}

impl DispatchBuf {
    /// `layers` bitmaps of 256 experts each, and three lists of at most `max_hits` entries (one per
    /// selected expert, so `rows * top_k` is the natural bound). `Ok(None)` off CUDA or on allocation
    /// failure; the caller then keeps the host-side path.
    pub fn acquire(device: &candle_core::Device, layers: usize, max_hits: usize) -> Result<Option<Self>> {
        if !device.is_cuda() || layers == 0 || max_hits == 0 {
            return Ok(None);
        }
        let bytes = Self::words(layers, max_hits) * 4;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: fresh mapped allocation of `bytes`; freed in `Drop`.
        let rc = unsafe { ffi::doorbell_alloc_mapped(&mut host, bytes) };
        if rc != 0 || host.is_null() {
            return Ok(None);
        }
        let mut dev_view: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` holds a live cudaHostAllocMapped allocation.
        unsafe { ffi::doorbell_get_device_ptr(host, &mut dev_view) };
        if dev_view.is_null() {
            // SAFETY: allocated above, not freed elsewhere yet.
            unsafe { ffi::doorbell_free_host(host) };
            return Ok(None);
        }
        // Zero the whole mapping: the bitmap starts empty (nothing resident) and the counts at 0, so a
        // reader that runs before the first dispatch sees "no lists", never garbage.
        // SAFETY: inside the allocation.
        unsafe { std::ptr::write_bytes(host, 0, bytes) };
        Ok(Some(Self { host: host.cast(), dev_view, layers, max_hits }))
    }

    /// `[bitmap_words, ...word offsets..., total]`. One place, so acquire/dispatch/read cannot drift.
    /// Words one layer's list block needs: three lists of `(ids, row, weights)` plus three counts.
    const fn block_words(l: usize) -> usize {
        // three lists of (ids, row, weights) + three counts + seq/done
        9 * l + 5
    }

    /// Absolute word offsets for `layer`'s block. The bitmap lives at the front, shared; **every list is
    /// per layer**, because a shared list lets a layer read the previous layer's (larger) result - that is
    /// what made the first flip read a stale count and index out of bounds.
    fn offsets(&self, layer: usize) -> [usize; 16] {
        let l = self.max_hits;
        let b = self.layers * 8 + layer * Self::block_words(l);
        [
            layer * 8,                  // [0]  this layer's bitmap
            b,                          // [1]  hits ids
            b + l,                      // [2]  hits row
            b + 2 * l,                  // [3]  hits weights
            b + 3 * l,                  // [4]  hits count
            b + 3 * l + 1,              // [5]  stream ids
            b + 4 * l + 1,              // [6]  stream row
            b + 5 * l + 1,              // [7]  stream weights
            b + 6 * l + 1,              // [8]  stream count
            b + 6 * l + 2,              // [9]  cpu ids
            b + 7 * l + 2,              // [10] cpu row
            b + 8 * l + 2,              // [11] cpu weights
            b + 9 * l + 2,              // [12] cpu count
            b + 9 * l + 3,              // [13] seq (host writes before the launch)
            b + 9 * l + 4,              // [14] done (the kernel publishes it last)
            b + 9 * l + 5,              // [15] end of this block
        ]
    }

    fn words(layers: usize, max_hits: usize) -> usize {
        layers * 8 + layers * Self::block_words(max_hits)
    }

    fn fword(&self, off: usize) -> *mut f32 {
        // SAFETY: same mapping, word-aligned; f32 and u32 share the alignment.
        unsafe { self.host.cast::<f32>().add(off) }
    }

    fn word(&self, off: usize) -> *mut u32 {
        // SAFETY: `off` is within the allocation by construction of `layout`.
        unsafe { self.host.cast::<u32>().add(off) }
    }

    /// The host-side bitmap, one word per 32 experts per layer. Write it when a slot is published or
    /// evicted, then launch with [`Self::dispatch`] on the same stream so the kernel sees the change.
    pub fn bitmap_mut(&mut self, layer: usize) -> &mut [u32] {
        assert!(layer < self.layers, "dispatch bitmap layer out of range");
        // SAFETY: eight u32 words per layer inside the mapping.
        unsafe { std::slice::from_raw_parts_mut(self.word(layer * 8), 8) }
    }

    /// Run the dispatch kernel for one layer. `ids`/`weights` are the layer's selection on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        layer: usize,
        ids: &Tensor,
        weights: &Tensor,
        n: usize,
        top_k: usize,
        frac_num: i32,
        frac_den: i32,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dev = ids.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let ids32 = ids.to_dtype(candle_core::DType::U32)?.contiguous()?;
        let w32 = weights.to_dtype(candle_core::DType::F32)?.contiguous()?;
        let (ids_storage, ids_layout) = ids32.storage_and_layout();
        let candle_core::Storage::Cuda(ids_s) = &*ids_storage else {
            candle_core::bail!("dispatch: ids not on CUDA");
        };
        let ids_slice = ids_s.as_cuda_slice::<u32>()?;
        let ids_view = ids_slice.slice(ids_layout.start_offset()..);
        let (ids_ptr, _ids_guard) = ids_view.device_ptr(&stream);
        let (w_storage, w_layout) = w32.storage_and_layout();
        let candle_core::Storage::Cuda(w_s) = &*w_storage else {
            candle_core::bail!("dispatch: weights not on CUDA");
        };
        let w_slice = w_s.as_cuda_slice::<f32>()?;
        let w_view = w_slice.slice(w_layout.start_offset()..);
        let (w_ptr, _w_guard) = w_view.device_ptr(&stream);

        assert!(layer < self.layers, "dispatch layer out of range");
        let o = self.offsets(layer);
        // Host-written seq for this launch; the kernel publishes it into `done` after its last store, so
        // `wait` can spin on one word instead of synchronising the device.
        let seq = self.next_seq(layer);
        // SAFETY: mapped word inside the allocation; the host owns it.
        unsafe { self.word(o[13]).write(seq) };
        let bitmap = self.word(o[0]).cast::<c_void>();
        // SAFETY: every pointer is inside this mapping except the caller's two device tensors, which the
        // launch is stream-ordered after.
        unsafe {
            ffi::doorbell_dispatch_topk(
                ids_ptr as *const c_void,
                w_ptr as *const c_void,
                n as i32,
                bitmap,
                top_k as i32,
                frac_num,
                frac_den,
                self.word(o[1]).cast(),
                self.word(o[2]).cast(),
                self.fword(o[3]).cast(),
                self.word(o[4]).cast(),
                self.word(o[5]).cast(),
                self.word(o[6]).cast(),
                self.fword(o[7]).cast(),
                self.word(o[8]).cast(),
                self.word(o[9]).cast(),
                self.word(o[10]).cast(),
                self.fword(o[11]).cast(),
                self.word(o[12]).cast(),
                self.max_hits as i32,
                seq,
                self.word(o[14]).cast(),
                stream.cu_stream() as i64,
            );
        }
        Ok(())
    }

    /// Monotonic per-layer seq, so `wait` can tell this launch's `done` from the previous one.
    fn next_seq(&self, layer: usize) -> u32 {
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let _ = layer;
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
    }

    /// Spin (host) until the layer's dispatch has published its lists. This replaces `device.synchronize()`
    /// in the flip path: one mapped word, the same protocol the top-k ring uses.
    pub fn wait(&self, layer: usize) {
        let o = self.offsets(layer);
        let want = unsafe { *self.word(o[13]) };
        let done = self.word(o[14]);
        // SAFETY: a mapped u32 the kernel published behind a system fence.
        while unsafe { std::ptr::read_volatile(done) } != want {
            std::hint::spin_loop();
        }
        // Ordering: the lists were written before `done`, and the kernel fenced before publishing it.
    }

    /// The last dispatch's decision: resident `(expert id, weight)`, staged ids with their weights, and the
    /// ids the CPU should compute. Safe once the stream passed the dispatch (the launch ends with a system
    /// fence).
    pub fn read(
        &self,
        layer: usize,
    ) -> (Vec<(u32, u32, f32)>, Vec<(u32, u32, f32)>, Vec<(u32, u32, f32)>) {
        let o = self.offsets(layer);
        let count = |off: usize| {
            // SAFETY: `off` is the count word of one list.
            (unsafe { *self.word(off) } as usize).min(self.max_hits)
        };
        let list = |ids: usize, row: usize, ws: usize, n: usize| {
            (0..n)
                .map(|i| {
                    // SAFETY: within the id, row and weight lists of one list.
                    unsafe { (*self.word(ids + i), *self.word(row + i), *self.fword(ws + i)) }
                })
                .collect::<Vec<_>>()
        };
        (
            list(o[1], o[2], o[3], count(o[4])),
            list(o[5], o[6], o[7], count(o[8])),
            list(o[9], o[10], o[11], count(o[12])),
        )
    }
}

impl Drop for DispatchBuf {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: allocated by doorbell_alloc_mapped, freed once.
            unsafe { ffi::doorbell_free_host(self.host) };
        }
    }
}
