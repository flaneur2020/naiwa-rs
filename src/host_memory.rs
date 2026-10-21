//! Host-side placement of the routed-expert stacks.
//!
//! - **Huge pages** (Linux): the stacks are asked to be backed by 2 MiB transparent huge pages,
//!   so the CPU's expert GEMVs walk them without a TLB miss per 4 KiB. Applies to every
//!   `--cpu-moe` layer.
//! - **Pinning** (CUDA): with `--expert-cache-gb`, the stacks are page-locked with
//!   `cuMemHostRegister`. A slot fill then copies straight from the stack by DMA. Copying from
//!   pageable memory instead stages every copy through the driver and synchronizes the stream
//!   before it starts.
//!
//! Both steps are best effort: when one fails, the stack stays where it is and everything still
//! works, only slower. Huge pages come first, because a pinned page can no longer be collapsed.

use candle_core::quantized::QTensor;
use candle_core::Device;
use mistralrs_core::MoEExperts;

const HUGE_PAGE: usize = 2 << 20;

/// The whole huge pages inside `[addr, addr + len)`, as `(start, len)`.
fn huge_page_span(addr: usize, len: usize) -> Option<(usize, usize)> {
    let start = addr.checked_next_multiple_of(HUGE_PAGE)?;
    let end = addr.checked_add(len)? / HUGE_PAGE * HUGE_PAGE;
    (end > start).then(|| (start, end - start))
}

/// Ask the kernel to back `bytes` with huge pages and collapse the pages already populated.
/// Only whole huge pages inside `bytes` are touched. Returns the bytes advised; 0 off Linux or
/// when the kernel refuses.
pub fn advise_huge_pages(bytes: &[u8]) -> usize {
    let Some((start, len)) = huge_page_span(bytes.as_ptr() as usize, bytes.len()) else {
        return 0;
    };
    #[cfg(target_os = "linux")]
    {
        // The kernel ABI value (Linux 6.1+); libc only defines it for glibc.
        const MADV_COLLAPSE: libc::c_int = 25;
        let ptr = start as *mut libc::c_void;
        // SAFETY: [start, start + len) lies inside `bytes`, and neither advice changes contents.
        unsafe {
            if libc::madvise(ptr, len, libc::MADV_HUGEPAGE) != 0 {
                return 0;
            }
            // Older kernels refuse; khugepaged then collapses the range in the background.
            libc::madvise(ptr, len, MADV_COLLAPSE);
        }
        len
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = start;
        let _ = len;
        0
    }
}

/// The host bytes of a CPU-resident stack; `None` for device storage, whose `data()` is a copy.
fn host_bytes(stack: &QTensor) -> candle_core::Result<Option<&[u8]>> {
    Ok(match stack.data()? {
        std::borrow::Cow::Borrowed(bytes) => Some(bytes),
        std::borrow::Cow::Owned(_) => None,
    })
}

/// Huge-page every expert stack of `experts`. Returns the bytes advised.
pub fn advise_expert_stacks(experts: &MoEExperts) -> candle_core::Result<usize> {
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return Ok(0);
    };
    let mut advised = 0;
    for stack in [&gate, &up, &down] {
        if let Some(bytes) = host_bytes(stack)? {
            advised += advise_huge_pages(bytes);
        }
    }
    Ok(advised)
}

/// Pin every expert stack of `experts` for copies to `device`. Stacks that cannot be pinned are
/// left pageable and reported in the error count.
pub fn pin_expert_stacks(experts: &MoEExperts, device: &Device) -> (Vec<PinnedStack>, usize) {
    let mut pinned = Vec::new();
    let mut failed = 0;
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return (pinned, failed);
    };
    for stack in [gate, up, down] {
        match pin_stack(device, stack) {
            Ok(Some(p)) => pinned.push(p),
            Ok(None) => {}
            Err(e) => {
                tracing::debug!("expert stack left pageable: {e}");
                failed += 1;
            }
        }
    }
    (pinned, failed)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::c_void;
    use std::sync::Arc;

    use candle_core::cuda::cudarc::driver::{sys, CudaContext};
    use candle_core::quantized::QTensor;
    use candle_core::Device;

    /// A page-locked expert stack. Holds the stack, so its memory cannot be freed while it is
    /// registered; unregisters on drop.
    pub struct PinnedStack {
        ctx: Arc<CudaContext>,
        ptr: *mut c_void,
        len: usize,
        _stack: Arc<QTensor>,
    }

    // SAFETY: `ptr` is only handed back to the driver, to unregister it.
    unsafe impl Send for PinnedStack {}
    unsafe impl Sync for PinnedStack {}

    impl PinnedStack {
        pub fn len(&self) -> usize {
            self.len
        }
    }

    impl Drop for PinnedStack {
        fn drop(&mut self) {
            if self.ctx.bind_to_thread().is_ok() {
                // SAFETY: `ptr` was registered by `pin_stack`, and `_stack` keeps it allocated.
                unsafe {
                    let _ = sys::cuMemHostUnregister(self.ptr);
                }
            }
        }
    }

    /// Page-lock `stack`'s host memory for `device`. `None` when the device is not CUDA, the
    /// stack is not in host memory, or the memory is already registered by someone else.
    pub fn pin_stack(
        device: &Device,
        stack: Arc<QTensor>,
    ) -> candle_core::Result<Option<PinnedStack>> {
        let Device::Cuda(dev) = device else {
            return Ok(None);
        };
        let ctx = dev.cuda_stream().context().clone();
        let Some(bytes) = super::host_bytes(&stack)? else {
            return Ok(None);
        };
        if bytes.is_empty() {
            return Ok(None);
        }
        let (ptr, len) = (bytes.as_ptr() as *mut c_void, bytes.len());
        ctx.bind_to_thread().map_err(candle_core::Error::wrap)?;
        // PORTABLE: the pages count as pinned in every context, not just this one.
        // SAFETY: [ptr, ptr + len) is the stack's live host buffer, kept alive by `_stack`.
        let r = unsafe { sys::cuMemHostRegister_v2(ptr, len, sys::CU_MEMHOSTREGISTER_PORTABLE) };
        if r == sys::CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED {
            return Ok(None);
        }
        r.result().map_err(candle_core::Error::wrap)?;
        Ok(Some(PinnedStack {
            ctx,
            ptr,
            len,
            _stack: stack,
        }))
    }
}

#[cfg(feature = "cuda")]
use cuda::pin_stack;
#[cfg(feature = "cuda")]
pub use cuda::PinnedStack;

/// Never constructed without CUDA.
#[cfg(not(feature = "cuda"))]
pub enum PinnedStack {}

#[cfg(not(feature = "cuda"))]
impl PinnedStack {
    pub fn len(&self) -> usize {
        match *self {}
    }
}

#[cfg(not(feature = "cuda"))]
fn pin_stack(
    _device: &Device,
    _stack: std::sync::Arc<QTensor>,
) -> candle_core::Result<Option<PinnedStack>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "cuda"))]
    use std::sync::Arc;

    use super::*;

    #[test]
    fn huge_page_span_keeps_whole_pages_inside_the_range() {
        let h = HUGE_PAGE;
        // aligned start, two and a half pages: two whole pages
        assert_eq!(huge_page_span(4 * h, 2 * h + h / 2), Some((4 * h, 2 * h)));
        // unaligned start: rounds up, and the tail is cut
        assert_eq!(huge_page_span(4 * h + 1, 3 * h), Some((5 * h, 2 * h)));
        // smaller than a page, or not spanning a whole aligned page
        assert_eq!(huge_page_span(4 * h, h - 1), None);
        assert_eq!(huge_page_span(4 * h + 1, h), None);
        assert_eq!(huge_page_span(usize::MAX - 10, 100), None);
    }

    #[test]
    fn host_bytes_borrows_cpu_storage() {
        let t = candle_core::Tensor::zeros((4, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, candle_core::quantized::GgmlDType::Q8_0).unwrap();
        let bytes = host_bytes(&qt).unwrap().unwrap();
        assert_eq!(bytes.len(), qt.storage_size_in_bytes());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn nothing_is_pinned_without_cuda() {
        let t = candle_core::Tensor::zeros((4, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, candle_core::quantized::GgmlDType::Q8_0).unwrap();
        assert!(pin_stack(&Device::Cpu, Arc::new(qt)).unwrap().is_none());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod h2d_probe {
    //! `cargo test --release h2d_bandwidth -- --nocapture`: what this box's host-to-device copy actually
    //! reaches, registered (`cuMemHostRegister`, our expert stacks) against plain pageable memory. The
    //! prefill stream measured 4.6-5.5 GB/s on 481 MiB copies; that is either the link or our path, and this
    //! says which.

    use candle_core::cuda::cudarc::driver::sys;

    const MIB: usize = 512;

    fn rate(label: &str, bytes: &[u8], registered: bool) {
        let device = candle_core::Device::new_cuda(0).expect("cuda device");
        let cuda = device.as_cuda_device().expect("cuda");
        let stream = cuda.cuda_stream();
        let mut dst = unsafe { cuda.alloc::<u8>(bytes.len()) }.expect("device buffer");
        if registered {
            unsafe {
                let r = sys::cuMemHostRegister_v2(
                    bytes.as_ptr() as *mut std::ffi::c_void,
                    bytes.len(),
                    sys::CU_MEMHOSTREGISTER_PORTABLE,
                );
                assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "host register");
            }
        }
        // One warm-up copy so the first-touch/paging costs are not in the measurement.
        stream.memcpy_htod(bytes, &mut dst).expect("warmup copy");
        stream.synchronize().expect("sync");
        let start = std::time::Instant::now();
        const ROUNDS: usize = 4;
        for _ in 0..ROUNDS {
            stream.memcpy_htod(bytes, &mut dst).expect("copy");
        }
        stream.synchronize().expect("sync");
        let elapsed = start.elapsed().as_secs_f64() / ROUNDS as f64;
        let gbs = bytes.len() as f64 / elapsed / 1e9;
        eprintln!(
            "H2D {label:10} {:4} MiB: {:.1} ms, {gbs:.1} GB/s",
            bytes.len() >> 20,
            elapsed * 1e3
        );
        if registered {
            unsafe {
                sys::cuMemHostUnregister(bytes.as_ptr() as *mut std::ffi::c_void);
            }
        }
    }

    /// The prefill stream's real shape: one layer is three separate stacks (gate/up/down), copied as three
    /// `memcpy`s in one batch, from THP-advised and registered buffers.
    fn rate_parts(label: &str, parts: &[Vec<u8>], registered: bool, advise_huge: bool) {
        let device = candle_core::Device::new_cuda(0).expect("cuda device");
        let cuda = device.as_cuda_device().expect("cuda");
        let stream = cuda.cuda_stream();
        let mut dsts: Vec<_> = parts
            .iter()
            .map(|p| unsafe { cuda.alloc::<u8>(p.len()) }.expect("device buffer"))
            .collect();
        let total: usize = parts.iter().map(|p| p.len()).sum();
        if advise_huge {
            for p in parts {
                super::advise_huge_pages(p);
            }
        }
        if registered {
            for p in parts {
                unsafe {
                    let r = sys::cuMemHostRegister_v2(
                        p.as_ptr() as *mut std::ffi::c_void,
                        p.len(),
                        sys::CU_MEMHOSTREGISTER_PORTABLE,
                    );
                    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "host register");
                }
            }
        }
        let mut copy = |dsts: &mut Vec<_>| {
            for (dst, src) in dsts.iter_mut().zip(parts) {
                stream.memcpy_htod(src, dst).expect("copy");
            }
            stream.synchronize().expect("sync");
        };
        copy(&mut dsts);
        let start = std::time::Instant::now();
        const ROUNDS: usize = 4;
        for _ in 0..ROUNDS {
            copy(&mut dsts);
        }
        let elapsed = start.elapsed().as_secs_f64() / ROUNDS as f64;
        eprintln!(
            "H2D {label:22} {:4} MiB x {}: {:.1} ms, {:.1} GB/s",
            total >> 20,
            parts.len(),
            elapsed * 1e3,
            total as f64 / elapsed / 1e9
        );
        if registered {
            for p in parts {
                unsafe {
                    sys::cuMemHostUnregister(p.as_ptr() as *mut std::ffi::c_void);
                }
            }
        }
    }

    /// Same bytes, same 3-part shape, registered: candle's default stream against a second stream built the
    /// way `CopyStream` builds one. The prefill stream copies on that second stream and measures 4.6-5.5 GB/s
    /// where the default stream reaches 13.4.
    #[test]
    #[ignore = "instrument: cargo test h2d_ -- --ignored --nocapture"]
    fn h2d_second_stream() {
        let device = candle_core::Device::new_cuda(0).expect("cuda device");
        let cuda = device.as_cuda_device().expect("cuda");
        let third = (MIB << 20) / 3;
        let parts: Vec<Vec<u8>> = (0..3).map(|_| vec![7u8; third]).collect();
        for p in &parts {
            unsafe {
                let r = sys::cuMemHostRegister_v2(
                    p.as_ptr() as *mut std::ffi::c_void,
                    p.len(),
                    sys::CU_MEMHOSTREGISTER_PORTABLE,
                );
                assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "host register");
            }
        }
        let mut dsts: Vec<_> = parts
            .iter()
            .map(|p| unsafe { cuda.alloc::<u8>(p.len()) }.expect("device buffer"))
            .collect();
        let second = cuda.cuda_stream().context().new_stream().expect("second stream");
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let default = cuda.cuda_stream().clone();
        for (label, stream) in [("default", &default), ("second", &second)] {
            let copy = |dsts: &mut Vec<_>| {
                for (dst, src) in dsts.iter_mut().zip(&parts) {
                    stream.memcpy_htod(src, dst).expect("copy");
                }
                stream.synchronize().expect("sync");
            };
            copy(&mut dsts);
            let start = std::time::Instant::now();
            const ROUNDS: usize = 4;
            for _ in 0..ROUNDS {
                copy(&mut dsts);
            }
            let elapsed = start.elapsed().as_secs_f64() / ROUNDS as f64;
            eprintln!(
                "H2D {label:8} stream {:4} MiB x 3: {:.1} ms, {:.1} GB/s",
                total >> 20,
                elapsed * 1e3,
                total as f64 / elapsed / 1e9
            );
        }
        for p in &parts {
            unsafe {
                sys::cuMemHostUnregister(p.as_ptr() as *mut std::ffi::c_void);
            }
        }
    }

    #[test]
    #[ignore = "instrument: cargo test h2d_ -- --ignored --nocapture"]
    fn h2d_three_parts() {
        let third = (MIB << 20) / 3;
        let parts: Vec<Vec<u8>> = (0..3).map(|_| vec![7u8; third]).collect();
        rate_parts("3 parts pageable", &parts, false, false);
        rate_parts("3 parts registered", &parts, true, false);
        rate_parts("3 parts reg+huge", &parts, true, true);
    }

    #[test]
    #[ignore = "instrument: cargo test h2d_ -- --ignored --nocapture"]
    fn h2d_bandwidth() {
        let pageable = vec![7u8; MIB << 20];
        rate("pageable", &pageable, false);
        drop(pageable);
        // Same size, same content, registered: the shape of our expert stacks.
        let mut pinned = vec![7u8; MIB << 20];
        rate("registered", &pinned, true);
        pinned.fill(9);
        rate("registered2", &pinned, true);
    }
}
