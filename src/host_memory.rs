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
