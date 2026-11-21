//! Layer buffer lifecycle: allocation, cache-slab borrowing, chunk begin/zero/empty, upload,
//! first-chunk re-planning, and CUDA-OOM buffer downgrade.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::QTensor;
use candle_core::Device;
use mistralrs_core::MoEExperts;
use tracing::{debug, info, warn};

use crate::copy_stream::{trim_device_pool, CopyDone, ReadFence, SlotMemory};

use super::planner::{
    activation_bytes, chunk_scratch, kv_reserve, log_plan, plan_with_lend_depth, plan_with_reserve,
    prompt_step_ceiling, Plan, MAX_HEADROOM, PLAN_GRID,
};
use super::PrefillStream;

/// One layer's host stacks (gate, up, down). Holding them keeps the memory that in-flight
/// uploads read alive.
pub(super) struct HostStacks(pub(super) [Arc<QTensor>; 3]);

pub(super) struct LayerBuf {
    /// The layer uploaded (or being uploaded) into this buffer.
    pub(super) loaded: Option<usize>,
    /// Transport probe: what the last upload moved and when it was queued.
    pub(super) queued: Option<Instant>,
    pub(super) bytes: usize,
    pub(super) copies: usize,
    pub(super) memory: Option<[SlotMemory; 3]>,
    pub(super) experts: Option<MoEExperts>,
    /// Recorded on candle's stream after the kernels that read this buffer.
    pub(super) released: ReadFence,
    /// Recorded on the copy stream after the upload into this buffer.
    pub(super) ready: CopyDone,
}

impl PrefillStream {
    pub(super) fn stacks(&self, layer: usize) -> Option<[Arc<QTensor>; 3]> {
        self.layers
            .read()
            .get(layer)
            .and_then(|s| s.as_ref())
            .map(|s| s.0.clone())
    }

    /// Bytes of the largest registered layer's three projections, plus `ROW_PADDING_MARGIN`
    /// covering `MATRIX_ROW_PADDING` and 256-byte alignment inside borrowed ranges (M35 §15).
    pub(super) fn layer_bytes(&self) -> usize {
        const ROW_PADDING_MARGIN: usize = 4 << 20;
        self.layers
            .read()
            .iter()
            .flatten()
            .map(|s| {
                s.0.iter().map(|t| t.storage_size_in_bytes()).sum::<usize>() + ROW_PADDING_MARGIN
            })
            .max()
            .unwrap_or(0)
    }

    /// What the pool can lend: its size when the stream borrows, 0 when the buffers are its own.
    fn lend_bytes(&self) -> usize {
        if !self.borrows() {
            return 0;
        }
        self.slab.as_ref().map_or(0, |slab| slab.bytes())
    }

    /// Whether this stream's layer buffers come from the cache's slab.
    pub(super) fn borrows(&self) -> bool {
        crate::moe::slab::Slab::borrowing() && self.slab.is_some() && self.cache.is_some()
    }

    /// Measure now and return what fits, or `None` if nothing does.
    pub(super) fn decide(&self) -> Option<(Plan, usize)> {
        let free = crate::copy_stream::free_bytes(&self.device).unwrap_or(0);
        let (stride, chunk) = plan_with_lend_depth(
            free,
            self.layer_bytes(),
            prompt_step_ceiling(),
            self.borrows(),
            kv_reserve(),
            self.lend_bytes(),
            self.depth,
        )?;
        Some((Plan { stride, chunk }, free))
    }

    /// The plan in force, or `(stride 0, chunk 0)` if it did not fit.
    pub(super) fn current(&self) -> Plan {
        (*self.plan.lock().unwrap()).unwrap_or(Plan {
            stride: 0,
            chunk: 0,
        })
    }

    pub(super) fn is_refused(&self) -> bool {
        self.refused.load(Ordering::Acquire)
    }

    fn allow_cpu() -> bool {
        crate::config::runtime().prefill_allow_cpu
    }

    fn strict() -> bool {
        crate::config::runtime().prefill_strict
    }

    /// Three 256-byte-aligned projection buffers carved from the slab's low end.
    fn borrow_buffers(
        &self,
        layer: usize,
        buf: usize,
        stacks: &[Arc<QTensor>; 3],
    ) -> candle_core::Result<[SlotMemory; 3]> {
        let slab = self.slab.as_ref().expect("checked by borrows()");
        let mut out = Vec::with_capacity(3);
        let span = (self.layer_bytes() + (2 << 20) - 1) & !((2 << 20) - 1);
        let mut offset = buf * span;
        for stack in stacks {
            let padded =
                stack.shape().elem_count() + candle_core::quantized::cuda::MATRIX_ROW_PADDING;
            let storage = slab.borrow_at(offset, padded, stack.dtype(), &self.device)?;
            // 256-byte aligned, matching `alloc_like` (M35 §16).
            offset = (offset + storage.allocated_bytes() + 255) & !255;
            out.push(self.copy.like_from_storage(stack, storage)?);
        }
        if crate::config::diag().layer_dbg {
            let used = offset - buf * span;
            let kinds: Vec<String> = stacks
                .iter()
                .map(|s| {
                    format!(
                        "{:?}/{}el/{}b",
                        s.dtype(),
                        s.shape().elem_count(),
                        s.storage_size_in_bytes()
                    )
                })
                .collect();
            eprintln!(
                "LAYOUT layer={layer} buf={buf} span={span} used={used} over={} pad={} {}",
                used as i64 - span as i64,
                candle_core::quantized::cuda::MATRIX_ROW_PADDING,
                kinds.join(" ")
            );
        }
        Ok([out.remove(0), out.remove(0), out.remove(0)])
    }

    /// Once-per-chunk work at layer 0: raise the borrow gate, order the copy stream after the
    /// cache's queued work, zero the span, empty overlapping cache slots, and sync prior fills.
    pub(super) fn begin_chunk(&self, layer: usize) -> candle_core::Result<()> {
        if !self.borrows() || layer != 0 {
            return Ok(());
        }
        let Some(slab) = self.slab.as_ref() else {
            return Ok(());
        };
        let stride = self.current().stride.max(1);
        let span = (self.layer_bytes() + (2 << 20) - 1) & !((2 << 20) - 1);
        let loaned = stride * span;
        {
            let _guard = self.borrow_gate.lock_access();
            self.borrow_gate.set_borrowed(true);
        }
        let base = slab.base()?;
        self.copy.after_device_work()?;
        let stream = self.copy.cuda_stream();
        let zeroed = loaned.max(650usize << 20).min(slab.bytes());
        // SAFETY: the span is inside the live slab; memset is queued before any upload.
        unsafe {
            let rc = candle_core::cuda::cudarc::driver::sys::cuMemsetD8Async(
                base,
                0,
                zeroed,
                stream.cu_stream(),
            );
            if rc != candle_core::cuda::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                candle_core::bail!("memset of the borrowed span failed: {rc:?}");
            }
        }
        if let Some(cache) = self.cache.as_ref() {
            let emptied = cache.empty_range(base, base + loaned as u64);
            self.copy.synchronize()?;
            crate::dbg_note(
                "prefill_borrow",
                &format!(
                    "chunk: borrowed {:.0} MiB of the pool in {stride} range(s), {emptied} slots emptied",
                    (loaned as f64) / (1u64 << 20) as f64
                ),
            );
        }
        Ok(())
    }

    /// Re-check the warmup plan once on the first long chunk and publish any smaller prompt step.
    pub(super) fn reserve(&self) -> candle_core::Result<usize> {
        let first = !self.checked.swap(true, Ordering::AcqRel);
        if !first {
            return Ok(self.current().stride);
        }
        let previous = self.current();
        match self.decide() {
            Some((now, free)) => {
                let decision = Plan {
                    stride: now.stride.min(previous.stride.max(1)),
                    chunk: now.chunk.min(match previous.chunk {
                        0 => now.chunk,
                        c => c,
                    }),
                };
                *self.plan.lock().unwrap() = Some(decision);
                log_plan(self, decision, free, "first chunk");
                if decision.chunk < previous.chunk {
                    mistralrs_core::PROMPT_STEP_CEILING.store(decision.chunk, Ordering::Release);
                    info!(
                        "prefill: the prompt step drops from {} to {} tokens for the rest of the \
                         session",
                        previous.chunk, decision.chunk
                    );
                }
                Ok(decision.stride)
            }
            None => {
                let mib = |b: usize| (b as f64) / (1u64 << 20) as f64;
                let free = crate::copy_stream::free_bytes(&self.device).unwrap_or(0);
                let floor = if self.borrows() {
                    chunk_scratch(PLAN_GRID) + MAX_HEADROOM + kv_reserve()
                } else {
                    self.layer_bytes() + chunk_scratch(PLAN_GRID) + MAX_HEADROOM + kv_reserve()
                };
                let give_up = {
                    let mut r = 0usize;
                    while r < (4096 << 20) {
                        if plan_with_reserve(
                            free + r,
                            self.layer_bytes(),
                            prompt_step_ceiling(),
                            self.borrows(),
                            kv_reserve(),
                        )
                        .is_some()
                        {
                            break;
                        }
                        r += 64 << 20;
                    }
                    r
                };
                warn!(
                    "prefill stream: off ({} + {:.0} MiB chunk scratch for a {}-row chunk + \
                     {:.0} MiB headroom + {:.0} MiB KV reserve = {:.0} MiB does not fit in {:.0} MiB free, of \
                     which the KV will take its own {:.0}). The whole prompt then runs on the CPU, 41x slower: \
                     drop --expert-cache-gb by at least {:.2} GB (or drop --prefill-stream, or set \
                     NAIWA_PREFILL_ALLOW_CPU=1 to accept the slow path)",
                    if self.borrows() {
                        "no layer buffer (the pool lends them)".to_string()
                    } else {
                        format!("one {:.0} MiB layer buffer", mib(self.layer_bytes()))
                    },
                    mib(chunk_scratch(previous.chunk.max(PLAN_GRID))),
                    previous.chunk.max(PLAN_GRID),
                    mib(MAX_HEADROOM),
                    mib(kv_reserve()),
                    mib(floor),
                    mib(free),
                    mib(kv_reserve()),
                    (give_up as f64) / (1024.0 * 1024.0 * 1024.0),
                );
                if !Self::allow_cpu() {
                    self.refused.store(true, Ordering::Release);
                    candle_core::bail!(
                        "prefill stream does not fit - see the warning above for the --expert-cache-gb to \
                         drop to. Set NAIWA_PREFILL_ALLOW_CPU=1 to run the prompt on the CPU anyway (41x \
                         slower), or drop --prefill-stream"
                    );
                }
                if Self::strict() {
                    candle_core::bail!(
                        "prefill stream: off (one {:.0} MiB layer buffer + {:.0} MiB activations + \
                         {:.0} MiB headroom does not fit in {:.0} MiB free), and NAIWA_PREFILL_STRICT is \
                         set. Lower --expert-cache-gb to leave room for the stream, or unset \
                         NAIWA_PREFILL_STRICT to let the CPU take the chunk (33x slower)",
                        mib(self.layer_bytes()),
                        mib(activation_bytes(previous.chunk.max(256))),
                        mib(MAX_HEADROOM),
                        mib(free),
                    );
                }
                *self.plan.lock().unwrap() = Some(Plan {
                    stride: 0,
                    chunk: 0,
                });
                Ok(0)
            }
        }
    }

    /// Give up one layer buffer after a CUDA OOM, trim the device pool, and continue.
    pub(super) fn downgrade(&self, stride: usize) -> candle_core::Result<()> {
        if Self::strict() {
            candle_core::bail!(
                "prefill stream: a buffer was refused with CUDA OOM (stride {stride}) and \
                 NAIWA_PREFILL_STRICT is set; the stream would otherwise continue with fewer buffers"
            );
        }
        let next = stride.saturating_sub(1);
        let chunk = self.current().chunk;
        *self.plan.lock().unwrap() = Some(Plan {
            stride: next,
            chunk,
        });
        for buf in &self.bufs {
            let mut buf = buf.lock().unwrap();
            buf.loaded = None;
            buf.experts = None;
            buf.memory = None;
        }
        trim_device_pool(&self.device)?;
        warn!(
            "prefill stream: out of device memory with {stride} layer buffer(s); continuing with \
             {next} — this chunk's remaining layers and the next chunk's first layers stay on the \
             CPU"
        );
        Ok(())
    }

    /// Queue the upload of `layer` into its buffer, unless it is already loaded.
    pub(super) fn load(&self, layer: usize, stride: usize) -> candle_core::Result<bool> {
        let Some(stacks) = self.stacks(layer) else {
            return Ok(false);
        };
        let mut buf = self.bufs[layer % stride.max(1)].lock().unwrap();
        if buf.loaded == Some(layer) {
            return Ok(true);
        }
        buf.loaded = None;
        let fits = buf.memory.as_ref().is_some_and(|m| {
            m.iter()
                .zip(&stacks)
                .all(|(m, s)| m.tensor().dtype() == s.dtype() && m.tensor().shape() == s.shape())
        });
        if !fits {
            let t_realloc = Instant::now();
            if buf.memory.is_some() {
                crate::dbg_note(
                    "prefill_load",
                    &format!("layer {layer}: waiting for the old upload"),
                );
                buf.ready.wait_on(&self.device)?;
                crate::dbg_note(
                    "prefill_load",
                    &format!("layer {layer}: old upload settled"),
                );
            }
            buf.experts = None;
            buf.memory = None;
            trace_buffer_bytes(&self.device, layer, &stacks);
            crate::dbg_note(
                "prefill_load",
                &format!(
                    "layer {layer}: allocating ({} bytes/stack)",
                    stacks[0].storage_size_in_bytes()
                ),
            );
            let memory = if self.borrows() {
                let which = layer % self.current().stride.max(1);
                self.borrow_buffers(layer, which, &stacks)?
            } else {
                [
                    self.copy.alloc_like(&stacks[0])?,
                    self.copy.alloc_like(&stacks[1])?,
                    self.copy.alloc_like(&stacks[2])?,
                ]
            };
            self.copy.after_device_work()?;
            let bytes: usize = stacks.iter().map(|s| s.storage_size_in_bytes()).sum();
            debug!(
                "prefill stream: {} MiB buffer for layer {layer}'s experts",
                bytes >> 20
            );
            buf.experts = Some(MoEExperts::from_qtensors(
                Arc::clone(memory[0].tensor()),
                Arc::clone(memory[1].tensor()),
                Arc::clone(memory[2].tensor()),
                self.num_experts,
                self.top_k,
                &self.comm,
                self.act,
            )?);
            buf.memory = Some(memory);
            crate::dbg_note(
                "prefill_realloc",
                &format!("layer {layer} shape change took {:?}", t_realloc.elapsed()),
            );
        }
        let data = [stacks[0].data()?, stacks[1].data()?, stacks[2].data()?];
        buf.bytes = data.iter().map(|d| d.len()).sum();
        buf.copies = data.len();
        let memory = buf.memory.as_ref().expect("allocated above");
        let parts = [
            (&memory[0], &data[0][..]),
            (&memory[1], &data[1][..]),
            (&memory[2], &data[2][..]),
        ];
        let queued = Instant::now();
        crate::dbg_note(
            "prefill_load",
            &format!("layer {layer}: queueing {} bytes", buf.bytes),
        );
        // SAFETY: registered stacks are page-locked, and `self.layers` keeps them alive
        // past `Drop` (`self.copy.synchronize()`).
        unsafe {
            self.copy.upload_async(&buf.released, &parts, &buf.ready)?;
        }
        buf.queued = Some(queued);
        buf.loaded = Some(layer);
        Ok(true)
    }
}

/// Debug line at a mixed-quant reallocation: how big the buffer is and how much was free.
fn trace_buffer_bytes(device: &Device, layer: usize, stacks: &[Arc<QTensor>; 3]) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let bytes: usize = stacks.iter().map(|s| s.storage_size_in_bytes()).sum();
    let free = device
        .as_cuda_device()
        .ok()
        .and_then(|d| {
            let stream = d.cuda_stream();
            stream.context().mem_get_info().ok()
        })
        .map_or(-1.0, |(f, _)| (f as f64) / (1u64 << 20) as f64);
    debug!(
        "prefill stream: {:.0} MiB buffer for layer {layer} ({free:.0} MiB free before it)",
        (bytes as f64) / (1u64 << 20) as f64
    );
}

/// A CUDA out-of-memory error, however deep in the wrapper it is.
pub(super) fn is_out_of_memory(error: &candle_core::Error) -> bool {
    let text = error.to_string();
    text.contains("OUT_OF_MEMORY") || text.contains("out of memory")
}
