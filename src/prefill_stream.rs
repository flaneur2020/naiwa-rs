//! `--prefill-stream`: run prefill's routed experts on the GPU, uploading each layer's experts
//! while the previous layer computes.
//!
//! Two device buffers each hold one whole layer of experts: the three stacked `[E, N, K]`
//! projections, uploaded verbatim from the pinned host stacks (one copy per projection), and a
//! `MoEExperts` over them, so the GPU runs mistral.rs's own MoE kernels. When layer `l`'s MoE
//! runs, everything below is queued without blocking the host:
//!
//! 1. Layer `l + 1` is uploaded into the other buffer on the copy stream, after that buffer's
//!    last readers (layer `l - 1`) are done. It overlaps layer `l`'s MoE and layer `l + 1`'s
//!    attention.
//! 2. candle's stream waits for layer `l`'s upload.
//! 3. Layer `l`'s experts run on the GPU; the buffer's read fence is recorded after them.
//!
//! The first layer of each chunk has no predecessor to prefetch it, so its upload is not
//! hidden. Every chunk streams every layer's experts once, so this only pays for long chunks:
//! shorter ones (`--prefill-stream-min-tokens`) and layers whose stacks are not pinned stay on
//! the CPU.
//!
//! **The buffers are the stream's own memory.** They used to be lent from the expert cache for
//! the duration of a long chunk and handed back before decoding, which measured as an unstable
//! cycle: the lend returns whole banks (1107 MiB of a 1300 MiB need at a 1 GB cache), the buffers
//! then need the difference from free VRAM, and the give-back/re-borrow churn leaked ~400 MiB of
//! free memory per conversation until a 3 MiB miss turned the stream off for good. So the stream
//! now reserves its own VRAM once (checked against the device before the first chunk) and keeps
//! it: `--expert-cache-gb` is the cache's budget, and the stream adds its two layer buffers on
//! top. If they do not fit, the stream says so once and prefill stays on the CPU.

use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::ExpertLoadContext;
use mistralrs_core::MoEExperts;
use mistralrs_quant::Comm;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::copy_stream::{
    trim_device_pool, CopyDone, CopyStream, CopyStreams, ReadFence, SlotMemory,
};

/// Default for `--prefill-stream-min-tokens`: chunks shorter than this stay on the CPU (Strata
/// streams every expert from 1024 tokens).
pub const DEFAULT_MIN_TOKENS: usize = 1024;

/// The operator's ceiling on the prompt step (`min(--max-num-batched-tokens,
/// --max-prefill-chunk-tokens)`), published by the commands before the model loads. The plan below
/// only ever chooses at or below it.
static PROMPT_STEP_CEILING: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `0` clears the ceiling (tests, non-CUDA).
pub fn set_prompt_step_ceiling(tokens: usize) {
    PROMPT_STEP_CEILING.store(tokens, std::sync::atomic::Ordering::Release);
}

fn prompt_step_ceiling() -> usize {
    match PROMPT_STEP_CEILING.load(std::sync::atomic::Ordering::Acquire) {
        0 => DEFAULT_MIN_TOKENS * 8,
        n => n,
    }
}

/// Free VRAM kept on top of what the plan below prices: the KV blocks and the copy-stream staging a
/// prompt step grows into, plus the driver's own bookkeeping.
const HEADROOM: usize = 192 << 20;

/// The prefill path's memory decision: how many layer buffers, and the prompt step they were
/// priced with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Plan {
    pub stride: usize,
    pub chunk: usize,
}

/// Rows one MoE call sees. A prompt step runs its layers in row batches of this size, so the
/// activation footprint is bounded by the batch and not by `--max-prefill-chunk-tokens` — the
/// "ubatch" effect: a bigger chunk then costs PCIe uploads only, never activation memory.
pub(crate) const SUB_ROWS: usize = 256;

/// Device bytes a prompt step of `chunk` tokens needs **on top of** the stream's own buffers.
///
/// Two terms. The MoE's is Strata's measured form (`160 MiB + 680 MiB per 1024 tokens`,
/// `src/program/generate.cpp:3216`, and our numbers agree — a 1024-token call blew a 512 MiB margin
/// and fits in 768 MiB) applied to the *batch* one call sees, which the row batching caps at
/// `SUB_ROWS`, so it does not grow with the chunk. Everything else in a prompt step — attention's
/// QKV/output, the gathered PLE rows, the copy-stream staging — does grow with the chunk, and this
/// second term is its measured envelope: 24 KiB a token is ~3 × hidden × 2 bytes at this model.
pub(crate) fn activation_bytes(chunk: usize) -> usize {
    let moe = (160 << 20) + (680 << 20) * chunk.min(SUB_ROWS) / 1024;
    let rest = chunk * (24 << 10);
    moe + rest
}

/// The whole prefill memory decision, in one pure function: the largest prompt step that fits, in
/// tokens, and how many layer buffers go with it.
///
/// **A bigger chunk beats the second buffer.** Every chunk streams every layer once (~31 GB of PCIe
/// at IQ1_M), so doubling the chunk halves the traffic per token, while the second buffer only hides
/// that upload behind the compute. Measured on this 12 GB card: one buffer at 1024 tokens prefill
/// 915 / 1815 / 3615 tokens at 201 / 356 / 416 tok/s; two buffers at 512 at 147 / 285. So the search
/// walks the 256-token grid from `ceiling` down and takes the second buffer only when the *same*
/// chunk affords it.
pub(crate) fn plan(free: usize, layer_bytes: usize, ceiling: usize) -> Option<(usize, usize)> {
    const GRID: usize = 256;
    // The activation term is an estimate, so it carries slack: on this card the formula alone would
    // take a 1280-token chunk with 3 MiB to spare (the exact shape of the miss that cost a whole
    // run's prefill). An eighth is enough to land on the 1024 the measurements proved.
    // The slack follows the free memory as well: the estimate's error is a share of what is
    // allocated, and on a big card a fixed floor would be noise. The floor is 512 MiB because a
    // prompt step allocates more than the two terms above — the paged-attention scratch and the
    // step's own buffers grow with `--max-num-batched-tokens`, and 192–227 MiB of slack OOM'd a
    // 1723-token chunk twice with two buffers resident.
    // `NAIWA_PREFILL_SLACK_MIB` (probe): the free-VRAM slack the plan keeps beyond the buffers and the
    // activations. The 512 MiB floor is what M10 measured as safe, and it is also the only thing standing
    // between one buffer and two at this card's free VRAM, so it is worth calibrating instead of trusting.
    let slack_override = std::env::var("NAIWA_PREFILL_SLACK_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|mib| mib << 20);
    let fits = |chunk: usize, stride: usize| {
        let act = activation_bytes(chunk);
        let head = slack_override.unwrap_or_else(|| (512 << 20).max(free / 8));
        free >= stride * layer_bytes + act + act / 8 + head
    };
    // Largest chunk with one buffer: `fits` is monotone in the chunk, so bisect.
    let (mut lo, mut hi, mut best) = (1usize, (ceiling.min(8192) / GRID).max(1), 0usize);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid * GRID, 1) {
            best = mid * GRID;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if best == 0 {
        return None;
    }
    let stride = if fits(best, 2) { 2 } else { 1 };
    Some((stride, best))
}

/// One layer's host stacks (gate, up, down). Holding them keeps the memory that in-flight
/// uploads read alive.
struct HostStacks([Arc<QTensor>; 3]);

struct LayerBuf {
    /// The layer uploaded (or being uploaded) into this buffer.
    loaded: Option<usize>,
    /// Transport probe: what the last upload moved and when it was queued, so the consumer's wait can report
    /// the copy's own wall time and rate rather than the whole layer's time.
    queued: Option<std::time::Instant>,
    bytes: usize,
    copies: usize,
    memory: Option<[SlotMemory; 3]>,
    experts: Option<MoEExperts>,
    /// Recorded on candle's stream after the kernels that read this buffer.
    released: ReadFence,
    /// Recorded on the copy stream after the upload into this buffer.
    ready: CopyDone,
}

pub struct PrefillStream {
    device: Device,
    copy: Arc<CopyStream>,
    num_experts: usize,
    top_k: usize,
    act: Activation,
    comm: Arc<Comm>,
    layers: RwLock<Vec<Option<HostStacks>>>,
    bufs: [Mutex<LayerBuf>; 2],
    /// The memory plan in force (buffers and prompt step), from warmup and re-checked at the first
    /// long chunk. `None` until someone decides; `stride == 0` means the stream is off.
    plan: Mutex<Option<Plan>>,
    /// The first long chunk has re-checked the plan against the free VRAM it actually sees.
    checked: std::sync::atomic::AtomicBool,
    /// Chunks shorter than this stay on the CPU.
    min_tokens: usize,
}

impl PrefillStream {
    /// `None` when `device` has no copy stream (not CUDA).
    pub fn new(
        device: &Device,
        copy: Option<Arc<CopyStream>>,
        num_experts: usize,
        top_k: usize,
        act: Activation,
        comm: Arc<Comm>,
        min_tokens: usize,
    ) -> candle_core::Result<Option<Self>> {
        let Some(copy) = copy else {
            return Ok(None);
        };
        let buf = || -> candle_core::Result<Mutex<LayerBuf>> {
            Ok(Mutex::new(LayerBuf {
                loaded: None,
                queued: None,
                bytes: 0,
                copies: 0,
                memory: None,
                experts: None,
                released: ReadFence::new(device)?,
                ready: CopyDone::new(&copy)?,
            }))
        };
        Ok(Some(Self {
            device: device.clone(),
            bufs: [buf()?, buf()?],
            copy,
            num_experts,
            top_k,
            act,
            comm,
            layers: RwLock::default(),
            plan: Mutex::new(None),
            checked: std::sync::atomic::AtomicBool::new(false),
            min_tokens,
        }))
    }

    /// Make `layer`'s host stacks available for streaming. Only call with stacks that are
    /// page-locked: an upload from pageable memory would block the forward thread.
    pub fn register(&self, layer: usize, experts: &MoEExperts) {
        let Some((gate, up, down)) = experts.expert_qtensors() else {
            return;
        };
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, || None);
        }
        layers[layer] = Some(HostStacks([gate, up, down]));
    }

    fn stacks(&self, layer: usize) -> Option<[Arc<QTensor>; 3]> {
        self.layers
            .read()
            .get(layer)
            .and_then(|s| s.as_ref())
            .map(|s| s.0.clone())
    }

    /// Bytes of the largest registered layer: one buffer's worth.
    fn layer_bytes(&self) -> usize {
        self.layers
            .read()
            .iter()
            .flatten()
            .map(|s| s.0.iter().map(|t| t.storage_size_in_bytes()).sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Measure now and return what fits, or `None` if nothing does.
    fn decide(&self) -> Option<(Plan, usize)> {
        let free = crate::copy_stream::free_bytes(&self.device).unwrap_or(0);
        let (stride, chunk) = plan(free, self.layer_bytes(), prompt_step_ceiling())?;
        Some((Plan { stride, chunk }, free))
    }

    /// The warmup decision: run the plan once the cache, the KV and the mapped buffers exist, keep
    /// it, and log the arithmetic. Returns the prompt step it chose.
    pub(super) fn plan(&self) -> Option<usize> {
        let (decision, free) = self.decide()?;
        *self.plan.lock().unwrap() = Some(decision);
        log_plan(self, decision, free, "warmup");
        Some(decision.chunk)
    }

    /// The plan in force, or `(stride 0, chunk 0)` if it did not fit.
    fn current(&self) -> Plan {
        (*self.plan.lock().unwrap()).unwrap_or(Plan { stride: 0, chunk: 0 })
    }

    /// How many buffers to use, decided once from free VRAM: `2` (overlap the next layer's upload
    /// with this layer's MoE), `1` (no overlap, half the memory), or `0` (the stream stays off).
    ///
    /// The warmup plan is an estimate of what the first chunk will see; between it and that chunk
    /// the staging ring and the KV grow (~200 MiB). So this re-measures once and takes the **worse**
    /// of the two decisions, publishing a smaller prompt step if that is what it takes — the
    /// scheduler reads the ceiling every step, so a mid-session shrink is a smaller next chunk, not
    /// a broken one.
    ///
    /// The number of buffers is the only memory decision here, and it is measured rather than
    /// assumed: two buffers need twice a layer, and on a 12 GB card that is what pushes a
    /// 1024-token chunk's MoE intermediates over the edge. One buffer still beats the CPU by an
    /// order of magnitude (the uploads are then serial with the compute instead of hidden).
    fn reserve(&self) -> candle_core::Result<usize> {
        let first = !self.checked.swap(true, std::sync::atomic::Ordering::AcqRel);
        if !first {
            return Ok(self.current().stride);
        }
        let previous = self.current();
        match self.decide() {
            Some((now, free)) => {
                // The warmup estimate versus what the first chunk sees: the worse of the two.
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
                    // The scheduler reads this every step: a smaller next chunk, not a broken one.
                    mistralrs_core::PROMPT_STEP_CEILING
                        .store(decision.chunk, std::sync::atomic::Ordering::Release);
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
                warn!(
                    "prefill stream: off (one {:.0} MiB layer buffer + {:.0} MiB activations + \
                     {:.0} MiB headroom does not fit in {:.0} MiB free). Lower --expert-cache-gb to \
                     leave room for the stream, or drop --prefill-stream",
                    mib(self.layer_bytes()),
                    mib(activation_bytes(previous.chunk.max(256))),
                    mib(HEADROOM),
                    mib(free),
                );
                *self.plan.lock().unwrap() = Some(Plan { stride: 0, chunk: 0 });
                Ok(0)
            }
        }
    }

    /// Give up a buffer: free everything the stream holds, hand the blocks to the driver, and keep
    /// going with one buffer fewer (or off). Called when an allocation the stream made was refused
    /// with a CUDA OOM — the driver's answer is the only reliable statement about what fits, so it
    /// is used as the signal instead of a size heuristic.
    fn downgrade(&self, stride: usize) -> candle_core::Result<()> {
        let next = stride.saturating_sub(1);
        let chunk = self.current().chunk;
        *self.plan.lock().unwrap() = Some(Plan { stride: next, chunk });
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

    /// Queue the upload of `layer` into its buffer, unless it is already there. `false`: the
    /// layer is not registered.
    ///
    /// A mixed-quant layer whose shape differs from the buffer's frees the old buffer and
    /// allocates the new one; that is where the prefill stream's memory peak comes from.
    fn load(&self, layer: usize, stride: usize) -> candle_core::Result<bool> {
        let Some(stacks) = self.stacks(layer) else {
            return Ok(false);
        };
        let mut buf = self.bufs[layer % stride.max(1)].lock().unwrap();
        if buf.loaded == Some(layer) {
            return Ok(true);
        }
        buf.loaded = None;
        // (Re)allocate when the layer's stacks differ in shape or dtype (mixed-quant GGUFs).
        let fits = buf.memory.as_ref().is_some_and(|m| {
            m.iter()
                .zip(&stacks)
                .all(|(m, s)| m.tensor().dtype() == s.dtype() && m.tensor().shape() == s.shape())
        });
        if !fits {
            if buf.memory.is_some() {
                // The old memory is freed in candle's stream order; an upload into it may still
                // be in flight on the copy stream, so order the free after it.
                buf.ready.wait_on(&self.device)?;
            }
            buf.experts = None;
            buf.memory = None;
            // The old block went back to CUDA's pool, which keeps it for reuse at that exact size.
            // This model's expert shapes differ per layer group (mixed-quant GGUF), so letting the
            // pool keep one block per shape defeats the two-buffer design: return the freed block
            // to the driver before allocating the next shape's. The switch is rare (per shape
            // change, not per layer) and the synchronize only waits for work already queued.
            trim_device_pool(&self.device)?;
            trace_buffer_bytes(&self.device, layer, &stacks);
            let memory = [
                self.copy.alloc_like(&stacks[0])?,
                self.copy.alloc_like(&stacks[1])?,
                self.copy.alloc_like(&stacks[2])?,
            ];
            // The allocations are queued on candle's stream: the copy waits for them.
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
        // SAFETY: registered stacks are page-locked, and `self.layers` keeps them alive and
        // unmodified for as long as this stream exists (`Drop` waits for the copies).
        let queued = std::time::Instant::now();
        if std::env::var_os("NAIWA_STREAM_SYNC").is_some() {
            // Serial probe: the blocking upload waits for the copies, so the line above reports the copy's own
            // wall time and rate instead of the queue-side latency (`wait_on` only *enqueues* a stream wait).
            // It also removes every overlap, so this mode measures throughput and nothing else.
            self.copy.upload(&buf.released, &parts)?;
        } else {
            unsafe { self.copy.upload_async(&buf.released, &parts, &buf.ready)? };
        }
        buf.queued = Some(queued);
        buf.loaded = Some(layer);
        Ok(true)
    }

    /// Run `layer`'s routed experts on the GPU. `None`: this chunk or layer stays on the CPU.
    pub fn forward(
        &self,
        layer: usize,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (batch, seq, _) = xs.dims3()?;
        if seq == 1 {
            return Ok(None);
        }
        if batch * seq < self.min_tokens || self.stacks(layer).is_none() {
            return Ok(None);
        }
        let mut stride = self.reserve()?;
        if stride == 0 {
            return Ok(None);
        }
        let loaded = match self.load(layer, stride) {
            Ok(loaded) => loaded,
            Err(e) if is_out_of_memory(&e) => {
                self.downgrade(stride)?;
                stride = stride.saturating_sub(1);
                if stride == 0 {
                    return Ok(None);
                }
                self.load(layer, stride)?
            }
            Err(e) => return Err(e),
        };
        if !loaded {
            return Ok(None);
        }
        // Stream the next layer in while this one computes — only with a second buffer.
        if stride == 2 {
            if let Err(e) = self.load(layer + 1, stride) {
                warn!(
                    "prefill stream: could not prefetch layer {}: {e}",
                    layer + 1
                );
            }
        }
        let buf = self.bufs[layer % stride].lock().unwrap();
        let experts = buf.experts.as_ref().expect("loaded");
        let t_wait = std::time::Instant::now();
        buf.ready.wait_on(&self.device)?;
        let waited = t_wait.elapsed();
        let t_compute = std::time::Instant::now();
        let out = experts.forward(xs, topk_weights.clone(), topk_ids);
        let transport = buf.queued.map(|queued| {
            let copy = queued.elapsed();
            let gbs = buf.bytes as f64 / copy.as_secs_f64().max(1e-9) / 1e9;
            format!(
                " copy={copy:?} {gbs:.1} GB/s {} MiB x {}",
                buf.bytes >> 20,
                buf.copies
            )
        });
        crate::dbg_note(
            "prefill_stream",
            &format!(
                "layer {layer} stride {stride} dma_wait={waited:?} compute={:?}{}",
                t_compute.elapsed(),
                transport.unwrap_or_default()
            ),
        );
        // Fence even on error: kernels reading the buffer may already be queued.
        buf.released.record_reads(&self.device)?;
        out.map(Some)
    }
}

/// The prefill stream of each CUDA device, created by the first layer that loads on it.
pub struct PrefillStreams {
    streams: Mutex<Vec<(Device, Arc<PrefillStream>)>>,
    min_tokens: usize,
}

impl PrefillStreams {
    /// Streams that run prefill chunks of `min_tokens` or more on the GPU.
    pub fn new(min_tokens: usize) -> Self {
        Self {
            streams: Mutex::default(),
            min_tokens,
        }
    }

    /// Run the memory plan for every device's stream and publish the chunk it chose to the
    /// scheduler (the smallest across devices). `None`: no stream exists, or no chunk fits.
    pub fn plan(&self) -> Option<usize> {
        let streams = self.streams.lock().unwrap();
        let chunk = streams
            .iter()
            .filter_map(|(_, stream)| stream.plan())
            .min()?;
        drop(streams);
        mistralrs_core::PROMPT_STEP_CEILING
            .store(chunk, std::sync::atomic::Ordering::Release);
        info!(
            "prefill: the scheduler's prompt step is capped at {chunk} tokens (from \
             --max-num-batched-tokens/--max-prefill-chunk-tokens {})",
            prompt_step_ceiling()
        );
        Some(chunk)
    }

    /// Register `experts` (this block's CPU stacks) for streaming and return the device's
    /// stream. `None` for the MTP head, for stacks that are not all page-locked (`pinned`), and
    /// off CUDA: such a layer keeps its prefill on the CPU.
    pub fn attach(
        &self,
        ctx: &ExpertLoadContext<'_>,
        experts: &MoEExperts,
        pinned: bool,
        copy_streams: &CopyStreams,
    ) -> candle_core::Result<Option<Arc<PrefillStream>>> {
        if ctx.site.is_mtp || !pinned || experts.expert_qtensors().is_none() {
            return Ok(None);
        }
        let device = ctx.layer_device;
        let mut streams = self.streams.lock().unwrap();
        let stream = match streams.iter().find(|(d, _)| d.same_device(device)) {
            Some((_, ps)) => Arc::clone(ps),
            None => {
                let Some(ps) = PrefillStream::new(
                    device,
                    copy_streams.get(device),
                    ctx.config.num_experts,
                    ctx.config.num_experts_per_tok,
                    ctx.act,
                    Arc::clone(ctx.comm),
                    self.min_tokens,
                )?
                else {
                    return Ok(None);
                };
                let ps = Arc::new(ps);
                streams.push((device.clone(), Arc::clone(&ps)));
                ps
            }
        };
        stream.register(ctx.site.layer, experts);
        Ok(Some(stream))
    }
}

impl Drop for PrefillStream {
    fn drop(&mut self) {
        // In-flight uploads read the host stacks this struct keeps alive.
        let _ = self.copy.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor};
    use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
    use mistralrs_core::layers::Activation;
    use mistralrs_core::MoEExperts;
    use mistralrs_quant::{Comm, Id};

    fn stack(e: usize, n: usize, k: usize, seed: f32) -> Arc<QTensor> {
        let data: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect();
        let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap())
    }

    use super::{activation_bytes, plan, SUB_ROWS};

    /// The plan is the only memory judgement the prefill path makes, so pin it: the chunk is as
    /// large as the free bytes allow, and the second buffer is taken only at the same chunk.
    #[test]
    fn plan_spends_memory_on_the_chunk_before_the_second_buffer() {
        let layer = 650 << 20;
        let gb = |g: usize| g << 30;
        let act = |c: usize| activation_bytes(c);
        let head = |free: usize| (512 << 20).max(free / 8);
        let fits = |free: usize, chunk: usize, stride: usize| {
            let a = act(chunk);
            free >= stride * layer + a + a / 8 + head(free)
        };

        // The MoE part stops at one row batch; the rest of the step still grows with the chunk.
        assert_eq!(
            act(8192) - act(SUB_ROWS),
            (8192 - SUB_ROWS) * (24 << 10),
            "only the attention term may scale"
        );

        // A big card: ceiling, both buffers.
        assert_eq!(plan(gb(12), layer, 8192).unwrap(), (2, 8192));

        // This 12 GB card at a 1 GB cache: 2049 MiB of free VRAM is one buffer at the ceiling.
        assert_eq!(plan(2049 << 20, layer, 8192).unwrap(), (1, 8192));
        // 2400 MiB is the edge where the second buffer just fits (2399 needed).
        assert_eq!(plan(2400 << 20, layer, 8192).unwrap(), (2, 8192));

        // The operator's ceiling is a ceiling, not a target.
        assert_eq!(plan(gb(12), layer, 512).unwrap().1, 512);

        // The second buffer only when the same chunk affords it.
        let need2 = |free_guess: usize| {
            // Solve for the free memory where two buffers at 8192 just fit.
            let mut free = free_guess;
            while !fits(free, 8192, 2) {
                free += 8 << 20;
            }
            free
        };
        // With the 512 MiB floor, a 12 GB card's ~2.3 GiB buys one buffer at the ceiling and
        // 8 GiB buys two; the floor is what keeps a prompt step's own scratch out of the way.
        assert_eq!(plan(gb(8), layer, 8192).unwrap(), (2, 8192));
        let just = need2(gb(1));
        assert_eq!(plan(just, layer, 8192).unwrap(), (2, 8192));
        // Well below it, one buffer at the same ceiling: the buffer count is what gives first.
        assert_eq!(plan(just * 8 / 10, layer, 8192).unwrap().0, 1);
        assert_eq!(plan(just * 8 / 10, layer, 8192).unwrap().1, 8192);

        // Nothing fits: the stream stays off rather than OOM-ing mid-chunk.
        assert_eq!(plan(layer, layer, 1024), None);
    }

    /// The fork's `MoEExperts::from_qtensors` computes the routed MoE of the stacks it is
    /// given: compare against a plain dequantized reference, with the full top-k and with a
    /// subset of it (the miss path's shape).
    #[test]
    fn from_qtensors_matches_a_dequantized_reference() {
        let (e, hidden, inter) = (4, 64, 32);
        let (gate, up, down) = (
            stack(e, inter, hidden, 0.1),
            stack(e, inter, hidden, 0.7),
            stack(e, hidden, inter, 1.3),
        );
        let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1).unwrap());
        let act = Activation::Silu;
        let experts =
            MoEExperts::from_qtensors(gate.clone(), up.clone(), down.clone(), e, 2, &comm, act)
                .unwrap();

        let tokens = 3;
        let xs: Vec<f32> = (0..tokens * hidden)
            .map(|i| (i as f32 * 0.11).cos())
            .collect();
        let xs = Tensor::from_vec(xs, (1, tokens, hidden), &Device::Cpu).unwrap();
        // Expert `ex` of a stack as its own QTensor (its byte slice), run through `QMatMul`.
        let expert = |q: &Arc<QTensor>, ex: usize| -> QMatMul {
            let dims = q.shape().dims();
            let data = q.data().unwrap();
            let stride = data.len() / dims[0];
            let one = qtensor_from_ggml(
                q.dtype(),
                &data[ex * stride..(ex + 1) * stride],
                vec![dims[1], dims[2]],
                &Device::Cpu,
            )
            .unwrap();
            QMatMul::QTensor(Arc::new(one))
        };
        let reference = |ids: &[[u32; 2]], ws: &[[f32; 2]], k: usize| -> Tensor {
            let mut rows = Vec::new();
            for t in 0..tokens {
                let x = xs.i((0, t)).unwrap().unsqueeze(0).unwrap(); // [1, hidden]
                let mut acc = Tensor::zeros((1, hidden), DType::F32, &Device::Cpu).unwrap();
                for j in 0..k {
                    let ex = ids[t][j] as usize;
                    let gx = expert(&gate, ex).forward(&x).unwrap();
                    let ux = expert(&up, ex).forward(&x).unwrap();
                    let h = (act.forward(&gx).unwrap() * ux).unwrap();
                    let y = expert(&down, ex).forward(&h).unwrap();
                    acc = (acc + (y * ws[t][j] as f64).unwrap()).unwrap();
                }
                rows.push(acc);
            }
            Tensor::cat(&rows, 0).unwrap().unsqueeze(0).unwrap()
        };
        // The experts' fused kernel and `QMatMul` quantize activations alike on aarch64 but not
        // on x86 (where the fused kernel stays in f32), so they agree only to Q8_0 rounding
        // (~0.008 here). A swapped routing weight or wrong expert moves the output by >0.3.
        const TOL: f32 = 2e-2;
        let ids = [[0u32, 2], [3, 1], [2, 0]];
        let ws = [[0.6f32, 0.4], [0.3, 0.7], [0.5, 0.5]];
        let max_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_keepdim(D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        // full top-k
        let ids_t = Tensor::new(&ids, &Device::Cpu).unwrap();
        let ws_t = Tensor::new(&ws, &Device::Cpu).unwrap();
        let got = experts.forward(&xs, ws_t, &ids_t).unwrap();
        let want = reference(&ids, &ws, 2);
        assert_eq!(got.dims(), want.dims());
        assert!(max_diff(&got, &want) < TOL, "full top-k diverges");

        // a subset of the routed experts (one per token)
        let sub_ids: Vec<u32> = ids.iter().map(|r| r[0]).collect();
        let sub_ws: Vec<f32> = ws.iter().map(|r| r[0]).collect();
        let got = experts
            .forward(
                &xs,
                Tensor::from_vec(sub_ws, (tokens, 1), &Device::Cpu).unwrap(),
                &Tensor::from_vec(sub_ids, (tokens, 1), &Device::Cpu).unwrap(),
            )
            .unwrap();
        let want = reference(&ids, &ws, 1);
        assert!(max_diff(&got, &want) < TOL, "subset of top-k diverges");
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

/// One line with the whole decision, so the numbers are checkable in the log.
fn log_plan(stream: &PrefillStream, plan: Plan, free: usize, when: &str) {
    let mib = |b: usize| (b as f64) / (1u64 << 20) as f64;
    let layer = stream.layer_bytes();
    let act = activation_bytes(plan.chunk);
    info!(
        "prefill ({when}): chunk {}, {} × {:.0} MiB layer buffer(s) + {:.0} MiB activations + \
         {:.0} MiB headroom = {:.0} MiB of {:.0} MiB free",
        plan.chunk,
        plan.stride,
        mib(layer),
        mib(act),
        mib(HEADROOM),
        mib(plan.stride * layer + act + HEADROOM),
        mib(free),
    );
}

/// A CUDA out-of-memory error, however deep in the wrapper it is.
fn is_out_of_memory(error: &candle_core::Error) -> bool {
    let text = error.to_string();
    text.contains("OUT_OF_MEMORY") || text.contains("out of memory")
}

/// One layer's MoE over a prompt step, in row batches of `SUB_ROWS`.
///
/// The routing is row-independent, so slicing the rows changes nothing but the GEMM shapes; it is
/// what keeps a 8192-token chunk's activations inside the same memory a 256-token one needs. Only
/// one sequence is sub-batched: with several sequences in the step the rows of one batch would have
/// to be interleaved, and a prompt step in this server is one sequence.
fn forward_batched(
    experts: &MoEExperts,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    batch: usize,
    seq: usize,
) -> candle_core::Result<Tensor> {
    if batch != 1 || seq <= SUB_ROWS {
        return experts.forward(xs, topk_weights.clone(), topk_ids);
    }
    let mut parts = Vec::with_capacity(seq.div_ceil(SUB_ROWS));
    let mut start = 0;
    while start < seq {
        let len = SUB_ROWS.min(seq - start);
        let x = xs.narrow(1, start, len)?;
        let w = topk_weights.narrow(0, start, len)?;
        let i = topk_ids.narrow(0, start, len)?;
        parts.push(experts.forward(&x, w, &i)?);
        start += len;
    }
    Tensor::cat(&parts, 1)
}
