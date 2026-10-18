//! GPU expert cache for `--cpu-moe --expert-cache-gb N`.
//!
//! Each decoder layer gets its own per-layer slot range from a total VRAM budget. Slots hold the
//! expert's *quantized* bytes (GGML block format) — never a dequantized copy — so both the GPU
//! and CPU paths run with the same quantization and the cache costs the same VRAM as the CPU tier
//! costs in RAM.
//!
//! # Format support
//!
//! - **GGUF** (Q4_K, Q6_K, …): `get_qtensor()` returns the stacked `[E, N, K]` `QTensor`; bytes
//!   are sliced per expert and re-assembled as a `[N, K]` `QTensor` on the slot device.
//! - **ISQ-quantized** (`--isq`): same path, because ISQ leaves a `QTensor` behind.
//! - **BF16 safetensors**: `get_qtensor()` returns `None`; the cache falls back to a dense
//!   `Tensor` copy instead. Dense slots cost ~4× more VRAM per expert than Q4K.
//!
//! # Prefill
//!
//! Every routed expert runs on the CPU (all of them are needed anyway). Each layer first starts
//! a background fill of the *next* layer's empty slots. On CUDA a fill is a copy on a dedicated
//! copy stream into slot memory allocated once at load, so it overlaps this layer's CPU experts
//! and the next layer's attention (`copy_stream`). Elsewhere it uploads through candle's stream.
//!
//! # Decode (single token)
//!
//! - Top-k is split into slot hits and misses. Hits run via `QMatMul` on the slot device, misses
//!   on the CPU with only the miss ids, through the fused two-pass decode (`offload::fused_decode`)
//!   or, for experts it does not serve, `MoEExperts::forward`; both sides at once, router weights
//!   applied once per side. The misses run on the `deferred` worker over host copies; every CUDA
//!   call stays on the calling thread.
//! - Misses are never loaded on the token path. Instead, each layer submits its activation to a
//!   lookahead thread that runs the next layer's router on it and loads the predicted experts:
//!   into empty slots freely, over least-recently-used residents at most `MAX_LOOKAHEAD_SWAPS`
//!   times. Whatever is not resident when the next layer runs goes to the CPU.
//! - A token captures its hits' weights (`Arc`s) under the layer lock, so a slot replaced later
//!   never changes what an in-flight token computes with.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use candle_core::cuda::cudarc::driver::sys;
use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::copy_stream::{self, CopyStream, CopyStreams, ReadFence, SlotMemory};
use crate::deferred;
use crate::host_memory::{self, PinnedStack};
use crate::offload::{attach_prefill, fused_decode as offload_fused_decode, CpuExperts};
use crate::prefill_stream::{ExpertMemoryLender, PrefillStream, PrefillStreams};

// Number of prefill tokens above which we skip the cache entirely.
// Calls with more tokens than this are prefill chunks: every expert runs on the CPU (or the
// prefill stream). Up to it they take the decode split, which is what a speculative verify step
// needs (the target runs the n drafts plus one token in one forward).
const PREFILL_THRESHOLD: usize = 16;

/// Upper bound on resident experts the decode lookahead replaces per layer per token (per row of
/// a verify step). Each swap
/// copies one expert's three projections host-to-device; without a cap a poor prediction would
/// turn every token into a full re-upload of every layer's top-k.
const MAX_LOOKAHEAD_SWAPS: usize = 2;

/// Replacement cap for one prediction pass, per row. `NAIWA_LH_SWAPS` overrides it for sweeps: more
/// swaps means more predicted experts displace residents (more DMA) in exchange for fewer CPU experts
/// on the critical path. The default is the shipped value; the env is a measurement knob only.
fn max_lookahead_swaps() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NAIWA_LH_SWAPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MAX_LOOKAHEAD_SWAPS)
    })
}

/// With `--prefill-stream`, each layer keeps its last slots in a separate bank of about its
/// share of the loan: two layers of experts, spread over the estimated 48 layers, plus a
/// quarter for layers with smaller slots than the largest one. The loan lends whole banks.
fn loan_bank_slots(num_experts: usize) -> usize {
    (2 * num_experts * 5).div_ceil(4 * 48)
}

/// With `--pcie-frac`, a decode step DMAs its highest-weight misses into a staging ring shared
/// by every layer of the model (Strata's `kStagingBlobs` shape), instead of a per-layer scratch
/// bank: per-layer banks could only hold `ceil(frac*top_k)` slots each, which at `frac` 0.55
/// meant 6 slots x 48 layers of VRAM and OOM'd, while cross-layer reuse needs one fixed pool of
/// slots that every layer's step borrows in turn. The fence model already serializes reusers —
/// a reader holds the `SlotBuf` `Arc` until it has recorded its fence, and the next layer's DMA
/// waits on that fence — so no cross-layer accounting is needed.
const SCRATCH_RING_FIRST: usize = 1 << 30;

/// Ring capacity: enough that consecutive layers' steps rarely collide on one slot, small
/// enough to be a rounding error next to the cache. Strata uses 16 (`kStagingBlobs`).
/// `NAIWA_SCRATCH_SLOTS` overrides.
/// Staging-ring slots per quant shape. Measured: 16 and 64 perform identically (the ring has no id -> slot
/// lookup, so a staged expert is copied again either way), so this is a constant, not a knob.
const fn scratch_ring_slots() -> usize {
    16
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

/// Decode-path counters over every cached layer of the process (one model per process).
struct Counters {
    installed: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    /// Experts served from the staging ring (PCIe), counted apart from cache hits.
    staged: AtomicU64,
    lookahead_loads: AtomicU64,
    swaps: AtomicU64,
    pcie_misses: AtomicU64,
    pcie_bytes: AtomicU64,
    /// v2 M2: decode-step MoE segments served by replaying a captured graph (1 `cuGraphLaunch`
    /// each) rather than ~5 raw launches.
    /// v2 M4: layer DMA gates captured at warmup (one per layer with staging in service).
    /// v2 M4: layer DMA gates replayed (one per layer per pass, when PCIe staging is on).
    gate_replays: AtomicU64,
    /// v2 M4: DMA-gate waits launched warm because the gate graph could not be captured. Must stay
    /// 0 — a non-zero value means the graph route failed and only the fallback ran.
    staging_eager_waits: AtomicU64,
}

#[cfg(feature = "cuda")]
static SCHEDULE: OnceLock<Arc<crate::doorbell::ScheduleBuf>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn schedule() -> Option<&'static Arc<crate::doorbell::ScheduleBuf>> {
    SCHEDULE.get()
}

#[cfg(feature = "cuda")]
fn zero_buf(max_len: usize) -> &'static [f32] {
    static ZEROS: OnceLock<Vec<f32>> = OnceLock::new();
    ZEROS.get_or_init(|| vec![0f32; max_len])
}

static COUNTERS: Counters = Counters {
    installed: AtomicBool::new(false),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    staged: AtomicU64::new(0),
    lookahead_loads: AtomicU64::new(0),
    swaps: AtomicU64::new(0),
    pcie_misses: AtomicU64::new(0),
    pcie_bytes: AtomicU64::new(0),
    gate_replays: AtomicU64::new(0),
    staging_eager_waits: AtomicU64::new(0),
};

/// Expert-cache counters since the last call to `take_stats`.
pub struct CacheStats {
    /// Routed experts of decode steps served from a slot.
    pub hits: u64,
    /// Routed experts of decode steps run on the CPU.
    pub misses: u64,
    /// Experts served from the staging ring over PCIe: neither a cache hit nor a CPU miss.
    pub staged: u64,
    /// Experts the lookahead loaded into empty slots.
    pub lookahead_loads: u64,
    /// Experts the lookahead loaded over a least-recently-used resident.
    pub swaps: u64,
    /// Lookahead requests dropped because the worker was busy.
    pub lookahead_dropped: u64,
    /// v2 M4: layer DMA gates captured at warmup.
    /// v2 M4: layer DMA gates replayed.
    pub gate_replays: u64,
    /// v2 M4: DMA-gate waits launched warm (gate graph unavailable); 0 in the normal path.
    pub staging_eager_waits: u64,
    /// Experts DMA'd into a scratch slot and run on the GPU instead of the CPU (`--pcie-frac`).
    pub pcie_misses: u64,
    /// Bytes those DMAs moved host-to-device.
    pub pcie_bytes: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }
}

/// Read and reset the counters; `None` when no expert cache is installed.
pub fn take_stats() -> Option<CacheStats> {
    let c = &COUNTERS;
    if !c.installed.load(Ordering::Relaxed) {
        return None;
    }
    let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
    Some(CacheStats {
        hits: take(&c.hits),
        misses: take(&c.misses),
        staged: take(&c.staged),
        lookahead_loads: take(&c.lookahead_loads),
        swaps: take(&c.swaps),
        lookahead_dropped: crate::lookahead::dropped(),
        pcie_misses: take(&c.pcie_misses),
        pcie_bytes: take(&c.pcie_bytes),
        gate_replays: take(&c.gate_replays),
        staging_eager_waits: take(&c.staging_eager_waits),
    })
}

// ---------------------------------------------------------------------------
// Provider state
// ---------------------------------------------------------------------------

struct ProviderInner {
    /// Total budget remaining in bytes.
    remaining_bytes: u64,
    /// Per-layer slot quota (bytes), determined on first build() call.
    bytes_per_layer: Option<u64>,
}

/// GPU expert cache provider.
///
/// Installed once before the model loads; `build()` is called once per MoE block during
/// weight loading. Decoder layers fill their quota from the total budget; the MTP head is
/// passed through to the CPU-only path.
pub struct GpuCacheMoeProvider {
    cache_gb: f64,
    inner: Mutex<ProviderInner>,
    registry: Arc<LayerRegistry>,
    copy_streams: CopyStreams,
    /// `--prefill-stream`: run long prefill chunks' experts on the GPU.
    prefill: Option<PrefillStreams>,
    /// `--pcie-frac`: share of decode misses to DMA into scratch slots and run on the GPU.
    pcie_frac: f32,
    /// `--lookahead-swaps`: how many predicted experts may displace residents per row and pass.
    lookahead_swaps: usize,
}

impl GpuCacheMoeProvider {
    pub fn new(
        cache_gb: f64,
        prefill: Option<PrefillStreams>,
        pcie_frac: f32,
        lookahead_swaps: usize,
    ) -> Self {
        COUNTERS.installed.store(true, Ordering::Relaxed);
        let total_bytes = (cache_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        let registry: Arc<LayerRegistry> = Arc::default();
        // The prefill stream's layer buffers are carved out of this cache while a prompt runs.
        let prefill = prefill.map(|p| {
            p.with_lender(Arc::new(CacheLoan {
                registry: Arc::clone(&registry),
            }))
        });
        Self {
            cache_gb,
            inner: Mutex::new(ProviderInner {
                remaining_bytes: total_bytes,
                bytes_per_layer: None,
            }),
            registry,
            copy_streams: CopyStreams::default(),
            prefill,
            pcie_frac,
            lookahead_swaps,
        }
    }

    /// v2 M1: materialize every quant shape's staging ring now, i.e. after model load but
    /// before the first decode, so all ring addresses are fixed for the process lifetime.
    ///
    /// W105 bug (a) forced the ring to be created lazily at the first decode: a contiguous
    /// bank allocated during load OOM'd 2-in-3 runs against load-time fragmentation. Load is
    /// over by the time this runs, so the allocator has large free blocks again — but the
    /// first *decode* is too late for a capture, which needs the address up front. Warmup
    /// closes that window. No-op for layers without a copy stream. Returns the ring count.
    /// v2 M1/M2: build everything that must have a fixed address before the first decode — every
    /// resident bank's fixed-grid capture path (always), and the staging rings (only with
    /// `--pcie-frac`, which is where the ~215 MiB belongs). Returns `(captures, rings)`.
    pub fn warmup(&self) -> candle_core::Result<(usize, usize)> {
        let layers = self.registry.all();
        #[cfg(feature = "cuda")]
        if let Some(first) = layers.first() {
            let device = first.slot_device.clone();
            let max_n = CAPTURE_MAX_BATCH * first.hidden;
            // One slot past the highest decoder layer: the MTP head is a site of its own
            // (`site.layer == layers.len()`), it is not a `CachedLayer` (its experts are host-resident),
            // and yet it publishes and waits on the *same* schedule. Sizing the schedule by the decoder
            // count alone put its `SYNC_CPU_EXPERTS`/payload slot past the end of the mapping, which
            // surfaced as CUDA_ERROR_ILLEGAL_ADDRESS on the first draft step.
            let slots = layers
                .iter()
                .map(|l| l.layer)
                .max()
                .map(|max| max + 2)
                .unwrap_or(layers.len());
            match crate::doorbell::ScheduleBuf::acquire(&device, WAITS_PER_LAYER, slots, max_n) {
                Ok(Some(sched)) => {
                    let _ = SCHEDULE.set(Arc::new(sched));
                }
                Ok(None) => warn!("expert cache: no schedule buffer; CPU-return node unbuilt"),
                Err(e) => warn!("expert cache: schedule allocation failed: {e}"),
            }
        }
        #[allow(unused_mut)]
        let mut captures = 0usize;
        // M9: no segment captures. Measured (`NAIWA_NO_CAPTURE=1` before the deletion): graphs on 43.5 tok/s,
        // graphs off 44.1 - identical, so the eager segment is the one fixed configuration and the capture
        // machinery below is dead code kept only until the deletion commit.
        #[cfg(feature = "cuda")]
        let _ = &layers;
        let mut rings = 0usize;
        if self.pcie_frac > 0.0 {
            for layer in &layers {
                layer.warmup_scratch_ring()?;
            }
            rings = self.registry.ring_creations.load(Ordering::Relaxed);
        }
        Ok((captures, rings))
    }

    /// v2 M1 criterion ①: rings created so far. Must not change once warmup has run and the
    /// decode has started.
    pub fn ring_creations(&self) -> usize {
        self.registry.ring_creations.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Byte-level expert extraction (never dequantize)
// ---------------------------------------------------------------------------

/// The byte range of expert `e` in a stacked `[E, N, K]` `QTensor`.
///
/// Row-block formats never cross expert boundaries, so `[e·stride, (e+1)·stride)` of the
/// stack's buffer is exactly expert `e`.
fn expert_range(qt: &QTensor, e: usize) -> candle_core::Result<std::ops::Range<usize>> {
    let dims = qt.shape().dims();
    if dims.len() < 2 {
        candle_core::bail!("expert QTensor must be at least 2-D, got {:?}", dims);
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        candle_core::bail!(
            "expert K={k} is not divisible by block_size={block_size} for dtype {:?}",
            qt.dtype()
        );
    }
    let stride = n * (k / block_size) * qt.dtype().type_size();
    let range = e * stride..(e + 1) * stride;
    if range.end > qt.storage_size_in_bytes() {
        candle_core::bail!(
            "expert {e} byte range {range:?} exceeds buffer length {}",
            qt.storage_size_in_bytes()
        );
    }
    Ok(range)
}

/// Build a single-expert `[N, K]` `QTensor` on `device` from a stacked `[E, N, K]` `QTensor`.
///
/// Copies straight out of the stack, with no intermediate host buffer. When the stack is pinned
/// (`host_memory`), that copy is an asynchronous DMA that may still be reading after this
/// returns; it is safe because the stack outlives every copy (the layer owns it).
fn expert_qtensor_on_device(
    stack: &QTensor,
    e: usize,
    device: &Device,
) -> candle_core::Result<Arc<QTensor>> {
    let dims = stack.shape().dims();
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let range = expert_range(stack, e)?;
    let data = stack.data()?;
    let qt = qtensor_from_ggml(stack.dtype(), &data[range], vec![n, k], device)?;
    Ok(Arc::new(qt))
}

/// Slot-bytes estimate for a single expert from a `QTensor` stack.
/// Returns `None` when the stack is not accessible (dense/float).
fn slot_bytes_from_qtensor(qt: &QTensor) -> Option<u64> {
    let dims = qt.shape().dims();
    if dims.len() < 2 {
        return None;
    }
    let (n, k) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let block_size = qt.dtype().block_size();
    if k % block_size != 0 {
        return None;
    }
    let type_size = qt.dtype().type_size();
    // gate + up + down (gate/up are [inter, hidden]; down is [hidden, inter] — same bytes)
    Some((3 * n * (k / block_size) * type_size) as u64)
}

// ---------------------------------------------------------------------------
// RoutedExpertsProvider impl
// ---------------------------------------------------------------------------

use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts, RoutedExpertsProvider};

impl RoutedExpertsProvider for GpuCacheMoeProvider {
    fn build(
        &self,
        ctx: ExpertLoadContext<'_>,
    ) -> candle_core::Result<Option<Arc<dyn RoutedExperts>>> {
        // Never cache CPU-device layers.
        if ctx.layer_device.is_cpu() {
            return Ok(None);
        }
        // The MTP head's experts run on the CPU like an uncached layer's. Returning `None` would
        // load all of them onto the GPU (512 Q4_0 experts, ~1.35 GB), which the cache needs.
        if ctx.site.is_mtp {
            let experts = ctx.load_builtin(Device::Cpu)?;
            host_memory::advise_expert_stacks(&experts)?;
            return Ok(Some(Arc::new(CpuExperts::new(
                experts,
                ctx.act,
                ctx.layer_device.clone(),
                ctx.site.layer,
                None,
                Vec::new(),
            ))));
        }

        let cpu_experts = ctx.load_builtin(Device::Cpu)?;
        let huge_bytes = host_memory::advise_expert_stacks(&cpu_experts)?;
        let num_experts = ctx.config.num_experts;

        // Try to get the quantized stacked weight for the gate projection — this tells us
        // the dtype and per-expert slot bytes, and whether we're in quantized or dense mode.
        let (slot_bytes, is_quantized) = {
            let qt_opt = cpu_experts.expert_qtensors();
            match qt_opt
                .as_ref()
                .and_then(|(gate, _, _)| slot_bytes_from_qtensor(gate))
            {
                Some(sb) => (sb, true),
                None => {
                    // Dense fallback: estimate from hidden/inter in BF16
                    let sb = 3u64
                        * ctx.config.hidden_size as u64
                        * ctx.config.moe_intermediate_size as u64
                        * 2; // BF16 = 2 bytes
                    (sb, false)
                }
            }
        };

        let mut inner = self.inner.lock().unwrap();

        // On first call, compute per-layer quota.  We assume all layers use the same weight
        // shape (true for Qwen3.8-Flash-Next).  Since we don't know n_layers up front, we
        // set bytes_per_layer lazily and refine downward as more layers arrive.
        if inner.bytes_per_layer.is_none() {
            // Estimate: 48 decoder layers is the Qwen3.8-Flash-Next default; the real value is
            // refined as layers arrive.  The quota can only stay the same or shrink, never grow,
            // because earlier layers have already claimed their slots.
            let estimated_layers = 48u64;
            let per_layer = inner.remaining_bytes / estimated_layers;
            inner.bytes_per_layer = Some(per_layer);
            info!(
                "expert cache: {:.1} GB total, ~{} slots/layer (estimated {} layers, {} bytes/slot)",
                self.cache_gb,
                per_layer / slot_bytes.max(1),
                estimated_layers,
                slot_bytes,
            );
        }

        let per_layer_budget = inner.bytes_per_layer.unwrap();
        let num_slots = ((per_layer_budget / slot_bytes.max(1)) as usize).min(num_experts);

        if num_slots == 0 {
            debug!(
                "expert cache: layer {} gets 0 slots ({} bytes/slot > {}/layer budget); skipping",
                ctx.site.layer, slot_bytes, per_layer_budget
            );
            drop(inner);
            let (pinned, prefill) =
                attach_prefill(&self.prefill, &ctx, &cpu_experts, &self.copy_streams)?;
            return Ok(Some(Arc::new(CpuExperts::new(
                cpu_experts,
                ctx.act,
                ctx.layer_device.clone(),
                ctx.site.layer,
                prefill,
                pinned,
            ))));
        }

        let layer_idx = ctx.site.layer;
        debug!(
            "expert cache: layer {} → {} GPU slots / {} experts ({}, {} B/slot)",
            layer_idx,
            num_slots,
            num_experts,
            if is_quantized { "quantized" } else { "dense" },
            slot_bytes,
        );

        // Charge the budget.
        let charged = slot_bytes * num_slots as u64;
        inner.remaining_bytes = inner.remaining_bytes.saturating_sub(charged);

        let hidden = ctx.config.hidden_size;
        let slot_device = ctx.layer_device.clone();

        drop(inner);

        // Page-lock the stacks so slot fills copy from them by DMA (after the huge-page advice:
        // pinned pages can no longer be collapsed).
        let (pinned, pin_failed) = host_memory::pin_expert_stacks(&cpu_experts, &slot_device);
        let pinned_bytes: usize = pinned.iter().map(PinnedStack::len).sum();
        debug!(
            "expert cache: layer {layer_idx} stacks: {} MiB on huge pages, {} MiB pinned",
            huge_bytes >> 20,
            pinned_bytes >> 20,
        );
        if pin_failed > 0 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                warn!(
                    "expert cache: some expert stacks could not be pinned; their slot fills \
                     copy from pageable memory (slower). Run with -v for details"
                )
            });
        }

        let prefill = match &self.prefill {
            Some(streams) => {
                streams.attach(&ctx, &cpu_experts, pin_failed == 0, &self.copy_streams)?
            }
            None => None,
        };

        // The same tensor the model loads as this block's router (`gate`); the decode
        // lookahead runs it on the CPU (R2a): the token thread publishes the activation
        // through a mapped flag channel (one kernel, no blocking D2H) and the worker
        // matmuls host-side. Without the router the layer is still cached, just never
        // predicted. (W92-W99 history: GPU router + logits D2H → CPU router + GPU
        // router + activation D2H → R2a channels; the weights are [128, 2048] F32,
        // ~1 MiB/layer on the host.)
        let router = match load_router(&ctx, num_experts, hidden, &Device::Cpu) {
            Ok(w) => Some(w),
            Err(e) => {
                debug!("expert cache: layer {layer_idx} has no router for lookahead: {e}");
                None
            }
        };
        // R2a: the publish channel, one per layer (each layer has its own worker-visible
        // activation; sized for the biggest MTP verify window: np4 + 1 target = 5 rows).
        let lookahead_chan = if slot_device.is_cuda() && router.is_some() {
            match LookaheadChannel::new(&slot_device, 5, hidden) {
                Ok(Some(c)) => Some(c),
                Ok(None) => {
                    debug!("expert cache: no mapped memory for the lookahead channel");
                    None
                }
                Err(e) => {
                    debug!("expert cache: lookahead channel failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // With a copy stream, each slot's memory is allocated once here and every fill overwrites
        // it in place from the copy stream; otherwise each fill uploads a fresh copy.
        // The slots are grouped in banks, one allocation per projection, so that a decode step
        // runs all of a bank's hits in one fused call; with a prefill stream the last bank is
        // the one lent to it.
        let tail = if prefill.is_some() {
            loan_bank_slots(num_experts).min(num_slots)
        } else {
            0
        };
        let banks: Vec<(usize, usize)> = [(0, num_slots - tail), (num_slots - tail, tail)]
            .into_iter()
            .filter(|&(_, len)| len > 0)
            .collect();
        let mut copy_stream = self.copy_streams.get(&slot_device);
        let mut slot_bufs = Vec::new();
        if let (Some(cs), Some(stacks)) = (&copy_stream, cpu_experts.expert_qtensors()) {
            for &(first, len) in &banks {
                for buf in alloc_bank(cs, &stacks, first, len, &slot_device)? {
                    slot_bufs.push(SlotEntry::new(Some(buf)));
                }
            }
            cs.after_device_work()?;
        }
        if slot_bufs.is_empty() {
            copy_stream = None;
        }

        // `--pcie-frac`: the model-wide staging ring, where a decode step DMAs the misses it
        // will run on the GPU, is NOT built here. It first materializes in `stream_pcie` of the
        // first decode step: during model load the cache banks and weight shards fragment
        // device memory, and allocating the ring's contiguous bank in that window failed on
        // some load orders (W105: 33 MiB at 11.6 GiB free, OOM in 2 of 3 runs). Once load has
        // finished the allocator has large free blocks left, where it fits reliably.
        // The ring's `first` sits past every resident slot, and it is kept out of
        // `slot_bufs`/`banks`, so `lend`/`give_back`/`admit`/`replace` never see it; `run_hits`
        // picks it up only because the injected hits carry its `Arc<Bank>`.

        let cached = Arc::new(CachedLayer {
            _pinned: pinned,
            cpu: cpu_experts,
            slot_device,
            layer: layer_idx,
            registry: Arc::clone(&self.registry),
            hidden,
            num_experts,
            top_k: ctx.config.num_experts_per_tok,
            act: ctx.act,
            router,
            lookahead_chan,
            #[cfg(feature = "cuda")]
            cpu_chan: std::sync::OnceLock::new(),
            state: Mutex::new(CachedLayerState {
                doorbell: None,
                doorbell_seq: 0,
                slots: HashMap::new(),
                resident: vec![None; num_slots],
                hit_counts: vec![0u64; num_experts],
                last_use: vec![0u64; num_slots],
                tick: 0,
                published: vec![None; num_slots],
                usable: num_slots,
            }),
            copy_stream,
            slot_bufs,
            banks,
            pcie_frac: self.pcie_frac,
            lookahead_swaps: self.lookahead_swaps,
            prefill,
            filling: AtomicBool::new(false),
            preload_claimed: AtomicU32::new(0),
        });
        self.registry.insert(layer_idx, &cached);
        Ok(Some(Arc::new(CachedExperts(cached))))
    }
}

fn load_router(
    ctx: &ExpertLoadContext<'_>,
    num_experts: usize,
    hidden: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    ctx.vb
        .pp("gate")
        .get((num_experts, hidden), "weight")?
        .to_device(device)?
        .to_dtype(DType::F32)
}

/// R2a: the lookahead's per-layer publish channel (GPU -> host). At each MoE entry the
/// token thread queues ONE kernel that copies this step's activation into mapped memory
/// and bumps seq; the lookahead worker spins for the seq and runs the router matmul on
/// the CPU (the router weights moved back to the host, `[128, 2048]` F32 ~ 1 MiB/layer).
/// The token thread never waits — no `to_vec1` D2H (W105b: 0.30-0.34 ms/layer of blocking
/// wait) and no way for the prediction to queue late on the stream (the W99 failure that
/// made `NAIWA_LH_MODE=new` regress hit 74% -> 38-45%).
struct LookaheadChannel {
    /// The channel's shared half: the mapped FlagBuf plus the publish/consume
    /// counters the token thread (publish) and lookahead worker (consume) share.
    inner: std::sync::Arc<LookaheadShared>,
}

/// Per-channel shared state. `next_seq` is touched only by the token thread;
/// `consumed` is written by whichever worker job ran last for this channel.
struct LookaheadShared {
    chan: Arc<crate::doorbell::FlagBuf>,
    /// The seq the last queued GPU publish will bump the counter to (0 before any):
    /// `last_seq + 1` is the value `wait_publish` waits for.
    next_seq: std::sync::atomic::AtomicU32,
    /// How many queued publishes the worker has consumed (`wait_publish` returned).
    /// The difference to `next_seq` is the queue depth; publish backpressure keeps
    /// it below `RING_DEPTH` so a lagging job's slot is never overwritten.
    consumed: std::sync::atomic::AtomicU32,
}

impl LookaheadChannel {
    /// The channel plus the CPU matmul weights `router` must already provide.
    ///
    /// `max_rows` must cover the LARGEST decode batch this layer will see — not
    /// just np+1. The target's verify batch is `staged + 1` rows and the MTP
    /// drafter runs its own windows; sizing for one shape and receiving a
    /// bigger one overflowed the slot and (on the last ring slot) wrote past
    /// the whole allocation into the neighboring doorbell buffer: garbage
    /// router ids shaped like the activation's f32 bits (W106). 16 rows is
    /// headroom over any current window (np4 verify = 5, observed verify = 7).
    fn new(device: &Device, max_rows: usize, hidden: usize) -> candle_core::Result<Option<Self>> {
        let max_rows = max_rows.max(16);
        Ok(crate::doorbell::FlagBuf::acquire(device, max_rows * hidden)
            .ok()
            .flatten()
            .map(|chan| Self {
                inner: std::sync::Arc::new(LookaheadShared {
                    chan: Arc::new(chan),
                    next_seq: std::sync::atomic::AtomicU32::new(0),
                    consumed: std::sync::atomic::AtomicU32::new(0),
                }),
            }))
    }

    /// Queue the publish on the stream (token thread): one kernel writes the `[rows,
    /// hidden]` activation into a slot and bumps seq. Returns the `LookaheadShared`
    /// handle plus the `(last_seq, slot)` the worker job needs to wait and read, or
    /// `None` when the worker is more than `RING_DEPTH - 1` publishes behind — the
    /// prediction is advisory, so skipping this publish is cheaper than overwriting
    /// an unread slot.
    fn publish(
        &self,
        x: &Tensor,
        n: usize,
    ) -> candle_core::Result<Option<(std::sync::Arc<LookaheadShared>, u32, usize)>> {
        let sh = &self.inner;
        // Hard error, not a skip: an oversized publish would overflow the slot
        // (the FlagBuf checks too, but fail before queueing anything).
        if n > sh.chan.max_n() {
            candle_core::bail!(
                "lookahead publish of {n} f32 exceeds the channel slot ({})",
                sh.chan.max_n()
            );
        }
        let next = sh.next_seq.load(Ordering::Relaxed);
        let consumed = sh.consumed.load(Ordering::Relaxed);
        if next - consumed
            >= (crate::doorbell::FlagBuf::RING_DEPTH as u32).saturating_sub(1)
        {
            return Ok(None); // backpressure: an unread slot would be overwritten
        }
        let slot = next as usize; // % RING_DEPTH inside publish_f32/wait_publish
        sh.chan.publish_f32(x, n, slot)?;
        sh.next_seq.store(next + 1, Ordering::Relaxed);
        Ok(Some((std::sync::Arc::clone(sh), next, slot)))
    }
}


// ---------------------------------------------------------------------------
// CachedLayer — the real workhorse
// ---------------------------------------------------------------------------

/// The most-routed expert not in (or reserved for) a slot, ties to the lowest id; `None` when
/// every slot is taken.
fn fill_candidate(resident: &[Option<usize>], hit_counts: &[u64]) -> Option<usize> {
    if !resident.contains(&None) {
        return None;
    }
    let mut taken = vec![false; hit_counts.len()];
    for &e in resident.iter().flatten() {
        taken[e] = true;
    }
    (0..hit_counts.len())
        .filter(|&e| !taken[e])
        .max_by_key(|&e| (hit_counts[e], std::cmp::Reverse(e)))
}

/// Indices of the `k` largest `logits`, largest first (ties: lowest index).
fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

/// Every expert in `rankings` once, ordered by how many rankings list it, then by its best rank,
/// then by id.
fn union_by_votes(rankings: impl IntoIterator<Item = Vec<usize>>) -> Vec<usize> {
    let mut seen: HashMap<usize, (usize, usize)> = HashMap::new(); // expert -> (votes, best rank)
    for ranking in rankings {
        for (rank, e) in ranking.into_iter().enumerate() {
            let entry = seen.entry(e).or_insert((0, rank));
            entry.0 += 1;
            entry.1 = entry.1.min(rank);
        }
    }
    let mut union: Vec<(usize, (usize, usize))> = seen.into_iter().collect();
    union.sort_by(|(a, (va, ra)), (b, (vb, rb))| vb.cmp(va).then(ra.cmp(rb)).then(a.cmp(b)));
    union.into_iter().map(|(e, _)| e).collect()
}

/// The misses to stream over PCIe (`--pcie-frac`): the distinct experts a step missed, ranked by
/// the total router weight routed to them, truncated to `cap`. An expert missed by several rows
/// is DMA'd once and serves them all, so the distinct count is what `frac` applies to; when the
/// fraction does not cover every miss, the ones that contribute most to the output go first.
fn streamed_misses(misses: &[Vec<(u32, f32)>], frac: f32, cap: usize) -> Vec<u32> {
    if frac <= 0.0 || cap == 0 {
        return Vec::new();
    }
    let mut weight: HashMap<u32, f32> = HashMap::new();
    for (id, w) in misses.iter().flatten() {
        *weight.entry(*id).or_insert(0.0) += *w;
    }
    let take = ((frac * weight.len() as f32).ceil() as usize).min(cap);
    let mut ranked: Vec<(u32, f32)> = weight.into_iter().collect();
    ranked.sort_by(|(a, wa), (b, wb)| wb.total_cmp(wa).then(a.cmp(b)));
    ranked.truncate(take);
    ranked.into_iter().map(|(id, _)| id).collect()
}

/// The least-recently-used published resident that is not in `protect`, as `(expert, slot)`.
fn lru_victim(
    slots: &HashMap<usize, usize>,
    last_use: &[u64],
    protect: &[usize],
) -> Option<(usize, usize)> {
    slots
        .iter()
        .filter(|(e, _)| !protect.contains(e))
        .min_by_key(|(&e, &slot)| (last_use[slot], e))
        .map(|(&e, &slot)| (e, slot))
}

/// `NAIWA_LOOKAHEAD_DBG=1`: per decode step, the synchronous D2H of the activation that the lookahead
/// needs, in ms. `NAIWA_NO_LOOKAHEAD=1` skips the prediction (and its D2H) entirely, so the cache only
/// ever holds what prefill left, and the A/B is the price of the prediction against its benefit.
mod lookahead_probe {
    use std::sync::Mutex;

    const WINDOW: u64 = 384; // eight layer passes

    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    /// Token-thread cost of queueing the publish (the `publish` call, host side).
    pub(super) fn record_publish(ms: f64) {
        if !on() {
            return;
        }
        static ACC: Mutex<(u64, f64)> = Mutex::new((0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += ms;
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG publishes {} publish_ms/step {:.3}",
                a.0,
                a.1 / WINDOW as f64
            );
            *a = (0, 0.0);
        }
    }

    /// Worker-side job duration: `wait_publish` latency plus the CPU matmul.
    pub(super) fn record_job(ms: f64) {
        if !on() {
            return;
        }
        static ACC: Mutex<(u64, f64, f64)> = Mutex::new((0, 0.0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += ms;
        if ms > a.2 {
            a.2 = ms;
        }
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG jobs {} job_ms/step {:.3} worst {} ms",
                a.0,
                a.1 / WINDOW as f64,
                a.2
            );
            *a = (0, 0.0, 0.0);
        }
    }

    pub(super) fn record(d2h_ms: f64, queued: bool) {
        let on = *ON.get_or_init(|| {
            std::env::var_os("NAIWA_LOOKAHEAD_DBG").is_some() || !queued
        });
        if !on {
            return;
        }
        static ACC: Mutex<(u64, f64)> = Mutex::new((0, 0.0));
        let mut a = ACC.lock().unwrap();
        a.0 += 1;
        a.1 += d2h_ms;
        if a.0 % WINDOW == 0 {
            eprintln!(
                "LOOKAHEAD_DBG steps {} d2h_ms/step {:.3}",
                a.0,
                a.1 / WINDOW as f64
            );
            *a = (0, 0.0);
        }
    }

    fn on() -> bool {
        *ON.get_or_init(|| std::env::var_os("NAIWA_LOOKAHEAD_DBG").is_some())
    }
}

/// `NAIWA_STEP_DBG=1`: at each layer's `decode_forward` entry, is the GPU still busy with
/// the previous layers of this step (host-bound: the token thread could have queued more)
/// or already done (GPU-bound: the stream ran dry waiting for the host)? Reports the busy
/// and dry fractions plus the host wall time between layer entries — the number R2 (no
/// host syncs, host runs ahead) competes to shrink is the DRY fraction.

/// The CPU job's gemv time, reported per pass by `--round-stats` (`NAIWA_ROUND_STATS`).
pub(crate) fn record_cpu_compute(ns: u64) {
    round_stats::record_cpu_compute(ns);
}

/// Per-pass accounting of the three producers, so the bench can print where a round's MoE time went:
/// resident hits, staged experts (PCIe) and CPU-computed misses, plus the CPU compute time that the
/// deferred probe measured for this pass.
mod round_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    static HITS: AtomicU64 = AtomicU64::new(0);
    static STAGED: AtomicU64 = AtomicU64::new(0);
    static CPU_MISSES: AtomicU64 = AtomicU64::new(0);
    static LAYERS: AtomicU64 = AtomicU64::new(0);
    static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
    static CPU_NS: AtomicU64 = AtomicU64::new(0);
    static HEADER: Mutex<bool> = Mutex::new(false);

    pub(super) fn record_plan(hits: usize, staged: usize, cpu: usize) {
        HITS.fetch_add(hits as u64, Ordering::Relaxed);
        STAGED.fetch_add(staged as u64, Ordering::Relaxed);
        CPU_MISSES.fetch_add(cpu as u64, Ordering::Relaxed);
    }

    /// Called by the staging path: how many bytes this layer copied over PCIe.
    pub(super) fn record_staged_bytes(bytes: u64) {
        STAGED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Called by the CPU job when it finishes: the gemv time of one job.
    pub(super) fn record_cpu_compute(ns: u64) {
        CPU_NS.fetch_add(ns, Ordering::Relaxed);
    }

    /// Called once per pass (layer 0).
    pub(super) fn finish_pass(on: bool) {
        let layers = LAYERS.fetch_add(1, Ordering::Relaxed) + 1;
        if !on || layers % 4 != 0 {
            return;
        }
        let h = HITS.swap(0, Ordering::Relaxed);
        let st = STAGED.swap(0, Ordering::Relaxed);
        let cpu = CPU_MISSES.swap(0, Ordering::Relaxed);
        let bytes = STAGED_BYTES.swap(0, Ordering::Relaxed);
        let cpu_ns = CPU_NS.swap(0, Ordering::Relaxed);
        let sel = h + st + cpu;
        if sel == 0 {
            return;
        }
        let mut hdr = HEADER.lock().unwrap();
        if !*hdr {
            *hdr = true;
            eprintln!(
                "ROUND_STATS rounds hit% staged% cpu% staged_MiB cpu_ms/stage"
            );
        }
        eprintln!(
            "ROUND_STATS {layers} {:.1} {:.1} {:.1} {:.1} {:.2}",
            100.0 * h as f64 / sel as f64,
            100.0 * st as f64 / sel as f64,
            100.0 * cpu as f64 / sel as f64,
            bytes as f64 / (1024.0 * 1024.0),
            cpu_ns as f64 / 1e6,
        );
    }
}

mod step_probe {
    use std::sync::Mutex;
    use std::time::Instant;

    use candle_core::Device;

    const WINDOW: usize = 48; // one token's worth of layers

    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(feature = "cuda")]
    mod imp {
        use candle_core::cuda::cudarc::driver::sys;
        use candle_core::Device;

        pub(super) struct Event(candle_core::cuda::cudarc::driver::CudaEvent);

        impl Event {
            pub(super) fn new(device: &Device) -> candle_core::Result<Self> {
                let Device::Cuda(dev) = device else {
                    unreachable!("caller checks is_cuda");
                };
                let event = dev
                    .cuda_stream()
                    .context()
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(candle_core::Error::wrap)?;
                Ok(Self(event))
            }

            pub(super) fn record(&self, device: &Device) -> candle_core::Result<()> {
                let Device::Cuda(dev) = device else {
                    unreachable!("caller checks is_cuda");
                };
                self.0.record(&dev.cuda_stream()).map_err(candle_core::Error::wrap)
            }

            pub(super) fn is_complete(&self) -> bool {
                // cudarc's `is_complete` returns `bool` (false on query errors too).
                self.0.is_complete()
            }
        }
    }

    #[cfg(feature = "cuda")]
    use imp::Event;

    struct Acc {
        last: Option<(Instant, Option<Event>)>,
        busy_ns: u128,
        dry_ns: u128,
        wait_ns: u128,
        n: usize,
    }

    impl Acc {
        const EMPTY: Acc = Acc { last: None, busy_ns: 0, dry_ns: 0, wait_ns: 0, n: 0 };
    }

    static ACC: Mutex<Acc> = Mutex::new(Acc::EMPTY);

    // Per-round wall time (round = one pass, marked at layer 0), because with speculative decoding a round
    // yields more than one token and TPOT stops describing it.
    static ROUND: Mutex<(Option<Instant>, Vec<f64>)> = Mutex::new((None, Vec::new()));

    pub(super) fn mark_round(layer: usize) {
        if layer != 0 {
            return;
        }
        crate::expert_cache::round_stats::finish_pass(
            std::env::var_os("NAIWA_ROUND_STATS").is_some(),
        );
        let mut g = ROUND.lock().unwrap();
        let now = Instant::now();
        if let Some(prev) = g.0.replace(now) {
            let ms = now.saturating_duration_since(prev).as_secs_f64() * 1e3;
            // A round is tens of ms here (speculation makes it longer), so only filter real gaps:
            // between requests the sampler and the bench take hundreds of ms, and a round is never < 5 ms.
            if (5.0..500.0).contains(&ms) {
                g.1.push(ms);
                if g.1.len() == 8 {
                    let mut v = g.1.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    eprintln!(
                        "ROUND_DBG n={} min {:.2} p50 {:.2} max {:.2} mean {:.2}",
                        v.len(),
                        v[0],
                        v[v.len() / 2],
                        v[v.len() - 1],
                        v.iter().sum::<f64>() / v.len() as f64
                    );
                    g.1.clear();
                }
            }
        }
    }

    pub(super) fn mark(device: &Device, layer: usize) {
        mark_round(layer);
        let on = *ON.get_or_init(|| std::env::var_os("NAIWA_STEP_DBG").is_some());
        if !on || !device.is_cuda() {
            return;
        }
        let mut acc = ACC.lock().unwrap();
        if let Some((t_last, ev_last)) = acc.last.take() {
            let now = Instant::now();
            // The wall time this layer entry arrived after the previous one; the GPU was
            // BUSY for the part before the previous entry's work completed, DRY after.
            let wall = now.duration_since(t_last).as_nanos();
            match ev_last {
                Some(ev) if ev.is_complete() => acc.dry_ns += wall,
                Some(_) => acc.busy_ns += wall,
                None => acc.wait_ns += wall,
            }
            acc.n += 1;
        }
        let recorded = Event::new(device)
            .and_then(|ev| ev.record(device).map(|_| ev))
            .ok();
        acc.last = Some((Instant::now(), recorded));
        if acc.n >= WINDOW {
            let a = &mut *acc;
            let ms = |ns: u128| ns as f64 / 1e6;
            eprintln!(
                "STEP_DBG layers {} busy_ms/layer {:.3} dry_ms/layer {:.3} (no-event {:.3})",
                a.n,
                ms(a.busy_ns) / a.n as f64,
                ms(a.dry_ns) / a.n as f64,
                ms(a.wait_ns) / a.n as f64,
            );
            *a = Acc::EMPTY;
        }
    }
}

/// Every cached layer of one model, by decoder-layer index, so layer `l` can start filling
/// layer `l + 1`'s slots. Weak references: the model owns the layers, not the registry.
#[derive(Default)]
struct LayerRegistry {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
    /// The model-wide `--pcie-frac` staging rings, keyed by the shape of the expert stacks
    /// (dtype and per-projection dims): mixed-quant models give different layers different
    /// GGML dtypes, and one bank cannot serve two shapes (W105: 524800-byte experts against a
    /// 627200-byte slot). Each shape gets its own ring of `scratch_ring_slots()` slots.
    /// Behind a `Mutex` because `ScratchRing::acquire` mutates each ring's cursor.
    scratch: Mutex<HashMap<ScratchKey, Arc<Mutex<ScratchRing>>>>,
    /// v2 M1: how many rings have actually been *created* (not looked up). `warmup_rings` runs
    /// after load and before the first decode; any further increment means a ring materialized
    /// on the decode path, i.e. an address that was not fixed up front.
    ring_creations: AtomicUsize,
}

/// A staging ring's shape key: every projection's per-expert byte count (folds in dtype and
/// dims, and separates a layer whose gate/up/down are heterogeneously quantized), plus the
/// stacks' dtype.
type ScratchKey = ([usize; 3], u8);

impl LayerRegistry {
    fn insert(&self, layer: usize, cached: &Arc<CachedLayer>) {
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, Weak::new);
        }
        layers[layer] = Arc::downgrade(cached);
    }

    fn get(&self, layer: usize) -> Option<Arc<CachedLayer>> {
        self.layers.read().get(layer).and_then(Weak::upgrade)
    }

    fn all(&self) -> Vec<Arc<CachedLayer>> {
        self.layers
            .read()
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }

    /// The staging ring for this shape of expert stacks on `device`, creating it at the first
    /// decode step that needs it. `None` only on allocation failure — a layer on another
    /// device simply never finds `cs` for it (copy streams are per-device).
    fn scratch_ring(
        &self,
        cs: &CopyStream,
        stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
        device: &Device,
    ) -> candle_core::Result<Option<Arc<Mutex<ScratchRing>>>> {
        // The per-projection expert byte counts are the ground truth a ring must match (they
        // fold in dtype and dims, including per-projection differences inside one layer).
        let bytes: [usize; 3] = std::array::from_fn(|i| {
            expert_range(&[&stacks.0, &stacks.1, &stacks.2][i], 0)
                .map_or(0, |r| r.len())
        });
        let key: ScratchKey = (bytes, stacks.0.dtype() as u8);
        let mut rings = self.scratch.lock().unwrap();
        if let Some(ring) = rings.get(&key) {
            return Ok(Some(Arc::clone(ring)));
        }
        let len = scratch_ring_slots();
        let bufs = alloc_bank(cs, stacks, SCRATCH_RING_FIRST, len, device)?;
        let per_slot: usize = bytes.iter().sum();
        let mut slots = Vec::with_capacity(len);
        for buf in bufs {
            slots.push(ScratchSlot { buf });
        }
        let ring = Arc::new(Mutex::new(ScratchRing { slots, cursor: 0 }));
        info!(
            "--pcie-frac: {len} staging-ring slots for dtype {:?} shared by its layers \
             (~{} MiB total, beyond the cache), DMA'd misses run on the GPU",
            stacks.0.dtype(),
            (per_slot * len) >> 20
        );
        rings.insert(key, Arc::clone(&ring));
        self.ring_creations.fetch_add(1, Ordering::Relaxed);
        Ok(Some(ring))
    }
}

/// Lends the cache's tail slots to the prefill stream while a prompt runs (see
/// `prefill_stream`): each layer gives up an equal share of the bytes asked for.
struct CacheLoan {
    registry: Arc<LayerRegistry>,
}

impl ExpertMemoryLender for CacheLoan {
    fn lend(&self, bytes: usize) -> candle_core::Result<usize> {
        let layers = self.registry.all();
        let share = bytes.div_ceil(layers.len().max(1));
        Ok(layers
            .iter()
            .map(|l| {
                let per_slot = l.slot_bytes();
                if per_slot == 0 {
                    return 0;
                }
                l.lend(share.div_ceil(per_slot)) * per_slot
            })
            .sum())
    }

    fn give_back(&self) -> candle_core::Result<()> {
        for layer in self.registry.all() {
            layer.give_back()?;
        }
        Ok(())
    }
}

struct CachedLayerState {
    /// Step 5 doorbell: publishes this layer's top-k into mapped memory so the host
    /// read needs no stream sync or D2H (the pattern the doorbell harness proved).
    /// Acquired lazily at first decode; None on non-CUDA or alloc failure, where
    /// decode keeps the `to_vec2` path. In the state lock for interior mutability
    /// through the shared `Arc`.
    doorbell: Option<Arc<crate::doorbell::DoorbellBuf>>,
    /// seq the last ring of this layer advanced past (starts 0; first ring -> 1).
    doorbell_seq: u32,
    /// Published residents: expert id -> slot. Only experts whose weights are fully written
    /// appear here, and it is the only table readers consult.
    slots: HashMap<usize, usize>,
    /// Slot -> expert id, set when a slot is *reserved* (before its weights are written), so
    /// two admitters can never claim the same slot.
    resident: Vec<Option<usize>>,
    hit_counts: Vec<u64>,
    /// Slot -> `tick` of its last decode hit (or its fill); drives LRU replacement.
    last_use: Vec<u64>,
    /// Decode steps seen by this layer.
    tick: u64,
    /// Slot -> its contents, for the slots listed in `slots`. Readers clone the `Arc` under
    /// the lock, so a slot replaced later never changes what an in-flight token computes with.
    published: Vec<Option<Arc<SlotBuf>>>,
    /// Slots `[0, usable)` are in service; the rest are lent to the prefill stream (empty,
    /// unpublished, and without device memory in copy-stream mode).
    usable: usize,
}

/// One expert's projections in a slot.
#[derive(Clone)]
struct SlotWeights {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

/// Slots `[first, first + len)` of a layer in copy-stream mode: one `[len, N, K]` device
/// allocation per projection, which the fused decode kernel indexes by `slot - first`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Bank {
    first: usize,
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

/// One slot's contents. In copy-stream mode `memory` is the slot's view of its bank, overwritten
/// in place on every fill; otherwise each fill makes a new `SlotBuf` with its own `weights`.
struct SlotBuf {
    weights: Option<SlotWeights>,
    memory: Option<[SlotMemory; 3]>,
    /// Copy-stream mode: the bank this slot is part of (held, so the bank outlives readers).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    bank: Option<Arc<Bank>>,
    /// Recorded by readers after enqueueing the kernels that read this slot.
    fence: ReadFence,
}

/// A slot's permanent buffer in copy-stream mode; `None` while the slot is lent. An upload
/// holds the lock for its duration, so lending a slot waits for any upload into it.
type SlotEntry = parking_lot::Mutex<Option<Arc<SlotBuf>>>;

/// One `--pcie-frac` staging slot of the global ring: a VRAM buffer a decode step DMAs a missed
/// expert into, plus the event that marks the DMA done for candle's stream to wait on.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct ScratchSlot {
    buf: Arc<SlotBuf>,
}

/// The model-wide `--pcie-frac` staging ring (see `SCRATCH_RING_FIRST`): one bank of slots
/// past every resident slot, shared by every layer, borrowed round-robin per decode step.
///
/// `acquire(n)` runs under the registry lock, so consecutive layers get disjoint slots even
/// though all steps are on the token thread (the lock also covers a future multi-threaded
/// caller). Round-robin keeps the reuse distance at one full revolution of the ring: by the
/// time the cursor returns to a slot, several layers' readers have long recorded their fences,
/// so `wait_for_readers` never spins in practice.
/// Ring slots' `Bank.first` is `SCRATCH_RING_FIRST`, past every resident slot.
struct ScratchRing {
    slots: Vec<ScratchSlot>,
    cursor: usize,
}

impl ScratchRing {
    /// Advance the cursor by `n` and return where this step's slots start. `n <= len` always
    /// (the callers clamp to the ring size).
    fn acquire(&mut self, n: usize) -> usize {
        let len = self.slots.len();
        debug_assert!(n <= len);
        let start = self.cursor;
        self.cursor = (self.cursor + n) % len;
        start
    }

    /// The `n` slots starting at `from`, wrapping past the end, in that order.
    fn take<'a>(&'a self, from: usize, n: usize) -> Vec<&'a ScratchSlot> {
        let len = self.slots.len();
        (0..n).map(|i| &self.slots[(from + i) % len]).collect()
    }
}

/// Allocate the device memory of slots `[first, first + len)` as one bank on the copy stream's
/// device. The caller orders the copy stream after the allocation (`after_device_work`).
fn alloc_bank(
    cs: &CopyStream,
    stacks: &(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>),
    first: usize,
    len: usize,
    device: &Device,
) -> candle_core::Result<Vec<Arc<SlotBuf>>> {
    let (gate, gate_views) = cs.alloc_bank(&stacks.0, len)?;
    let (up, up_views) = cs.alloc_bank(&stacks.1, len)?;
    let (down, down_views) = cs.alloc_bank(&stacks.2, len)?;
    let bank = Arc::new(Bank {
        first,
        gate,
        up,
        down,
    });
    gate_views
        .into_iter()
        .zip(up_views)
        .zip(down_views)
        .map(|((g, u), d)| {
            Ok(Arc::new(SlotBuf {
                weights: None,
                memory: Some([g, u, d]),
                bank: Some(Arc::clone(&bank)),
                fence: ReadFence::new(device)?,
            }))
        })
        .collect()
}

impl Bank {}

/// The v2 M1/M2 fixed-grid batch bound, matching the lookahead channel's policy
/// (`LookaheadChannel::new`): it must cover the largest decode batch, with 16 rows of headroom
/// over any current window (np4 verify = 5, observed verify = 7). Calls above it use the old
/// host-geometry path — the allowed row-bucket fallback.
#[cfg(feature = "cuda")]
const CAPTURE_MAX_BATCH: usize = 16;

#[cfg(feature = "cuda")]
const WAITS_PER_LAYER: u32 = 3;

/// The fused MoE decode kernel's activation id, if it has one for `act`.
#[cfg(feature = "cuda")]
fn fused_act(act: Activation) -> Option<i32> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::ACT_SILU),
        Activation::GeluPytorchTanh => Some(mistralrs_quant::ACT_GELU_PYTORCH_TANH),
        _ => None,
    }
}

/// `sum_j weights[t, j] * expert(bank, ids[t, j])(x2d[t])` for each of `rows` tokens,
/// `[rows, hidden]` F32, by mistral.rs's fused MoE decode (quantize, gate+up+act, quantize,
/// down+weighted sum). `ids` and `weights` are row-major `[rows, topk]`.
#[cfg(feature = "cuda")]
fn fused_decode(
    bank: &Bank,
    x2d: &Tensor,
    ids: Vec<u32>,
    weights: Vec<f32>,
    rows: usize,
    act_type: i32,
) -> candle_core::Result<Tensor> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    let dev = x2d.device().as_cuda_device()?;
    let topk = ids.len() / rows;
    let ids = Tensor::new(ids, x2d.device())?;
    let weights = Tensor::new(weights, x2d.device())?;
    let (ids_storage, _) = ids.storage_and_layout();
    let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("expert cache: fused decode ids are not on CUDA");
    };
    let (w_storage, _) = weights.storage_and_layout();
    let candle_core::Storage::Cuda(w_cuda) = &*w_storage else {
        candle_core::bail!("expert cache: fused decode weights are not on CUDA");
    };
    let w_slice = w_cuda.as_cuda_slice::<f32>()?;
    let w_ptr = w_slice.device_ptr(w_slice.stream()).0 as *const f32;
    // SAFETY: `w_ptr` points at `topk` f32s of `weights`, whose free is ordered on candle's
    // stream after the kernels queued here.
    unsafe {
        mistralrs_quant::indexed_moe_fused_decode(
            &bank.gate,
            &bank.up,
            &bank.down,
            x2d,
            ids_cuda.as_cuda_slice::<u32>()?,
            w_ptr,
            rows,
            topk,
            act_type,
            dev,
        )
    }
}

/// A resident expert a token is routed to: its slot buffer, slot index and routing weight.
type Hit = (Arc<SlotBuf>, usize, f32);

struct CachedLayer {
    /// Page-locked registrations of `cpu`'s stacks; each holds its stack, so it stays valid
    /// for as long as it is registered.
    _pinned: Vec<PinnedStack>,
    cpu: MoEExperts,
    slot_device: Device,
    layer: usize,
    registry: Arc<LayerRegistry>,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    act: Activation,
    /// `[num_experts, hidden]` F32 on the CPU; `None` disables the lookahead for this layer.
    router: Option<Tensor>,
    /// R2a: the layer's GPU->host activation channel. `Some` only when the router exists
    /// and the slot device is CUDA.
    lookahead_chan: Option<LookaheadChannel>,
    /// M3: this layer's CPU-return channel, with its own `(slot, seq)` numbering. Created at
    /// warmup so a captured wait node is never stranded on a grown-and-retired channel, and so
    /// the numbering is 1:1 with this layer's jobs by construction.
    #[cfg(feature = "cuda")]
    cpu_chan: std::sync::OnceLock<Arc<deferred::CpuChannel>>,
    state: Mutex<CachedLayerState>,
    /// Uploads fills alongside compute; `None` off CUDA, where fills go through candle.
    copy_stream: Option<Arc<CopyStream>>,
    /// Copy-stream mode: every slot's permanent buffer (empty otherwise).
    slot_bufs: Vec<SlotEntry>,
    /// Copy-stream mode: the banks as `(first slot, len)`, in slot order. Lent whole.
    banks: Vec<(usize, usize)>,
    /// `--lookahead-swaps`: how many predicted experts may displace residents per row and pass
    /// (`0` means a prediction can only fill an empty slot).
    lookahead_swaps: usize,
    /// `--pcie-frac`: share of a step's misses to stream over PCIe into the model-wide staging
    /// ring (held by the registry, created lazily at the first decode — `stream_pcie`; 0.0
    /// disables the feature).
    pcie_frac: f32,
    /// Runs long prefill chunks' experts on the GPU (`--prefill-stream`).
    prefill: Option<Arc<PrefillStream>>,
    /// A background fill of this layer is running; at most one at a time.
    filling: AtomicBool,
    /// v2 M5b: the pass whose preload copies the prediction thread has claimed (0 = none). Claimed
    /// before the copies are queued and before the point is published, so the token thread replays
    /// the preload gate exactly when a publish is guaranteed to arrive for its pass.
    preload_claimed: AtomicU32,
}

impl CachedLayer {
    /// v2 M4: is the PCIe staging path in service for this layer? One predicate for all three
    /// sites — capture the DMA gate, publish the point, replay the gate — because they must agree:
    /// a gate whose point is never published spins forever, and at `--pcie-frac 0` the copy stream
    /// still exists (the cache banks use it) while nothing is ever staged.
    #[cfg(feature = "cuda")]
    fn staging_enabled(&self) -> bool {
        self.pcie_frac > 0.0 && self.copy_stream.is_some()
    }

    /// M4/M5b/M6: gate this layer's segment on `kind` by launching the wait node on the stream.
    ///
    /// M6 folds the gate graphs away: the node was already launched warm whenever capture failed,
    /// and a plain launch is *cheaper* than replaying a one-node graph (measured 5.9 us vs 15.4 us
    /// per launch), so a captured gate buys nothing. Stream ordering is identical.
    #[cfg(feature = "cuda")]
    fn replay_gate(&self, kind: u32) -> candle_core::Result<()> {
        let Some(sched) = schedule() else {
            return Ok(());
        };
        COUNTERS.gate_replays.fetch_add(1, Ordering::Relaxed);
        let id = sched.sync_id(self.layer as u32, kind);
        sched.wait_ready(&self.slot_device, id)
    }

    /// v2 M1: create this layer's staging ring now (see `GpuCacheMoeProvider::warmup_rings`).
    /// A no-op when the layer has no copy stream or no CPU `QTensor`s. Called once after load.
    fn warmup_scratch_ring(&self) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        self.registry.scratch_ring(
            cs,
            &(
                Arc::clone(&gate),
                Arc::clone(&up),
                Arc::clone(&down),
            ),
            &self.slot_device,
        )?;
        Ok(())
    }

    /// Byte-slice expert `expert_id` out of the CPU stacks onto the slot device — never
    /// dequantized. `None` when the backend holds no `QTensor`s.
    fn expert_weights(&self, expert_id: usize) -> candle_core::Result<Option<SlotWeights>> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(None);
        };
        Ok(Some(SlotWeights {
            gate: expert_qtensor_on_device(&gate, expert_id, &self.slot_device)?,
            up: expert_qtensor_on_device(&up, expert_id, &self.slot_device)?,
            down: expert_qtensor_on_device(&down, expert_id, &self.slot_device)?,
        }))
    }

    /// Put `expert_id` into an empty slot. The slot is reserved under the lock before the
    /// (slow) copy and published only after it, so concurrent admitters never share a slot
    /// and readers never see a half-written one. `None`: already resident or no empty slot.
    fn admit(&self, expert_id: usize, sync: bool) -> candle_core::Result<Option<usize>> {
        let slot = {
            let mut st = self.state.lock().unwrap();
            if st.resident.contains(&Some(expert_id)) {
                return Ok(None);
            }
            let usable = st.usable;
            let Some(slot) = st.resident[..usable].iter().position(Option::is_none) else {
                return Ok(None);
            };
            st.resident[slot] = Some(expert_id);
            slot
        };
        self.load_reserved(expert_id, slot, sync)
    }

    /// Replace the least-recently-used resident outside `protect` with `expert_id`. The victim
    /// is unpublished and the slot reserved in one critical section. Tokens already holding the
    /// victim keep computing with it: a fresh upload leaves the old buffer alone, and an
    /// in-place one waits for them (`upload_in_place`).
    fn replace(
        &self,
        expert_id: usize,
        protect: &[usize],
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        let slot = {
            let mut st = self.state.lock().unwrap();
            if st.resident.contains(&Some(expert_id)) {
                return Ok(None);
            }
            let Some((victim, slot)) = lru_victim(&st.slots, &st.last_use, protect) else {
                return Ok(None);
            };
            st.slots.remove(&victim);
            st.published[slot] = None;
            st.resident[slot] = Some(expert_id);
            slot
        };
        self.load_reserved(expert_id, slot, sync)
    }

    /// Copy `expert_id` into `slot`, which the caller has reserved, then publish it. A failed
    /// copy frees the slot.
    fn load_reserved(
        &self,
        expert_id: usize,
        slot: usize,
        sync: bool,
    ) -> candle_core::Result<Option<usize>> {
        let loaded = match &self.copy_stream {
            Some(cs) => self.upload_in_place(cs, expert_id, slot, sync),
            None => self.expert_weights(expert_id).map(|w| {
                w.map(|weights| {
                    Arc::new(SlotBuf {
                        weights: Some(weights),
                        memory: None,
                        bank: None,
                        fence: ReadFence::none(),
                    })
                })
            }),
        };
        let mut st = self.state.lock().unwrap();
        match loaded {
            // Lent while this upload ran: drop it rather than publish into a lent slot.
            Ok(Some(_)) if slot >= st.usable => {
                st.resident[slot] = None;
                Ok(None)
            }
            Ok(Some(buf)) => {
                st.published[slot] = Some(buf);
                st.slots.insert(expert_id, slot);
                st.last_use[slot] = st.tick;
                Ok(Some(slot))
            }
            Ok(None) => {
                st.resident[slot] = None;
                Ok(None)
            }
            Err(e) => {
                st.resident[slot] = None;
                Err(e)
            }
        }
    }

    /// Overwrite `slot`'s device memory with `expert_id` on the copy stream. The slot is
    /// reserved and unpublished, so no new reader can reach it; this waits for the readers
    /// that captured it earlier to record their fence, then the copy waits on that fence.
    fn upload_in_place(
        &self,
        cs: &CopyStream,
        expert_id: usize,
        slot: usize,
        sync: bool,
    ) -> candle_core::Result<Option<Arc<SlotBuf>>> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(None);
        };
        // Held for the whole upload: lending this slot waits for it.
        let entry = self.slot_bufs[slot].lock();
        let Some(buf) = entry.as_ref() else {
            return Ok(None); // lent
        };
        let Some(memory) = &buf.memory else {
            return Ok(None);
        };
        copy_stream::wait_for_readers(buf);
        let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
        let parts = [
            (&memory[0], &g[expert_range(&gate, expert_id)?]),
            (&memory[1], &u[expert_range(&up, expert_id)?]),
            (&memory[2], &d[expert_range(&down, expert_id)?]),
        ];
        if sync {
            cs.upload(&buf.fence, &parts)?;
        } else {
            // M5b: the caller claims this pass and publishes `SYNC_PRELOAD` on this same stream right
            // after the copies, so the layer's gate — not a host wait — orders the reader. Blocking
            // here was 81% of a prediction job.
            // SAFETY: the sources are page-locked stacks that outlive the copies, and the caller
            // publishes the completion point on the same stream.
            unsafe { cs.queue_async(&buf.fence, &parts)? };
        }
        Ok(Some(Arc::clone(buf)))
    }

    /// Bytes of one slot: one expert's gate, up and down.
    fn slot_bytes(&self) -> usize {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return 0;
        };
        [&gate, &up, &down]
            .iter()
            .map(|s| expert_range(s, 0).map_or(0, |r| r.len()))
            .sum()
    }

    /// Take at least the top `k` usable slots out of service for the prefill stream (whole banks
    /// in copy-stream mode): unpublish and empty them, then release their device memory. A free
    /// is ordered on candle's stream after any kernel already reading the slot. Returns the
    /// number of slots lent.
    fn lend(&self, k: usize) -> usize {
        let (from, to) = {
            let mut st = self.state.lock().unwrap();
            let to = st.usable;
            let want = to.saturating_sub(k);
            // A bank's memory is one allocation: lend from the start of the bank holding `want`.
            let from = self
                .banks
                .iter()
                .map(|&(first, _)| first)
                .filter(|&first| first <= want)
                .max()
                .unwrap_or(want);
            for slot in from..to {
                if let Some(e) = st.resident[slot].take() {
                    if st.slots.get(&e) == Some(&slot) {
                        st.slots.remove(&e);
                    }
                }
                st.published[slot] = None;
            }
            st.usable = from;
            (from, to)
        };
        for entry in &self.slot_bufs[from.min(self.slot_bufs.len())..to.min(self.slot_bufs.len())] {
            // Waits for an upload into the slot; its result is then discarded.
            if let Some(buf) = entry.lock().take() {
                copy_stream::wait_for_readers(&buf);
            }
        }
        to - from
    }

    /// Put every lent slot back in service, empty (with fresh device memory in copy-stream
    /// mode), and refill the layer in the background by routing count.
    fn give_back(self: &Arc<Self>) -> candle_core::Result<()> {
        let (from, to) = {
            let st = self.state.lock().unwrap();
            (st.usable, st.resident.len())
        };
        if from == to {
            return Ok(());
        }
        if let (Some(cs), Some(stacks)) = (&self.copy_stream, self.cpu.expert_qtensors()) {
            for &(first, len) in self.banks.iter().filter(|&&(first, _)| first >= from) {
                let bufs = alloc_bank(cs, &stacks, first, len, &self.slot_device)?;
                for (entry, buf) in self.slot_bufs[first..first + len].iter().zip(bufs) {
                    *entry.lock() = Some(buf);
                }
            }
            cs.after_device_work()?;
        }
        self.state.lock().unwrap().usable = to;
        self.spawn_fill();
        Ok(())
    }

    /// This layer's predicted experts for `rows` activations `x` (`[rows, hidden]`), by its own
    /// router: the union of each row's top-k, the experts most rows want first.
    ///
    /// R2a: the router lives on the CPU and `x` arrives as a CPU tensor (read from the layer's
    /// mapped channel in the lookahead job), so the matmul never touches the GPU stream — no
    /// `to_device`, no `dev_synchronize`. The W92-W99 stream-position hazards (prediction
    /// queuing behind the calling layer's tail kernels, late arrivals halving the hit rate)
    /// cannot occur: the activation is published by a stream-ordered kernel the moment the
    /// layer's compute is done.
    fn predict(&self, x: &Tensor) -> candle_core::Result<Vec<usize>> {
        let Some(router) = &self.router else {
            return Ok(Vec::new());
        };
        let rows = x.dims().first().copied().unwrap_or(0);
        let x = x.to_dtype(DType::F32)?.reshape((rows, self.hidden))?;
        let logits = x.matmul(&router.t()?)?;
        Ok(union_by_votes(
            logits.to_vec2::<f32>()?.iter().map(|row| top_k_ids(row, self.top_k)),
        ))
    }

    /// M5b: turn the prediction for `x` into **asynchronous** copies on the copy stream, then claim
    /// this pass and publish `SYNC_PRELOAD` for it — all from the prediction thread, in the gap after
    /// the CPU-expert batch, so the DMA hides in the host's work for the next layer.
    ///
    /// The copies are queued *before* the claim is published, and the publish is monotonic
    /// (`publish_point_on_max`), so the token thread either sees no claim (nothing to wait for) or
    /// sees one whose point is already on its way and can only move the word forward.
    fn prefetch_predicted(&self, x: &Tensor, pass: u32) -> candle_core::Result<()> {
        let rows = x.dims().first().copied().unwrap_or(0);
        let predicted = self.predict(x)?;
        if self.copy_stream.is_none() {
            return Ok(());
        }
        let max_swaps = self.lookahead_swaps * rows;
        let mut swaps = 0;
        for &e in &predicted {
            if e >= self.num_experts {
                continue;
            }
            if self.admit(e, false)?.is_some() {
                COUNTERS.lookahead_loads.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if swaps < max_swaps && self.replace(e, &predicted, false)?.is_some() {
                swaps += 1;
                COUNTERS.swaps.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Claim then publish: a reader that sees `pass` knows the point is coming.
        if let Some(sched) = schedule() {
            self.preload_claimed.store(pass, Ordering::Release);
            let id = sched.sync_id(self.layer as u32, crate::doorbell::SYNC_PRELOAD);
            sched.publish_point_on_max(&self.slot_device, id, pass)?;
        }
        Ok(())
    }

    /// The next expert to fill into this layer, if a slot is still empty.
    fn next_fill_candidate(&self) -> Option<usize> {
        let st = self.state.lock().unwrap();
        fill_candidate(&st.resident[..st.usable], &st.hit_counts)
    }

    /// Fill every empty slot of this layer on a background thread and return immediately.
    /// No-op when the layer is full or a fill is already running.
    fn spawn_fill(self: &Arc<Self>) {
        if self.next_fill_candidate().is_none() {
            return;
        }
        if self
            .filling
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            while let Some(e) = this.next_fill_candidate() {
                // Background fill: synchronous copies, no per-pass gate protects these slots.
                match this.admit(e, true) {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(err) => {
                        debug!("expert cache: layer {} fill stopped: {err}", this.layer);
                        break;
                    }
                }
            }
            this.filling.store(false, Ordering::Release);
        });
    }

    /// The weighted sum of the hits' outputs, `[rows, hidden]` F32 on the slot device, for
    /// `hits[t]` the hits of row `t` of `x2d: [rows, hidden]`. In copy-stream mode each bank's hits
    /// run as one fused MoE decode over every row (four kernels); otherwise every expert runs on
    /// its own.
    fn run_hits(
        &self,
        hits: &[Vec<Hit>],
        x2d: &Tensor,
        act: Activation,
    ) -> candle_core::Result<(Tensor, Option<Arc<Bank>>)> {
        let rows = hits.len();
        // M9: no captures, so nothing to remember for the CPU-return node.
        let capture_bank: Option<Arc<Bank>> = None;
        let mut acc: Option<Tensor> = None;
        let mut add = |y: Tensor| -> candle_core::Result<()> {
            acc = Some(match acc.take() {
                Some(a) => (a + y)?,
                None => y,
            });
            Ok(())
        };
        #[cfg(feature = "cuda")]
        if let (Some(act_type), true) = (
            fused_act(act),
            hits.iter().flatten().all(|(buf, _, _)| buf.bank.is_some()),
        ) {
            // Per bank, each row's (local slot, weight) pairs. The kernel takes the same number
            // per row, so short rows are padded with a weight-0 entry on a slot of that bank that
            // some row hits (published and fenced, so it holds finite values).
            let mut groups: Vec<(&Arc<Bank>, Vec<Vec<(u32, f32)>>)> = Vec::new();
            for (t, row) in hits.iter().enumerate() {
                for (buf, slot, w) in row {
                    let bank = buf.bank.as_ref().expect("checked above");
                    let local = (slot - bank.first) as u32;
                    let i = match groups.iter().position(|(b, _)| Arc::ptr_eq(b, bank)) {
                        Some(i) => i,
                        None => {
                            groups.push((bank, vec![Vec::new(); rows]));
                            groups.len() - 1
                        }
                    };
                    groups[i].1[t].push((local, *w));
                }
            }
            for (bank, per_row) in groups {
                let width = per_row.iter().map(Vec::len).max().unwrap_or(0);
                let pad = per_row.iter().flatten().next().map_or(0, |&(l, _)| l);
                let mut ids = Vec::with_capacity(rows * width);
                let mut ws = Vec::with_capacity(rows * width);
                for row in &per_row {
                    for i in 0..width {
                        let (l, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                        ids.push(l);
                        ws.push(w);
                    }
                }
                add(fused_decode(bank, x2d, ids, ws, rows, act_type)?)?;
            }
            return match acc {
                Some(acc) => Ok((acc, capture_bank)),
                None => Ok((
                    Tensor::zeros((rows, self.hidden), DType::F32, &self.slot_device)?,
                    None,
                )),
            };
        }
        // One expert at a time, row by row.
        let mut out = Vec::with_capacity(rows);
        for (t, row) in hits.iter().enumerate() {
            let x = x2d.narrow(0, t, 1)?;
            let mut sum = Tensor::zeros((1, self.hidden), DType::F32, &self.slot_device)?;
            for (buf, _, w) in row {
                let Some(weights) = &buf.weights else {
                    candle_core::bail!("expert cache: slot has no weights to run");
                };
                let y = Self::run_expert(weights, &x, act)?;
                sum = (sum + (y.to_dtype(DType::F32)? * *w as f64)?)?;
            }
            out.push(sum);
        }
        add(Tensor::cat(&out, 0)?)?;
        Ok((acc.expect("added above"), capture_bank))
    }

    /// Run one token through a cached expert. `x2d: [1, hidden]` on the slot device.
    fn run_expert(w: &SlotWeights, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
        let g = QMatMul::QTensor(w.gate.clone()).forward(x2d)?;
        let u = QMatMul::QTensor(w.up.clone()).forward(x2d)?;
        let h = (act.forward(&g)? * u)?;
        QMatMul::QTensor(w.down.clone()).forward(&h)
    }

    /// Decode path (one token, or the few rows of a speculative verify step): split each row's
    /// top-k into GPU hits and CPU misses, run both concurrently, sum. Misses are never loaded on
    /// the token path — they run on the CPU.
    ///
    /// Every CUDA call stays on the calling thread; the misses run on the `deferred` worker over
    /// host copies. Launching the hits from a second thread while this one copied the activation
    /// to the host produced NaN logits on CUDA once the lookahead was swapping slots.
    ///
    /// Returns without waiting for the misses: their sum is copied in on candle's stream when
    /// the worker is done (`deferred`), so the next layer's launches overlap the CPU experts.
    /// `--pcie-frac`: DMA the highest-weight misses of this step into the scratch slots and turn
    /// them into hits, so `run_hits` runs them on the GPU with the residents and the rest stay on
    /// the CPU. The chosen ids are removed from `misses`, leaving the caller's CPU padding and
    /// `deferred::start` untouched. The injected hits carry the scratch `SlotBuf`, so the caller's
    /// fence loop covers them and the next step's DMA waits on this step's readers.
    ///
    /// `upload_async` returns as soon as the copy is queued; the layer's DMA gate (a captured wait
    /// node, M4) spins until the copy stream has published `SYNC_STAGING` for this pass, so candle's
    /// stream never waits on the copy stream and the host enqueues nothing per streamed expert.
    fn stream_pcie(
        self: &Arc<Self>,
        misses: &mut [Vec<(u32, f32)>],
        hits: &mut [Vec<Hit>],
        pass: u32,
    ) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let staged = self.stage_misses(cs, misses, hits);
        // M4: release this layer's DMA gate on *every* path — nothing staged, staged, or failed.
        // The gate is replayed for every layer of every pass, so a point left unpublished would
        // spin forever. This is a stream memory op on the copy stream, ordered after the copies
        // above it, so a node spinning on the word cannot see it before the data has landed.
        #[cfg(feature = "cuda")]
        if let Some(sched) = schedule() {
            let id = sched.sync_id(self.layer as u32, crate::doorbell::SYNC_STAGING);
            sched.publish_point_on(cs.cu_stream(), id, pass)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pass;
        staged
    }

    /// The DMA half of [`Self::stream_pcie`]: pick this step's misses, queue their copies on the
    /// copy stream, and turn them into hits.
    ///
    /// The copies record no completion event: the caller publishes this layer's `SYNC_STAGING`
    /// point on this same stream right after them, and the layer's gate graph spins on that word.
    fn stage_misses(
        &self,
        cs: &Arc<CopyStream>,
        misses: &mut [Vec<(u32, f32)>],
        hits: &mut [Vec<Hit>],
    ) -> candle_core::Result<()> {
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        // Lazily create the ring at the first decode step that has misses to stream (model
        // load is over by then, so the allocation is not fighting load-time fragmentation).
        let Some(ring) = self.registry.scratch_ring(
            cs,
            &(
                Arc::clone(&gate),
                Arc::clone(&up),
                Arc::clone(&down),
            ),
            &self.slot_device,
        )?
        else {
            return Ok(());
        };
        // Everything below runs under the ring mutex: the whole pipeline is on the token
        // thread, so this lock only excludes the (one-time) creation race in `scratch_ring`.
        // Working on `&ScratchSlot` directly matters: `wait_for_readers` waits for the refcount
        // to fall back to the ring's own `Arc` — cloning into a borrow list first would count
        // that clone and spin forever (W105: three 900s-timeout hangs, exactly here).
        let slot_of = {
            let mut ring = ring.lock().unwrap();
            let chosen = streamed_misses(misses, self.pcie_frac, ring.slots.len());
            if chosen.is_empty() {
                HashMap::new()
            } else {
                let len = ring.slots.len();
                let start = ring.acquire(chosen.len());
                let taken = ring.take(start, chosen.len());
                let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
                let mut slot_of: HashMap<u32, (Arc<SlotBuf>, usize)> =
                    HashMap::with_capacity(chosen.len());
                for (i, scratch) in taken.into_iter().enumerate() {
                    let expert = chosen[i];
                    let pos = (start + i) % len;
                    // Same guard as `upload_in_place`: wait for the readers that captured this
                    // slot last step of the ring (they record their fence before dropping it),
                    // then let the copy stream wait on that fence before overwriting. This step
                    // is itself the next reader through the `Arc` injected into `slot_of`.
                    copy_stream::wait_for_readers(&scratch.buf);
                    let buf = &scratch.buf;
                    let memory = buf.memory.as_ref().expect("scratch slot has memory");
                    let (er, e) = (expert_range(&gate, expert as usize)?, expert as usize);
                    let bytes = er.len()
                        + expert_range(&up, e)?.len()
                        + expert_range(&down, e)?.len();
                    // SAFETY: the stacks are page-locked, and `self.cpu` keeps them alive and
                    // unmodified for the model's lifetime, well past `done`.
                    unsafe {
                        round_stats::record_staged_bytes(bytes as u64);
                        cs.queue_async(
                            &buf.fence,
                            &[
                                (&memory[0], &g[er]),
                                (&memory[1], &u[expert_range(&up, e)?]),
                                (&memory[2], &d[expert_range(&down, e)?]),
                            ],
                        )?;
                    }
                    // M4: the completion is no longer a `cudaStreamWaitEvent` on candle's stream;
                    // the copy stream publishes the layer's `SYNC_STAGING` point itself (below),
                    // and this layer's gate graph spins on it.
                    COUNTERS.pcie_misses.fetch_add(1, Ordering::Relaxed);
                    COUNTERS.pcie_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
                    slot_of.insert(expert, (Arc::clone(buf), pos));
                }
                slot_of
            }
        };
        let bank_first = SCRATCH_RING_FIRST;
        if slot_of.is_empty() {
            return Ok(());
        }
        // Move every (row, weight) routed to a streamed expert from the CPU list to the hits.
        for (row_misses, row_hits) in misses.iter_mut().zip(hits.iter_mut()) {
            row_misses.retain(|(id, w)| match slot_of.get(id) {
                Some((buf, slot)) => {
                    row_hits.push((Arc::clone(buf), bank_first + slot, *w));
                    false
                }
                None => true,
            });
        }
        Ok(())
    }

    fn decode_forward(
        self: &Arc<Self>,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let rows = batch * seq;
        let x2d = xs.reshape((rows, hidden))?;
        step_probe::mark(&self.slot_device, self.layer);
        // M3/M4: this layer's pass index. Bumped once per pass at layer 0, together with the device
        // epoch, and only once the schedule exists — so the two counters cannot drift. It has to be
        // known before the DMA, because that is what publishes the staging point for this pass.
        #[cfg(feature = "cuda")]
        let pass = {
            static PASS: AtomicU32 = AtomicU32::new(0);
            if self.layer == 0 {
                match schedule() {
                    Some(sched) => {
                        let p = PASS.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Err(e) = sched.bump_epoch(&self.slot_device) {
                            warn!("expert cache: epoch bump failed: {e}");
                        }
                        p
                    }
                    None => 0,
                }
            } else {
                PASS.load(Ordering::Relaxed)
            }
        };
        #[cfg(not(feature = "cuda"))]
        let pass: u32 = 0;
        // Start predicting the next layer's experts now, off the token path; whatever is not
        // resident by the time that layer runs goes to the CPU.
        //
        // R2a: the token thread queues ONE publish kernel on the stream (copy this
        // step's activation into the layer's mapped slot, bump seq) and hands the
        // (last_seq, slot) to the lookahead job. The worker spins for the seq, rebuilds
        // the activation on the CPU and matmuls against the CPU router — the prediction
        // never touches the GPU stream. This removes the W105b 0.30-0.34 ms/layer
        // blocking `to_vec1` D2H from the token thread AND the W99 late-prediction
        // hazard (the publish is stream-ordered right after this layer's last kernel,
        // so the worker can never run before the data exists); `NAIWA_LH_MODE` is
        // obsolete with the channel path.
        if let Some(next) = self.registry.get(self.layer + 1) {
            if next.router.is_some() {
                let queued = std::env::var_os("NAIWA_NO_LOOKAHEAD").is_none();
                if !self.slot_device.is_cuda() {
                    // Non-CUDA slot device: cheap host copy, same job shape.
                    let x = x2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let x = x.flatten_all()?.to_vec1::<f32>()?;
                    if queued {
                        let next = Arc::clone(&next);
                        crate::lookahead::submit(Box::new(move || {
                            let r = x.len() / next.hidden.max(1);
                            let x = Tensor::from_vec(x, (r, next.hidden), &Device::Cpu);
                            if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                debug!("expert cache: layer {} lookahead failed: {e}", next.layer);
                            }
                        }));
                    }
                    lookahead_probe::record(0.0, queued);
                } else if let (true, Some(chan)) = (queued, &self.lookahead_chan) {
                    let n = rows * hidden;
                    let next = Arc::clone(&next);
                    let t0 = std::time::Instant::now();
                    let pub_res = chan.publish(&x2d, n);
                    lookahead_probe::record_publish(t0.elapsed().as_secs_f64() * 1e3);
                    if let Some((sh, last_seq, slot)) = pub_res? {
                        crate::lookahead::submit(Box::new(move || {
                            let t0 = std::time::Instant::now();
                            // The GAP is the scheduler's business, not this
                            // closure's: the lookahead thread only drains its queue
                            // while no expert batch is running, and a batch's
                            // completion wakes it (W106: the router matmul is 2-3
                            // cores and running it beside a batch measured 10
                            // ms/step slower).
                            // Stale check: this layer queued a NEWER prediction
                            // while this job waited — the newer publish carries a
                            // fresher activation (its `wait_publish` job will do
                            // the prefetch), so drop this one. Every layer
                            // resubmits every step; dropping old ones is free.
                            if sh.next_seq.load(Ordering::Relaxed) > last_seq.wrapping_add(1) {
                                lookahead_probe::record_job(t0.elapsed().as_secs_f64() * 1e3);
                                sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                                return;
                            }
                            if let Some(v) = sh.chan.wait_publish(last_seq, n, slot) {
                                let x = Tensor::from_vec(v, (rows, next.hidden), &Device::Cpu);
                                if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                                    debug!(
                                        "expert cache: layer {} lookahead failed: {e}",
                                        next.layer
                                    );
                                }
                            }
                            lookahead_probe::record_job(t0.elapsed().as_secs_f64() * 1e3);
                            // Paid back either way: a timed-out publish's slot must
                            // stop counting against the backpressure budget.
                            // `fetch_max`: team threads finish out of order — a
                            // plain store from an older job would regress the
                            // counter and wedge the backpressure shut.
                            sh.consumed.fetch_max(last_seq + 1, Ordering::Relaxed);
                        }));
                    }
                    lookahead_probe::record(0.0, true);
                } else if queued {
                    // CUDA without the channel (the mapped allocation failed): fall
                    // back to the W105 blocking `to_vec1` on the token thread rather
                    // than silently losing the lookahead.
                    let t0 = std::time::Instant::now();
                    let x_clone = x2d
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let d2h = t0.elapsed().as_secs_f64() * 1e3;
                    let next = Arc::clone(&next);
                    crate::lookahead::submit(Box::new(move || {
                        let x = Tensor::from_vec(x_clone, (rows, next.hidden), &Device::Cpu);
                        if let Err(e) = x.and_then(|x| next.prefetch_predicted(&x, pass)) {
                            debug!("expert cache: layer {} lookahead failed: {e}", next.layer);
                        }
                    }));
                    lookahead_probe::record(d2h, true);
                } else {
                    lookahead_probe::record(0.0, false);
                }
            }
        }

        let n = rows * self.top_k;
        // Step 5 doorbell: lazily acquire (first decode), queue the ring on the same
        // stream the top-k kernels ran on, then spin host-side for the payload — no
        // stream sync, no D2H launch. Falls back to `to_vec2` when unavailable.
        let ring = {
            let mut st = self.state.lock().unwrap();
            if st.doorbell.is_none() {
                let cap = n.max(8 * self.top_k).max(64);
                st.doorbell = crate::doorbell::DoorbellBuf::acquire(&self.slot_device, cap)
                    .ok()
                    .flatten()
                    .map(Arc::new);
            }
            let db = st.doorbell.clone().filter(|db| db.max_n() >= n);
            match db {
                Some(db) => match db.ring(topk_ids, topk_weights, n) {
                    Ok(()) => {
                        let seq = st.doorbell_seq;
                        st.doorbell_seq = seq.wrapping_add(1);
                        Some((db, seq))
                    }
                    Err(_) => None,
                },
                None => None,
            }
        };
        let (ids_rows, weight_rows): (Vec<Vec<u32>>, Vec<Vec<f32>>);
        if let Some((db, seq)) = ring {
            let (_, ids, ws) = db.wait_and_read(seq, n);
            // W106 sanitation: a garbage id here means the doorbell payload was read
            // stale or corrupted; log everything needed to diagnose and fall back to
            // the blocking D2H for this step rather than indexing out of bounds.
            if ids.iter().any(|&id| id as usize >= self.num_experts) {
                let bad: Vec<u32> = ids
                    .iter()
                    .copied()
                    .filter(|id| *id as usize >= self.num_experts)
                    .take(8)
                    .collect();
                candle_core::bail!(
                    "doorbell garbage: layer {} seq {} n {} cap {} rows {} top_k {} ids_shape {:?} bad_ids {:x?}",
                    self.layer,
                    seq,
                    n,
                    db.max_n(),
                    rows,
                    self.top_k,
                    topk_ids.shape(),
                    bad
                );
            }
            ids_rows = ids.chunks(self.top_k).map(<[u32]>::to_vec).collect();
            weight_rows = ws.chunks(self.top_k).map(<[f32]>::to_vec).collect();
        } else {
            ids_rows = topk_ids.reshape((rows, ()))?.to_vec2::<u32>()?;
            weight_rows = topk_weights
                .reshape((rows, ()))?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
        }

        let mut hits: Vec<Vec<Hit>> = vec![Vec::new(); rows];
        let mut misses: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        // M6 P2 flip (`NAIWA_DISPATCH_USE`) is still not armed. Two things it needs first:
        //   1. the lists are shared across layers in one mapping, and the counts are read after a launch
        //      the host does not wait on, so a layer can read the previous layer's larger list (the OOB
        //      that killed the dummy run: len 1, index 210). The list area must be per layer and the count
        //      published with a seq the host waits on - the same protocol the top-k ring uses today.
        //   2. the row count must come from the selection, not from `xs` (already handled here, but it is
        //      only safe once 1 is fixed).
        {
            let mut st = self.state.lock().unwrap();
            st.tick += 1;
            let tick = st.tick;
            for (t, (ids, ws)) in ids_rows.iter().zip(&weight_rows).enumerate() {
                for (&id, &w) in ids.iter().zip(ws) {
                    st.hit_counts[id as usize] += 1;
                    let resident = st
                        .slots
                        .get(&(id as usize))
                        .copied()
                        .and_then(|slot| st.published[slot].clone().map(|buf| (slot, buf)));
                    match resident {
                        Some((slot, buf)) => {
                            st.last_use[slot] = tick;
                            hits[t].push((buf, slot, w));
                        }
                        None => misses[t].push((id, w)),
                    }
                }
            }
        }

        // Hit rate counts only experts actually resident in the cache. Staged ones are injected into
        // `hits` by `stream_pcie` below, so the count has to happen *before* it - otherwise a layer that
        // streams everything reads as a 100% cache hit.
        // Probe `NAIWA_GPU_ONLY_EXPERTS`: once the cache is full, drop every expert that is not resident and
        // never stage or compute one. This is *lossy* (the layer loses those experts' contribution, so the
        // text changes) - it exists to measure what a round costs with no CPU leg and no DMA at all, i.e. the
        // upper bound a perfect cache would give.
        #[cfg(feature = "cuda")]
        if std::env::var_os("NAIWA_GPU_ONLY_EXPERTS").is_some() {
            let full = {
                let st = self.state.lock().unwrap();
                st.resident[..st.usable].iter().all(|s| s.is_some())
            };
            if full {
                for row in misses.iter_mut() {
                    row.clear();
                }
            }
        }
        let n_hits: usize = hits.iter().map(Vec::len).sum();
        COUNTERS.hits.fetch_add(n_hits as u64, Ordering::Relaxed);



        // `--pcie-frac`: DMA the highest-weight misses into the staging ring and run them on the
        // GPU with the hits, leaving the rest on the CPU. Every CUDA call stays on this thread.
        // `stage_misses` only *appends* to `hits`, so the resident prefix is exactly what was there
        // before the call - which is what lets the split below run the residents without the ring.
        #[cfg(feature = "cuda")]
        let resident_prefix: Vec<usize> = hits.iter().map(Vec::len).collect();
        if self.pcie_frac > 0.0 {
            self.stream_pcie(&mut misses, &mut hits, pass)?;
        }

        let n_cpu: usize = misses.iter().map(Vec::len).sum();
        COUNTERS.misses.fetch_add(n_cpu as u64, Ordering::Relaxed);
        // Experts that went to the staging ring: counted apart from both hits and CPU misses.
        #[cfg(feature = "cuda")]
        {
            let staged_n: usize = hits.iter().map(Vec::len).sum::<usize>() - n_hits;
            COUNTERS.staged.fetch_add(staged_n as u64, Ordering::Relaxed);
            round_stats::record_plan(n_hits, staged_n, n_cpu);
        }

        let compute: Option<Box<dyn FnOnce() -> candle_core::Result<Vec<f32>> + Send>> =
            if n_cpu == 0 {
                None
            } else {
                let width = misses.iter().map(Vec::len).max().unwrap_or(0);
                let pad = misses.iter().flatten().next().map_or(0, |&(id, _)| id);
                let mut ids = Vec::with_capacity(rows * width);
                let mut ws = Vec::with_capacity(rows * width);
                for row in &misses {
                    for i in 0..width {
                        let (id, w) = row.get(i).copied().unwrap_or((pad, 0.0));
                        ids.push(id);
                        ws.push(w);
                    }
                }
                let xs_cpu = x2d.unsqueeze(0)?.to_device(&Device::Cpu)?;
                let this = Arc::clone(self);
                Some(Box::new(move || {
                    let x = xs_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    if let Some(ys) =
                        offload_fused_decode(&this.cpu, this.act, &x, &ids, &ws, width)?
                    {
                        return Ok(ys);
                    }
                    let miss_w_t = Tensor::from_vec(ws, (rows, width), &Device::Cpu)?;
                    let miss_ids_t = Tensor::from_vec(ids, (rows, width), &Device::Cpu)?;
                    this.cpu
                        .forward(&xs_cpu, miss_w_t, &miss_ids_t)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                }))
            };

        // M9: there are no segment captures, so no captured CPU-return node can consume this layer's
        // result - the deferred channel always carries it.
        let fast = false;

        // Publication must stay strictly FIFO per point: the payload slot is single-buffered and
        // `ready` must not go backwards. So *every* point for this layer goes through the worker,
        // including the "no misses, publish zeros" case — a synchronous publish from the token
        // thread could overtake a slower job for the previous pass and invert them.
        #[cfg(feature = "cuda")]
        let cpu = if fast {
            let sched = schedule().expect("checked above");
            let len = rows * self.hidden;
            let job: Box<dyn FnOnce() -> candle_core::Result<Vec<f32>> + Send> = match compute {
                Some(f) => f,
                None => Box::new(move || Ok(vec![0f32; len])),
            };
            deferred::start_schedule(
                Arc::clone(sched),
                self.layer as u32,
                pass,
                len,
                job,
            )?;
            None
        } else {
            let chan = self.cpu_chan.get();
            compute
                .map(|f| deferred::start(&self.slot_device, chan, rows, self.hidden, f))
                .transpose()?
        };
        #[cfg(not(feature = "cuda"))]
        #[allow(unused_variables)]
        let cpu = compute
            .map(|f| deferred::start(&self.slot_device, None, rows, self.hidden, f))
            .transpose()?;

        // M4: the copy stream published this layer's `SYNC_STAGING` point on its own stream, after
        // this step's copies. Replaying the layer's gate spins here, inside a captured node, so
        // nothing below can read a staging slot that is still in flight — and the host enqueues no
        // per-expert stream wait to get that guarantee.
        //
        // When warmup could not capture the gate, the *same* node is launched warm: one host call
        // per layer, like the `cudaStreamWaitEvent` it replaces, so the protocol stays one protocol.
        // M5b: same for the preload point, but only when this pass was actually claimed — a layer
        // with no preload this pass has nothing to wait for, and replaying its gate would wait for a
        // publish that nobody owes (`NAIWA_NO_LOOKAHEAD` makes that the normal case).
        #[cfg(feature = "cuda")]
        if self.preload_claimed.load(Ordering::Acquire) == pass {
            self.replay_gate(crate::doorbell::SYNC_PRELOAD)?;
        }
        // S2 (`NAIWA_STAGE_SPLIT`): run the resident experts *before* the staging wait, so the DMA lands
        // under their compute instead of in front of it. The staged ones still need the ring, so they keep
        // the gate. Off by default while it is validated.
        #[cfg(feature = "cuda")]
        let split = std::sync::OnceLock::<bool>::new();
        #[cfg(feature = "cuda")]
        let split = *split.get_or_init(|| {
            std::env::var_os("NAIWA_STAGE_SPLIT").is_some() && self.staging_enabled()
        });
        #[cfg(not(feature = "cuda"))]
        let split = false;

        #[cfg(feature = "cuda")]
        let resident_acc = if split {
            let residents: Vec<Vec<Hit>> = hits
                .iter()
                .zip(&resident_prefix)
                .map(|(row, &n)| row[..n].to_vec())
                .collect();
            let acc = self.run_hits(&residents, &x2d, act);
            let mut fenced = Ok(());
            for (buf, _, _) in residents.iter().flatten() {
                if let Err(e) = buf.fence.record_reads(&self.slot_device) {
                    fenced = fenced.and(Err(e));
                }
            }
            if fenced.is_err() {
                let _ = self.slot_device.synchronize();
            }
            fenced?;
            Some(acc?)
        } else {
            None
        };

        // M4: ... and the staging point, which every pass publishes.
        #[cfg(feature = "cuda")]
        if self.staging_enabled() {
            self.replay_gate(crate::doorbell::SYNC_STAGING)?;
        }

        #[cfg(feature = "cuda")]
        let hits_to_run: Vec<Vec<Hit>> = if split {
            hits.iter()
                .zip(&resident_prefix)
                .map(|(row, &n)| row[n..].to_vec())
                .collect()
        } else {
            Vec::new()
        };
        #[cfg(feature = "cuda")]
        let acc = if let Some((resident_out, resident_bank)) = resident_acc {
            if hits_to_run.iter().all(|row| row.is_empty()) {
                Ok((resident_out, resident_bank))
            } else {
                let (staged_out, staged_bank) = self.run_hits(&hits_to_run, &x2d, act)?;
                Ok(((resident_out + staged_out)?, resident_bank.or(staged_bank)))
            }
        } else {
            self.run_hits(&hits, &x2d, act)
        };
        #[cfg(not(feature = "cuda"))]
        let acc = self.run_hits(&hits, &x2d, act);
        // Kernels reading these slots may be queued even if a later step failed: fence every
        // slot before letting go of it, so an in-place fill cannot overtake them. If a fence
        // cannot be recorded, wait for the device instead.
        let mut fenced = Ok(());
        for (buf, _, _) in hits.iter().flatten() {
            if let Err(e) = buf.fence.record_reads(&self.slot_device) {
                fenced = fenced.and(Err(e));
            }
        }
        if fenced.is_err() {
            let _ = self.slot_device.synchronize();
        }
        drop(hits);
        #[allow(unused_variables)]
        let (gpu_out, capture_bank) = acc?;
        fenced?;


        // Queue the misses' upload even if the hits failed, so the worker's buffer is released.
        let cpu_out = cpu.map(deferred::Pending::upload).transpose();
        let sum = match cpu_out? {
            None => gpu_out,
            Some(c) => (gpu_out + c)?,
        };
        sum.to_dtype(xs.dtype())?.reshape((batch, seq, hidden))
    }
}

/// The model holds `Arc<dyn RoutedExperts>`; this wrapper keeps an `Arc<CachedLayer>` so the
/// layer can hand itself to background fills.
struct CachedExperts(Arc<CachedLayer>);

impl RoutedExperts for CachedExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let layer = &self.0;
        let (batch, seq, hidden_dim) = xs.dims3()?;
        debug_assert_eq!(hidden_dim, layer.hidden);
        let num_tokens = batch * seq;

        // Prefill: every routed expert runs on the CPU. Meanwhile, fill the empty slots of the
        // next layer in the background, off the token path. This layer is filled too, for the
        // first layer of the stack
        // (no predecessor triggers it); a layer already full or filling is a no-op.
        if num_tokens > PREFILL_THRESHOLD {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            if let Some(next) = layer.registry.get(layer.layer + 1) {
                next.spawn_fill();
            }
            layer.spawn_fill();
            if let Some(ps) = &layer.prefill {
                if let Some(out) = ps.forward(layer.layer, xs, &topk_weights, topk_ids)? {
                    return Ok(out);
                }
            }
            let out = layer.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&layer.slot_device);
        }

        // Decode: single token, split GPU hits / CPU misses.
        if let Some(ps) = &layer.prefill {
            ps.note_decode(layer.layer)?;
        }
        layer.decode_forward(xs, &topk_weights, topk_ids, layer.act)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// A stacked `[E, N, K]` Q8_0 tensor whose experts hold distinct values.
    fn stacked(e: usize, n: usize, k: usize) -> QTensor {
        let data: Vec<f32> = (0..e * n * k)
            .map(|i| (i % 97) as f32 * 0.01 - 0.4)
            .collect();
        let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::Q8_0).unwrap()
    }

    #[test]
    fn expert_ranges_tile_the_stack_exactly() {
        let (e, n, k) = (4, 8, 64);
        let qt = stacked(e, n, k);
        let all = qt.data().unwrap().into_owned();
        let stride = all.len() / e;
        let mut rebuilt = Vec::new();
        for i in 0..e {
            let r = expert_range(&qt, i).unwrap();
            assert_eq!(r, i * stride..(i + 1) * stride);
            rebuilt.extend_from_slice(&all[r]);
        }
        assert_eq!(rebuilt, all);
        assert!(expert_range(&qt, e).is_err());
    }

    #[test]
    fn single_expert_qtensor_matches_its_slice_of_the_stack() {
        let (e, n, k) = (3, 4, 32);
        let qt = stacked(e, n, k);
        let one = expert_qtensor_on_device(&qt, 1, &Device::Cpu).unwrap();
        assert_eq!(one.shape().dims(), &[n, k]);
        assert_eq!(one.dtype(), GgmlDType::Q8_0);
        let full = qt.dequantize(&Device::Cpu).unwrap().get(1).unwrap();
        let got = one.dequantize(&Device::Cpu).unwrap();
        let diff = (full - got).unwrap().abs().unwrap().sum_all().unwrap();
        assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn slot_bytes_count_three_projections() {
        let qt = stacked(2, 8, 64);
        let per_expert = qt.data().unwrap().len() / 2;
        assert_eq!(slot_bytes_from_qtensor(&qt), Some(3 * per_expert as u64));
    }

    #[test]
    fn fill_candidate_prefers_hot_unreserved_experts() {
        // expert 2 is hottest but already reserved; 3 and 1 tie, lowest id wins
        let resident = [Some(2), None];
        let hits = [0, 5, 9, 5];
        assert_eq!(fill_candidate(&resident, &hits), Some(1));
        // with no history, fill in id order
        assert_eq!(fill_candidate(&[None, None], &[0, 0, 0]), Some(0));
        // full layer: nothing to do
        assert_eq!(fill_candidate(&[Some(0), Some(1)], &[3, 3, 3]), None);
        // every expert already placed
        assert_eq!(fill_candidate(&[Some(0), Some(1), None], &[1, 1]), None);
    }

    #[test]
    fn registry_does_not_keep_layers_alive() {
        let reg = LayerRegistry::default();
        assert!(reg.get(3).is_none());
        assert!(reg.layers.read().is_empty());
    }

    #[test]
    fn union_by_votes_orders_by_votes_then_rank() {
        // 7 is in all three rankings; 3 in two; 1 and 9 once, 1 at a better rank.
        let got = union_by_votes([vec![3, 7, 9], vec![7, 3], vec![1, 7]]);
        assert_eq!(got, vec![7, 3, 1, 9]);
        assert_eq!(union_by_votes([vec![4, 2]]), vec![4, 2]);
    }

    #[test]
    fn streamed_misses_ranks_by_total_weight_and_dedups() {
        // 3 is missed by both rows (0.4 + 0.1 = 0.5); 1 by one (0.9); 2 by one (0.2).
        let misses = vec![vec![(1u32, 0.9f32), (3, 0.4)], vec![(3, 0.1), (2, 0.2)]];
        // frac 1.0, cap 3: all three distinct experts, highest summed weight first.
        assert_eq!(streamed_misses(&misses, 1.0, 3), vec![1, 3, 2]);
        // frac 0.5 of 3 distinct -> 2 experts.
        assert_eq!(streamed_misses(&misses, 0.5, 3), vec![1, 3]);
        // cap 1 clamps below the fraction.
        assert_eq!(streamed_misses(&misses, 1.0, 1), vec![1]);
        // off, or no scratch to hold anything.
        assert!(streamed_misses(&misses, 0.0, 3).is_empty());
        assert!(streamed_misses(&misses, 1.0, 0).is_empty());
    }

    #[test]
    fn scratch_ring_slots_is_fixed() {
        // Measured: 16 and 64 slots perform identically, so the ring size is a constant (M9).
        assert_eq!(scratch_ring_slots(), 16);
    }

    #[test]
    fn top_k_ids_orders_by_logit_then_index() {
        assert_eq!(top_k_ids(&[0.1, 0.9, 0.5, 0.9], 3), vec![1, 3, 2]);
        assert_eq!(top_k_ids(&[1.0, 2.0], 5), vec![1, 0]);
    }

    #[test]
    fn lru_victim_skips_protected_and_prefers_oldest() {
        // expert -> slot; slot 1 is oldest but holds a protected (predicted) expert
        let slots = HashMap::from([(10, 0), (11, 1), (12, 2)]);
        let last_use = [5, 1, 3];
        assert_eq!(lru_victim(&slots, &last_use, &[11]), Some((12, 2)));
        assert_eq!(lru_victim(&slots, &last_use, &[]), Some((11, 1)));
        assert_eq!(lru_victim(&slots, &last_use, &[10, 11, 12]), None);
    }

    /// A cached layer on the CPU (fresh-upload mode) over small Q8_0 stacks: `e` experts,
    /// `slots` slots, top-2.
    fn cpu_layer(e: usize, slots: usize) -> Arc<CachedLayer> {
        let (hidden, inter) = (64, 32);
        let q = |n: usize, k: usize, seed: f32| {
            let data: Vec<f32> = (0..e * n * k)
                .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
                .collect();
            let t = Tensor::from_vec(data, (e, n, k), &Device::Cpu).unwrap();
            Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap())
        };
        let comm = Arc::new(
            mistralrs_quant::Comm::from_device(mistralrs_quant::Id::new(), &Device::Cpu, 0, 1)
                .unwrap(),
        );
        let cpu = MoEExperts::from_qtensors(
            q(inter, hidden, 0.1),
            q(inter, hidden, 0.7),
            q(hidden, inter, 1.3),
            e,
            2,
            &comm,
            Activation::Silu,
        )
        .unwrap();
        Arc::new(CachedLayer {
            _pinned: Vec::new(),
            cpu,
            slot_device: Device::Cpu,
            layer: 0,
            registry: Arc::default(),
            hidden,
            num_experts: e,
            top_k: 2,
            act: Activation::Silu,
            router: None,
            lookahead_chan: None,
            #[cfg(feature = "cuda")]
            cpu_chan: std::sync::OnceLock::new(),
            state: Mutex::new(CachedLayerState {
                doorbell: None,
                doorbell_seq: 0,
                slots: HashMap::new(),
                resident: vec![None; slots],
                hit_counts: vec![0; e],
                last_use: vec![0; slots],
                tick: 0,
                published: vec![None; slots],
                usable: slots,
            }),
            copy_stream: None,
            slot_bufs: Vec::new(),
            banks: Vec::new(),
            pcie_frac: 0.0,
            lookahead_swaps: 2,
            prefill: None,
            filling: AtomicBool::new(false),
            preload_claimed: AtomicU32::new(0),
        })
    }

    fn wait_for_fill(layer: &CachedLayer) {
        while layer.filling.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }

    #[test]
    fn lent_slots_leave_service_and_come_back_refilled() {
        let layer = cpu_layer(8, 4);
        for e in [5, 6, 7, 4] {
            assert!(layer.admit(e, true).unwrap().is_some());
        }
        assert_eq!(layer.lend(2), 2);
        {
            let st = layer.state.lock().unwrap();
            assert_eq!(st.usable, 2);
            assert_eq!(st.resident, vec![Some(5), Some(6), None, None]);
            assert!(st.published[2].is_none() && st.published[3].is_none());
            assert_eq!(st.slots.len(), 2);
        }
        // no room while lent: lent slots are not admitted into, nor filled
        assert!(layer.admit(1, true).unwrap().is_none());
        assert!(layer.next_fill_candidate().is_none());

        // give back: refilled by routing count (2 and 3 are the hottest non-residents)
        layer.state.lock().unwrap().hit_counts = vec![0, 1, 9, 8, 0, 0, 0, 0];
        layer.give_back().unwrap();
        wait_for_fill(&layer);
        let st = layer.state.lock().unwrap();
        assert_eq!(st.usable, 4);
        assert_eq!(st.resident, vec![Some(5), Some(6), Some(2), Some(3)]);
        assert!(st.published.iter().all(Option::is_some));
    }

    #[test]
    fn decode_split_matches_running_every_expert_on_the_cpu() {
        let layer = cpu_layer(8, 4);
        for e in [0, 2] {
            assert!(layer.admit(e, true).unwrap().is_some());
        }
        let xs: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).cos()).collect();
        let xs = Tensor::from_vec(xs, (1, 1, 64), &Device::Cpu).unwrap();
        // expert 0 is a hit, expert 3 a miss
        let ids = Tensor::new(&[[0u32, 3]], &Device::Cpu).unwrap();
        let ws = Tensor::new(&[[0.6f32, 0.4]], &Device::Cpu).unwrap();
        let got = layer
            .decode_forward(&xs, &ws, &ids, Activation::Silu)
            .unwrap();
        let want = layer.cpu.forward(&xs, ws, &ids).unwrap();
        assert_eq!(got.dims(), want.dims());
        let diff = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Hits run per-expert `QMatMul`s, the reference the fused kernel: equal on aarch64, but
        // on x86 only to Q8_0 activation rounding (~0.005). A swapped weight moves it by ~0.09.
        assert!(diff < 2e-2, "hit/miss split diverges by {diff}");
    }

    #[test]
    fn verify_rows_split_matches_running_every_expert_on_the_cpu() {
        // A verify step: three rows, with hits and misses mixed, rows with no hit (row 2) and no
        // miss (row 1), and expert 3 missed by two rows.
        let layer = cpu_layer(8, 4);
        for e in [0, 2] {
            assert!(layer.admit(e, true).unwrap().is_some());
        }
        let xs: Vec<f32> = (0..3 * 64).map(|i| (i as f32 * 0.11).cos()).collect();
        let xs = Tensor::from_vec(xs, (1, 3, 64), &Device::Cpu).unwrap();
        let ids = Tensor::new(&[[0u32, 3], [2, 0], [3, 5]], &Device::Cpu).unwrap();
        let ws = Tensor::new(&[[0.6f32, 0.4], [0.7, 0.3], [0.5, 0.5]], &Device::Cpu).unwrap();
        let got = layer
            .decode_forward(&xs, &ws, &ids, Activation::Silu)
            .unwrap();
        let want = layer.cpu.forward(&xs, ws, &ids).unwrap();
        assert_eq!(got.dims(), want.dims());
        let diff = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 2e-2, "verify-row split diverges by {diff}");
    }
}
