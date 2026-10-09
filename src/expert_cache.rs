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
//!   via `MoEExperts::forward` on the CPU with only the miss ids (the fork's `forward_gather`
//!   reads top-k from `topk_ids`), both at once; router weights are applied once per side. The
//!   misses run on a scoped thread over host copies; every CUDA call stays on the calling thread.
//! - Misses are never loaded on the token path. Instead, each layer submits its activation to a
//!   lookahead thread that runs the next layer's router on it and loads the predicted experts:
//!   into empty slots freely, over least-recently-used residents at most `MAX_LOOKAHEAD_SWAPS`
//!   times. Whatever is not resident when the next layer runs goes to the CPU.
//! - A token captures its hits' weights (`Arc`s) under the layer lock, so a slot replaced later
//!   never changes what an in-flight token computes with.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Weak};

use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::copy_stream::{self, CopyStream, CopyStreams, ReadFence, SlotMemory};
use crate::host_memory::{self, PinnedStack};
use crate::offload::{attach_prefill, CpuExperts};
use crate::prefill_stream::{ExpertMemoryLender, PrefillStream, PrefillStreams};

// Number of prefill tokens above which we skip the cache entirely.
// A decode step is always 1 token, so PREFILL_THRESHOLD = 1 is exact.
const PREFILL_THRESHOLD: usize = 1;

/// Upper bound on resident experts the decode lookahead replaces per layer per token. Each swap
/// copies one expert's three projections host-to-device; without a cap a poor prediction would
/// turn every token into a full re-upload of every layer's top-k.
const MAX_LOOKAHEAD_SWAPS: usize = 2;

/// With `--prefill-stream`, each layer keeps its last slots in a separate bank of about its
/// share of the loan: two layers of experts, spread over the estimated 48 layers, plus a
/// quarter for layers with smaller slots than the largest one. The loan lends whole banks.
fn loan_bank_slots(num_experts: usize) -> usize {
    (2 * num_experts * 5).div_ceil(4 * 48)
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

/// Decode-path counters over every cached layer of the process (one model per process).
struct Counters {
    installed: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    lookahead_loads: AtomicU64,
    swaps: AtomicU64,
    lookahead_dropped: AtomicU64,
}

static COUNTERS: Counters = Counters {
    installed: AtomicBool::new(false),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    lookahead_loads: AtomicU64::new(0),
    swaps: AtomicU64::new(0),
    lookahead_dropped: AtomicU64::new(0),
};

/// Expert-cache counters since the last call to `take_stats`.
pub struct CacheStats {
    /// Routed experts of decode steps served from a slot.
    pub hits: u64,
    /// Routed experts of decode steps run on the CPU.
    pub misses: u64,
    /// Experts the lookahead loaded into empty slots.
    pub lookahead_loads: u64,
    /// Experts the lookahead loaded over a least-recently-used resident.
    pub swaps: u64,
    /// Lookahead requests dropped because the worker was busy.
    pub lookahead_dropped: u64,
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
        lookahead_loads: take(&c.lookahead_loads),
        swaps: take(&c.swaps),
        lookahead_dropped: take(&c.lookahead_dropped),
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
}

impl GpuCacheMoeProvider {
    pub fn new(cache_gb: f64, prefill: Option<PrefillStreams>) -> Self {
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
        }
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
        // Never cache MTP-head experts or CPU-device layers.
        if ctx.site.is_mtp || ctx.layer_device.is_cpu() {
            return Ok(None);
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

        // The same tensor the model loads as this block's router (`gate`); the decode lookahead
        // runs it on the CPU. Without it the layer is still cached, just never predicted.
        let router = match load_router(&ctx, num_experts, hidden) {
            Ok(w) => Some(w),
            Err(e) => {
                debug!("expert cache: layer {layer_idx} has no router for lookahead: {e}");
                None
            }
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
            state: Mutex::new(CachedLayerState {
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
            prefill,
            filling: AtomicBool::new(false),
        });
        self.registry.insert(layer_idx, &cached);
        Ok(Some(Arc::new(CachedExperts(cached))))
    }
}

/// Router weights `[num_experts, hidden]` as F32 on the CPU, read the way the model reads them.
fn load_router(
    ctx: &ExpertLoadContext<'_>,
    num_experts: usize,
    hidden: usize,
) -> candle_core::Result<Tensor> {
    ctx.vb
        .pp("gate")
        .get((num_experts, hidden), "weight")?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)
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

/// One decode lookahead request: predict `layer`'s experts from the previous layer's input.
struct LookaheadJob {
    layer: Arc<CachedLayer>,
    x: Vec<f32>,
}

/// A single background thread that runs the next layer's router on the current activation and
/// loads the predicted experts. The queue is one deep and a submission made while the worker
/// is busy is dropped, so decode never waits on it.
#[derive(Default)]
struct Lookahead {
    tx: Mutex<Option<SyncSender<LookaheadJob>>>,
}

impl Lookahead {
    fn submit(&self, job: LookaheadJob) {
        let mut tx = self.tx.lock().unwrap();
        let sender = tx.get_or_insert_with(|| {
            let (tx, rx) = sync_channel::<LookaheadJob>(1);
            std::thread::spawn(move || {
                for job in rx {
                    if let Err(e) = job.layer.prefetch_predicted(&job.x) {
                        debug!(
                            "expert cache: layer {} lookahead failed: {e}",
                            job.layer.layer
                        );
                    }
                }
            });
            tx
        });
        match sender.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                COUNTERS.lookahead_dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => *tx = None,
        }
    }
}

/// Every cached layer of one model, by decoder-layer index, so layer `l` can start filling
/// layer `l + 1`'s slots. Weak references: the model owns the layers, not the registry.
#[derive(Default)]
struct LayerRegistry {
    layers: RwLock<Vec<Weak<CachedLayer>>>,
    lookahead: Lookahead,
}

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

/// The fused MoE decode kernel's activation id, if it has one for `act`.
#[cfg(feature = "cuda")]
fn fused_act(act: Activation) -> Option<i32> {
    match act {
        Activation::Silu | Activation::Swish => Some(mistralrs_quant::ACT_SILU),
        Activation::GeluPytorchTanh => Some(mistralrs_quant::ACT_GELU_PYTORCH_TANH),
        _ => None,
    }
}

/// `sum_i weights[i] * expert(bank, ids[i])(x2d)` for one token, `[1, hidden]` F32, by
/// mistral.rs's fused MoE decode (quantize, gate+up+act, quantize, down+weighted sum).
#[cfg(feature = "cuda")]
fn fused_decode(
    bank: &Bank,
    x2d: &Tensor,
    ids: Vec<u32>,
    weights: Vec<f32>,
    act_type: i32,
) -> candle_core::Result<Tensor> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    let dev = x2d.device().as_cuda_device()?;
    let topk = ids.len();
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
            1,
            topk,
            act_type,
            dev,
        )
    }
}

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
    state: Mutex<CachedLayerState>,
    /// Uploads fills alongside compute; `None` off CUDA, where fills go through candle.
    copy_stream: Option<Arc<CopyStream>>,
    /// Copy-stream mode: every slot's permanent buffer (empty otherwise).
    slot_bufs: Vec<SlotEntry>,
    /// Copy-stream mode: the banks as `(first slot, len)`, in slot order. Lent whole.
    banks: Vec<(usize, usize)>,
    /// Runs long prefill chunks' experts on the GPU (`--prefill-stream`).
    prefill: Option<Arc<PrefillStream>>,
    /// A background fill of this layer is running; at most one at a time.
    filling: AtomicBool,
}

impl CachedLayer {
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
    fn admit(&self, expert_id: usize) -> candle_core::Result<Option<usize>> {
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
        self.load_reserved(expert_id, slot)
    }

    /// Replace the least-recently-used resident outside `protect` with `expert_id`. The victim
    /// is unpublished and the slot reserved in one critical section. Tokens already holding the
    /// victim keep computing with it: a fresh upload leaves the old buffer alone, and an
    /// in-place one waits for them (`upload_in_place`).
    fn replace(&self, expert_id: usize, protect: &[usize]) -> candle_core::Result<Option<usize>> {
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
        self.load_reserved(expert_id, slot)
    }

    /// Copy `expert_id` into `slot`, which the caller has reserved, then publish it. A failed
    /// copy frees the slot.
    fn load_reserved(&self, expert_id: usize, slot: usize) -> candle_core::Result<Option<usize>> {
        let loaded = match &self.copy_stream {
            Some(cs) => self.upload_in_place(cs, expert_id, slot),
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
        cs.upload(
            &buf.fence,
            &[
                (&memory[0], &g[expert_range(&gate, expert_id)?]),
                (&memory[1], &u[expert_range(&up, expert_id)?]),
                (&memory[2], &d[expert_range(&down, expert_id)?]),
            ],
        )?;
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

    /// This layer's top-k experts for activation `x` (`[hidden]`), by its own router.
    fn predict(&self, x: &[f32]) -> candle_core::Result<Vec<usize>> {
        let Some(router) = &self.router else {
            return Ok(Vec::new());
        };
        let x = Tensor::from_slice(x, (self.hidden, 1), &Device::Cpu)?;
        let logits = router.matmul(&x)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(top_k_ids(&logits, self.top_k))
    }

    /// Load the experts predicted for `x`: into empty slots freely, and over LRU residents at
    /// most `MAX_LOOKAHEAD_SWAPS` times. A wrong prediction only costs a slot.
    fn prefetch_predicted(&self, x: &[f32]) -> candle_core::Result<()> {
        let predicted = self.predict(x)?;
        let mut swaps = 0;
        for &e in &predicted {
            if e >= self.num_experts {
                continue;
            }
            if self.admit(e)?.is_some() {
                COUNTERS.lookahead_loads.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if swaps < MAX_LOOKAHEAD_SWAPS && self.replace(e, &predicted)?.is_some() {
                swaps += 1;
                COUNTERS.swaps.fetch_add(1, Ordering::Relaxed);
            }
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
                match this.admit(e) {
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

    /// The weighted sum of the hits' outputs for one token, `[1, hidden]` F32 on the slot
    /// device. In copy-stream mode each bank's hits run as one fused MoE decode (four kernels);
    /// otherwise every expert runs on its own.
    fn run_hits(
        &self,
        hits: &[(Arc<SlotBuf>, usize, f32)],
        x2d: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
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
            hits.iter().all(|(buf, _, _)| buf.bank.is_some()),
        ) {
            let mut groups: Vec<(&Arc<Bank>, Vec<u32>, Vec<f32>)> = Vec::new();
            for (buf, slot, w) in hits {
                let bank = buf.bank.as_ref().expect("checked above");
                let local = (slot - bank.first) as u32;
                match groups.iter_mut().find(|(b, _, _)| Arc::ptr_eq(b, bank)) {
                    Some((_, ids, ws)) => {
                        ids.push(local);
                        ws.push(*w);
                    }
                    None => groups.push((bank, vec![local], vec![*w])),
                }
            }
            for (bank, ids, ws) in groups {
                add(fused_decode(bank, x2d, ids, ws, act_type)?)?;
            }
        } else {
            for (buf, _, w) in hits {
                let Some(weights) = &buf.weights else {
                    candle_core::bail!("expert cache: slot has no weights to run");
                };
                let y = Self::run_expert(weights, x2d, act)?;
                add((y.to_dtype(DType::F32)? * *w as f64)?)?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        for (buf, _, w) in hits {
            let Some(weights) = &buf.weights else {
                candle_core::bail!("expert cache: slot has no weights to run");
            };
            let y = Self::run_expert(weights, x2d, act)?;
            add((y.to_dtype(DType::F32)? * *w as f64)?)?;
        }
        match acc {
            Some(acc) => Ok(acc),
            None => Tensor::zeros((1, self.hidden), DType::F32, &self.slot_device),
        }
    }

    /// Run one token through a cached expert. `x2d: [1, hidden]` on the slot device.
    fn run_expert(w: &SlotWeights, x2d: &Tensor, act: Activation) -> candle_core::Result<Tensor> {
        let g = QMatMul::QTensor(w.gate.clone()).forward(x2d)?;
        let u = QMatMul::QTensor(w.up.clone()).forward(x2d)?;
        let h = (act.forward(&g)? * u)?;
        QMatMul::QTensor(w.down.clone()).forward(&h)
    }

    /// Decode path: split top-k into GPU hits and CPU misses, run both concurrently, sum.
    /// Misses are never loaded on the token path — they run on the CPU.
    ///
    /// Every CUDA call stays on the calling thread: the hits' kernels are only enqueued here, so
    /// they still run while a scoped thread computes the misses on host tensors. Launching the
    /// hits from a second thread while this one copied the activation to the host produced NaN
    /// logits on CUDA once the lookahead was swapping slots.
    fn decode_forward(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        act: Activation,
    ) -> candle_core::Result<Tensor> {
        // Start predicting the next layer's experts now, off the token path; whatever is not
        // resident by the time that layer runs goes to the CPU.
        if let Some(next) = self.registry.get(self.layer + 1) {
            if next.router.is_some() {
                let x = xs.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                self.registry
                    .lookahead
                    .submit(LookaheadJob { layer: next, x });
            }
        }

        let ids_vec = topk_ids.i(0)?.to_vec1::<u32>()?;
        let weights_vec = topk_weights.i(0)?.to_vec1::<f32>()?;

        let mut hits: Vec<(Arc<SlotBuf>, usize, f32)> = Vec::new();
        let mut miss_ids: Vec<u32> = Vec::new();
        let mut miss_weights: Vec<f32> = Vec::new();
        {
            let mut st = self.state.lock().unwrap();
            st.tick += 1;
            let tick = st.tick;
            for (&id, &w) in ids_vec.iter().zip(&weights_vec) {
                st.hit_counts[id as usize] += 1;
                let resident = st
                    .slots
                    .get(&(id as usize))
                    .copied()
                    .and_then(|slot| st.published[slot].clone().map(|buf| (slot, buf)));
                match resident {
                    Some((slot, buf)) => {
                        st.last_use[slot] = tick;
                        hits.push((buf, slot, w));
                    }
                    None => {
                        miss_ids.push(id);
                        miss_weights.push(w);
                    }
                }
            }
        }

        COUNTERS
            .hits
            .fetch_add(hits.len() as u64, Ordering::Relaxed);
        COUNTERS
            .misses
            .fetch_add(miss_ids.len() as u64, Ordering::Relaxed);
        let x_gpu = xs.i((0, 0))?.unsqueeze(0)?; // [1, hidden]

        // Host inputs for the misses, copied before any thread starts.
        let miss_inputs = if miss_ids.is_empty() {
            None
        } else {
            Some((
                xs.to_device(&Device::Cpu)?,
                Tensor::new(miss_weights, &Device::Cpu)?.unsqueeze(0)?,
                Tensor::new(miss_ids, &Device::Cpu)?.unsqueeze(0)?,
            ))
        };

        let (gpu_result, cpu_result) = std::thread::scope(|s| {
            let cpu = miss_inputs.map(|(xs_cpu, miss_w_t, miss_ids_t)| {
                s.spawn(move || -> candle_core::Result<Tensor> {
                    let out = self.cpu.forward(&xs_cpu, miss_w_t, &miss_ids_t)?;
                    out.squeeze(0) // [1, 1, hidden] -> [1, hidden]
                })
            });

            let acc = self.run_hits(&hits, &x_gpu, act);
            // Kernels reading these slots may be queued even if a later step failed: fence
            // every slot before letting go of it, so an in-place fill cannot overtake them.
            // If a fence cannot be recorded, wait for the device instead.
            let mut fenced = Ok(());
            for (buf, _, _) in &hits {
                if let Err(e) = buf.fence.record_reads(&self.slot_device) {
                    fenced = fenced.and(Err(e));
                }
            }
            if fenced.is_err() {
                let _ = self.slot_device.synchronize();
            }
            drop(hits);
            let gpu = acc.and_then(|acc| fenced.map(|()| acc));

            let cpu = cpu.map(|h| h.join().unwrap()).transpose();
            (gpu, cpu)
        });

        let gpu_out = gpu_result?;
        let sum = match cpu_result? {
            None => gpu_out.to_dtype(xs.dtype())?,
            Some(c) => {
                let c_gpu = c.to_device(&self.slot_device)?.to_dtype(DType::F32)?;
                (gpu_out + c_gpu)?.to_dtype(xs.dtype())?
            }
        };
        sum.unsqueeze(0) // [1, 1, hidden]
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
            state: Mutex::new(CachedLayerState {
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
            prefill: None,
            filling: AtomicBool::new(false),
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
            assert!(layer.admit(e).unwrap().is_some());
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
        assert!(layer.admit(1).unwrap().is_none());
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
            assert!(layer.admit(e).unwrap().is_some());
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
}
