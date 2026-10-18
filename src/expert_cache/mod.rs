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

mod decode;
mod expert_bytes;
mod layer;
mod probe;
mod provider;
mod registry;
mod staging;
#[cfg(test)]
mod tests;

pub(crate) use probe::record_cpu_compute;
pub use provider::GpuCacheMoeProvider;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;
use parking_lot::RwLock;

use crate::copy_stream::{CopyStream, ReadFence, SlotMemory};
#[cfg(feature = "cuda")]
use crate::deferred;
use crate::host_memory::PinnedStack;
use crate::prefill_stream::PrefillStream;

use layer::LookaheadChannel;

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
/// Fixed: 16 measured the same as 64, and a bigger ring only costs VRAM.
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

/// The v2 M1/M2 fixed-grid batch bound, matching the lookahead channel's policy
/// (`LookaheadChannel::new`): it must cover the largest decode batch, with 16 rows of headroom
/// over any current window (np4 verify = 5, observed verify = 7). Calls above it use the old
/// host-geometry path — the allowed row-bucket fallback.
#[cfg(feature = "cuda")]
const CAPTURE_MAX_BATCH: usize = 16;

#[cfg(feature = "cuda")]
const WAITS_PER_LAYER: u32 = 3;

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
    /// `--no-lookahead`: this layer does not predict for the next one.
    lookahead: bool,
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
