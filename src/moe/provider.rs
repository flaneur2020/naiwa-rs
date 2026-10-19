//! The `RoutedExpertsProvider` that builds one cached layer per MoE block. The provider owns the
//! model-wide layer index and staging rings; each cached layer borrows both.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use tracing::{debug, info, warn};

use super::expert::CpuExperts;
use crate::copy_stream::{CopyDone, CopyStreams};
use crate::host_memory::{self, pin_expert_stacks, PinnedStack};
use crate::prefill_stream::{ExpertMemoryLender, PrefillStream, PrefillStreams};
use crate::stats;
use mistralrs_core::moe_offload::{ExpertLoadContext, RoutedExperts, RoutedExpertsProvider};
use mistralrs_core::MoEExperts;

use super::{
    alloc_bank, loan_bank_slots, slot_bytes_from_qtensor, CachedExperts, CachedLayer, LayerState,
    Layers, LookaheadChannel, Slot, StagingRings, MAX_DECODE_ROWS,
};

/// With `--prefill-stream`, pin `experts`' stacks and attach the layer to its device's prefill
/// stream. The pinned registrations must live as long as the layer.
fn attach_prefill(
    prefill: &Option<PrefillStreams>,
    ctx: &ExpertLoadContext<'_>,
    experts: &MoEExperts,
    copy_streams: &CopyStreams,
) -> candle_core::Result<(Vec<PinnedStack>, Option<Arc<PrefillStream>>)> {
    let Some(streams) = prefill else {
        return Ok((Vec::new(), None));
    };
    // Prefill uploads read straight from the stacks, asynchronously: they must be pinned.
    let (pinned, failed) = pin_expert_stacks(experts, ctx.layer_device);
    let stream = streams.attach(ctx, experts, failed == 0, copy_streams)?;
    Ok((pinned, stream))
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
    layers: Arc<Layers>,
    staging: Arc<StagingRings>,
    copy_streams: CopyStreams,
    /// `--prefill-stream`: run long prefill chunks' experts on the GPU.
    prefill: Option<PrefillStreams>,
    /// `--pcie-frac`: share of decode misses to DMA into staging slots and run on the GPU.
    pcie_frac: f32,
    /// `--lookahead-swaps`: how many predicted experts may displace residents per row and pass.
    lookahead_swaps: usize,
    /// `--no-lookahead`: the provider's prediction is off (tests only).
    lookahead: bool,
}

impl GpuCacheMoeProvider {
    pub fn new(
        cache_gb: f64,
        prefill: Option<PrefillStreams>,
        pcie_frac: f32,
        lookahead_swaps: usize,
        lookahead: bool,
    ) -> Self {
        stats::record_cache_installed();
        let total_bytes = (cache_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        let layers: Arc<Layers> = Arc::default();
        let staging: Arc<StagingRings> = Arc::default();
        // The prefill stream's layer buffers are carved out of this cache while a prompt runs.
        let lender: Arc<dyn ExpertMemoryLender> = layers.clone();
        let prefill = prefill.map(|p| p.with_lender(lender));
        Self {
            cache_gb,
            inner: Mutex::new(ProviderInner {
                remaining_bytes: total_bytes,
                bytes_per_layer: None,
            }),
            layers,
            staging,
            copy_streams: CopyStreams::default(),
            prefill,
            pcie_frac,
            lookahead_swaps,
            lookahead,
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
        let layers = self.layers.all();
        // Every mapped buffer is sized and allocated here, once: the top-k doorbell per layer (for
        // the largest decode window) and the engine-wide CPU-result ring. A staging ring is only
        // built with `--pcie-frac` (that is where its ~215 MiB belongs).
        #[cfg(feature = "cuda")]
        if let Some(first) = layers.first() {
            for layer in &layers {
                layer.warmup_doorbell()?;
            }
            crate::executor::warmup_result_channel(
                first.slot_device(),
                MAX_DECODE_ROWS * first.hidden(),
            )?;
        }
        #[allow(unused_mut)]
        let mut captures = 0usize;
        // M9: no segment captures. Measured before the deletion (graphs on/off): 43.5 vs 44.1 tok/s,
        // graphs off 44.1 - identical, so the eager segment is the one fixed configuration.
        let mut rings = 0usize;
        if self.pcie_frac > 0.0 {
            for layer in &layers {
                layer.warmup_staging_ring()?;
            }
            rings = self.staging.creations();
        }
        Ok((captures, rings))
    }

    /// v2 M1 criterion ①: rings created so far. Must not change once warmup has run and the
    /// decode has started.
    pub fn ring_creations(&self) -> usize {
        self.staging.creations()
    }
}

// ---------------------------------------------------------------------------
// RoutedExpertsProvider impl
// ---------------------------------------------------------------------------

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
        let slots: Vec<Slot> = (0..num_slots).map(|_| Slot::empty()).collect();
        if let (Some(cs), Some(stacks)) = (&copy_stream, cpu_experts.expert_qtensors()) {
            for &(first, len) in &banks {
                for (i, buf) in alloc_bank(cs, &stacks, first, len, &slot_device)?
                    .into_iter()
                    .enumerate()
                {
                    *slots[first + i].storage.lock() = Some(buf);
                }
            }
            cs.after_device_work()?;
        }
        if slots.iter().all(|s| s.storage.lock().is_none()) {
            copy_stream = None;
        }

        // `--pcie-frac`: the model-wide staging ring, where a decode step DMAs the misses it
        // will run on the GPU, is NOT built here. It first materializes in `stream_pcie` of the
        // first decode step: during model load the cache banks and weight shards fragment
        // device memory, and allocating the ring's contiguous bank in that window failed on
        // some load orders (W105: 33 MiB at 11.6 GiB free, OOM in 2 of 3 runs). Once load has
        // finished the allocator has large free blocks left, where it fits reliably.
        // The ring's `first` sits past every resident slot, and it is kept out of
        // slot storage/banks, so `lend`/`give_back`/`admit`/`replace` never see it; `run_hits`
        // picks it up only because the injected hits carry its `Arc<Bank>`.

        // Both completion events are created here (the decode path must not allocate a sync
        // object). `staging_done: None` is what disables `--pcie-frac` for a layer whose copies
        // nobody would wait for; `preload: None` means the layer's fills go through candle's
        // stream, where they are already ordered.
        let staging_done = match (&copy_stream, self.pcie_frac > 0.0) {
            (Some(cs), true) => Some(Arc::new(CopyDone::new(cs)?)),
            _ => None,
        };
        let preload = match &copy_stream {
            Some(cs) => Some(crate::doorbell::PassCell::new(Arc::new(CopyDone::new(cs)?))),
            None => None,
        };

        let cached = Arc::new(CachedLayer {
            _pinned: pinned,
            cpu: cpu_experts,
            slot_device,
            layer: layer_idx,
            layers: Arc::clone(&self.layers),
            staging: Arc::clone(&self.staging),
            hidden,
            num_experts,
            top_k: ctx.config.num_experts_per_tok,
            act: ctx.act,
            router,
            lookahead_chan,
            #[cfg(feature = "cuda")]
            state: Mutex::new(LayerState {
                doorbell: None,
                slots,
                hit_counts: vec![0u64; num_experts],
                tick: 0,
                usable: num_slots,
            }),
            copy_stream,
            banks,
            pcie_frac: self.pcie_frac,
            lookahead_swaps: self.lookahead_swaps,
            lookahead: self.lookahead,
            prefill,
            filling: AtomicBool::new(false),
            preload,
            staging_done,
        });
        self.layers.insert(layer_idx, &cached);
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
