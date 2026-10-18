//! `--cpu-moe`: routed experts live in host memory and run on the CPU, while everything else
//! (attention, DeltaNet, routers, shared experts, the MTP head, the KV cache) stays on the GPU.
//!
//! With `--expert-cache-gb <N>`, the most-used experts are additionally cached in VRAM for decode
//! (`expert_cache`). With `--prefill-stream`, long prefill chunks run the experts on the GPU,
//! streaming each layer's experts in while the previous layer computes (`prefill_stream`).

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::{
    install_routed_experts_provider, ExpertLoadContext, RoutedExperts, RoutedExpertsProvider,
};
use mistralrs_core::MoEExperts;
use tracing::{debug, info};

use crate::copy_stream::CopyStreams;
use crate::expert_cache::GpuCacheMoeProvider;
use crate::host_memory::{advise_expert_stacks, pin_expert_stacks, PinnedStack};
use crate::prefill_stream::{PrefillStream, PrefillStreams};

/// Device-layer map that keeps every decoder layer on the main device; mistral.rs clamps the
/// count to the model's layer count. Taken-over experts are not counted, so auto mapping would
/// spill whole layers to the CPU.
pub const ALL_LAYERS_ON_DEVICE: usize = usize::MAX;

/// `prefill_stream`: `Some(min_tokens)` with `--prefill-stream`.
/// `pcie_frac`: share of decode misses to DMA to the GPU (`--pcie-frac`), 0.0 when off.
pub fn install_cpu_moe(
    expert_cache_gb: Option<f64>,
    prefill_stream: Option<usize>,
    pcie_frac: f32,
    lookahead_swaps: usize,
    lookahead: bool,
) -> Result<()> {
    let prefill = prefill_stream.map(PrefillStreams::new);
    if let Some(min_tokens) = prefill_stream {
        info!(
            "--prefill-stream: prefill chunks of {min_tokens} tokens or more run their experts on the GPU"
        );
    }
    if let Some(gb) = expert_cache_gb {
        let provider = Arc::new(GpuCacheMoeProvider::new(gb, prefill, pcie_frac, lookahead_swaps, lookahead));
        let _ = GPU_CACHE_PROVIDER.set(Arc::clone(&provider));
        install_routed_experts_provider(provider)?;
        info!(
            "--cpu-moe + expert cache: routed experts in host memory; {:.1} GB GPU cache",
            gb
        );
    } else {
        install_routed_experts_provider(Arc::new(CpuMoeProvider {
            prefill,
            copy_streams: CopyStreams::default(),
        }))?;
        info!("--cpu-moe: routed experts load into host memory and run on the CPU");
    }
    Ok(())
}

/// The installed GPU expert-cache provider, kept so the engine can warm it up after the
/// model finishes loading (`warmup_rings`). `None` when `--expert-cache-gb` was not used.
static GPU_CACHE_PROVIDER: OnceLock<Arc<GpuCacheMoeProvider>> = OnceLock::new();

/// v2 M1/M2: build every fixed-address structure after load and before the first decode — each
/// resident bank's fixed-grid capture path, and (with `--pcie-frac`) the quant shapes' staging
/// rings. Returns `(captures, rings)`; `(0, 0)` when no GPU expert cache is installed.
pub fn warmup() -> Result<(usize, usize)> {
    match GPU_CACHE_PROVIDER.get() {
        Some(provider) => Ok(provider.warmup()?),
        None => Ok((0, 0)),
    }
}

/// v2 M1 criterion ①: staging rings created so far. Must not change after warmup — a further
/// increment means a ring materialized on the decode path, i.e. an address that was not fixed
/// up front (which a capture cannot tolerate).
pub fn ring_creations() -> usize {
    GPU_CACHE_PROVIDER.get().map_or(0, |p| p.ring_creations())
}

struct CpuMoeProvider {
    prefill: Option<PrefillStreams>,
    copy_streams: CopyStreams,
}

impl RoutedExpertsProvider for CpuMoeProvider {
    fn build(
        &self,
        ctx: ExpertLoadContext<'_>,
    ) -> candle_core::Result<Option<Arc<dyn RoutedExperts>>> {
        if ctx.layer_device.is_cpu() {
            return Ok(None);
        }
        debug!(
            "Routed experts of {} layer {} load on the CPU",
            if ctx.site.is_mtp { "MTP" } else { "decoder" },
            ctx.site.layer
        );
        let experts = ctx.load_builtin(Device::Cpu)?;
        let huge_bytes = advise_expert_stacks(&experts)?;
        debug!("  {} MiB of its stacks on huge pages", huge_bytes >> 20);
        let (pinned, prefill) = attach_prefill(&self.prefill, &ctx, &experts, &self.copy_streams)?;
        Ok(Some(Arc::new(CpuExperts::new(
            experts,
            ctx.act,
            ctx.layer_device.clone(),
            ctx.site.layer,
            prefill,
            pinned,
        ))))
    }
}

/// With `--prefill-stream`, pin `experts`' stacks and attach the layer to its device's prefill
/// stream. The pinned registrations must live as long as the layer.
pub(crate) fn attach_prefill(
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

/// Calls of at most this many tokens are decode-sized (one token, or a speculative verify step)
/// and take the fused CPU path; longer ones are prefill chunks, where the generic path's routed
/// gemvs already amortize their per-call cost.
pub(crate) const FUSED_MAX_TOKENS: usize = 16;

/// A decode step's routed experts on the CPU in two parallel passes
/// (`mistralrs_quant::fused_moe_decode`): gate+up+act per distinct (expert, row), then down with
/// the router-weighted sum. At decode sizes the generic `MoEExperts::forward` - three routed
/// gemvs with candle ops between them - spends more on its per-call structure than on the dots.
///
/// `xs` is `rows x hidden` F32; `ids` and `weights` are row-major `[rows, topk]`, weight-0 slots
/// skipped. `None` when the generic path must run instead: an activation other than SiLU, experts
/// that are not GGUF stacks, a dtype or shape the fused path does not serve, or
pub(crate) fn fused_decode(
    experts: &MoEExperts,
    act: Activation,
    xs: &[f32],
    ids: &[u32],
    weights: &[f32],
    topk: usize,
) -> candle_core::Result<Option<Vec<f32>>> {
    static GENERIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // M9: one fixed behaviour - the fused path for the activations it supports, the generic one otherwise.
    let _ = &GENERIC;
    if !matches!(act, Activation::Silu | Activation::Swish) {
        return Ok(None);
    }
    let Some((gate, up, down)) = experts.expert_qtensors() else {
        return Ok(None);
    };
    let t0 = std::time::Instant::now();
    let ys = mistralrs_quant::fused_moe_decode(&gate, &up, &down, xs, ids, weights, topk);
    fused_probe::record(t0, ids, weights, topk);
    ys
}

/// `NAIWA_FUSED_DBG=1`: wall time of the fused CPU decode per call, per (expert, row) pair and per
/// distinct expert, and the idle gap between calls, printed every `WINDOW` calls (10 layer passes).
mod fused_probe {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    const WINDOW: u64 = 480;

    #[derive(Default)]
    struct Acc {
        calls: u64,
        busy_ns: u64,
        gap_ns: u64,
        pairs: u64,
        experts: u64,
        rows: u64,
        last_end: Option<Instant>,
    }

    static ON: OnceLock<bool> = OnceLock::new();
    static ACC: Mutex<Acc> = Mutex::new(Acc {
        calls: 0,
        busy_ns: 0,
        gap_ns: 0,
        pairs: 0,
        experts: 0,
        rows: 0,
        last_end: None,
    });
    static HEADER: AtomicBool = AtomicBool::new(false);

    pub(super) fn record(t0: Instant, ids: &[u32], weights: &[f32], topk: usize) {
        if !*ON.get_or_init(|| std::env::var_os("NAIWA_FUSED_DBG").is_some()) {
            return;
        }
        let end = Instant::now();
        let mut pairs: Vec<(u32, usize)> = ids
            .iter()
            .zip(weights)
            .enumerate()
            .filter(|(_, (_, &w))| w != 0.0)
            .map(|(i, (&e, _))| (e, i / topk.max(1)))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let mut experts: Vec<u32> = pairs.iter().map(|p| p.0).collect();
        experts.dedup();
        let mut a = ACC.lock().unwrap();
        if let Some(last) = a.last_end {
            // Gaps over 50 ms are between steps (sampling, the bench's own work), not layers.
            let gap = t0.saturating_duration_since(last).as_nanos() as u64;
            if gap < 50_000_000 {
                a.gap_ns += gap;
            }
        }
        a.last_end = Some(end);
        a.calls += 1;
        a.busy_ns += (end - t0).as_nanos() as u64;
        a.pairs += pairs.len() as u64;
        a.experts += experts.len() as u64;
        a.rows += (ids.len() / topk.max(1)) as u64;
        if a.calls == WINDOW {
            if !HEADER.swap(true, Ordering::Relaxed) {
                eprintln!("FUSED_DBG calls rows/call pairs/call experts/call us/call us/pair us/expert gap_us/call");
            }
            let c = a.calls as f64;
            let us = a.busy_ns as f64 / 1e3;
            eprintln!(
                "FUSED_DBG {} {:.2} {:.2} {:.2} {:.1} {:.1} {:.1} {:.1}",
                a.calls,
                a.rows as f64 / c,
                a.pairs as f64 / c,
                a.experts as f64 / c,
                us / c,
                us / a.pairs.max(1) as f64,
                us / a.experts.max(1) as f64,
                a.gap_ns as f64 / 1e3 / c,
            );
            *a = Acc {
                last_end: a.last_end,
                ..Acc::default()
            };
        }
    }
}

/// Routed experts holding the model's own weights in host memory, run by mistral.rs's own
/// kernels. MoE steps move activations in and the result back; long prefill chunks run on the
/// GPU instead when a prefill stream is attached.
pub(crate) struct CpuExperts {
    /// Page-locked registrations of `experts`' stacks (prefill streaming only).
    _pinned: Vec<PinnedStack>,
    experts: Arc<MoEExperts>,
    /// The experts' activation, for the fused decode path.
    act: Activation,
    layer_device: Device,
    layer: usize,
    prefill: Option<Arc<PrefillStream>>,
}

impl CpuExperts {
    pub(crate) fn new(
        experts: MoEExperts,
        act: Activation,
        layer_device: Device,
        layer: usize,
        prefill: Option<Arc<PrefillStream>>,
        pinned: Vec<PinnedStack>,
    ) -> Self {
        Self {
            _pinned: pinned,
            experts: Arc::new(experts),
            act,
            layer_device,
            layer,
            prefill,
        }
    }
}

impl RoutedExperts for CpuExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        if let Some(ps) = &self.prefill {
            if let Some(ys) = ps.forward(self.layer, xs, &topk_weights, topk_ids)? {
                return Ok(ys);
            }
        }
        let xs_cpu = xs.to_device(&Device::Cpu)?;
        let weights_cpu = topk_weights.to_device(&Device::Cpu)?;
        let ids_cpu = topk_ids.to_device(&Device::Cpu)?;
        let (batch, seq, hidden) = xs.dims3()?;
        let tokens = batch * seq;
        if tokens > FUSED_MAX_TOKENS {
            // A prefill chunk.
            let ys = self.experts.forward(&xs_cpu, weights_cpu, &ids_cpu)?;
            return ys.to_device(&self.layer_device);
        }
        // One token, or a speculative verify step's rows.
        let (experts, act) = (Arc::clone(&self.experts), self.act);
        let run = move || -> candle_core::Result<Vec<f32>> {
            let x = xs_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let ids = ids_cpu.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
            let ws = weights_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            if let Some(ys) = fused_decode(&experts, act, &x, &ids, &ws, ids.len() / tokens)? {
                return Ok(ys);
            }
            experts
                .forward(&xs_cpu, weights_cpu, &ids_cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        if seq == 1 && self.layer_device.is_cuda() {
            // Decode: return before the CPU experts finish; the next layer's launches overlap
            // them (`deferred`).
            let ys = crate::deferred::start(&self.layer_device, None, batch, hidden, run)?.upload()?;
            return ys.reshape((batch, seq, hidden))?.to_dtype(xs.dtype());
        }
        Tensor::from_vec(run()?, (batch, seq, hidden), &Device::Cpu)?
            .to_dtype(xs.dtype())?
            .to_device(&self.layer_device)
    }
}
