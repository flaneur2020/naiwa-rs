//! Core `CachedLayer` state, split by lifecycle into three inline views:
//! - [`LayerWeights`] — immutable host weights, router, and geometry fixed at model load.
//! - [`LayerCache`] — mutable slot directory, hit counters, and VRAM borrow coordination.
//! - [`LayerTransport`] — async DMA/doorbell endpoints (`copy_stream`, `staging`, `preload`, `lookahead_chan`).
//!
//! Companion modules:
//! - [`super::layer_registry`] — model-wide `Layers` registry and Decode ↔ Prefill VRAM handoff.
//! - [`super::activation_channel`] — `LookaheadChannel` (advisory).
//! - [`super::cache_policy`] — pure selection/ranking/victim functions (`#[inline]`).
//! - [`super::cache_updates`] — `admit`, `replace`, `promote_staged`, `prefetch_predicted`, `spawn_fill`.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use candle_core::{Device, Tensor};
use mistralrs_core::layers::Activation;
use mistralrs_core::moe_offload::RoutedExperts;
use mistralrs_core::MoEExperts;

use crate::copy_stream::{CopyDone, CopyStream};
use crate::doorbell::{PassCell, TopkDoorbell};
use crate::host_memory::PinnedStack;
use crate::prefill_stream::PrefillStream;
use crate::stats;

use super::activation_channel::LookaheadChannel;
use super::layer_registry::Layers;
use super::slab::BorrowGate;
use super::{Slot, StagingRings};

/// The largest row count the *decode split* may be asked to take, i.e. what every warmup-sized
/// mapped buffer and the CPU-result ring must cover. Ceiling for `NAIWA_PREFILL_THRESHOLD`.
pub(crate) const MAX_DECODE_ROWS: usize = 256;

#[inline]
fn prefill_threshold() -> usize {
    crate::config::runtime().prefill_threshold
}

/// Mutable per-layer decode state: doorbell, slots, and LRU/hit counters.
pub(crate) struct LayerState {
    pub(crate) doorbell: Option<Arc<TopkDoorbell>>,
    pub(crate) slots: Vec<Slot>,
    pub(crate) hit_counts: Vec<u64>,
    /// Decode steps seen by this layer.
    pub(crate) tick: u64,
    /// Slots `[0, usable)` are in service; the rest are lent to the prefill stream.
    pub(crate) usable: usize,
}

/// Immutable host weights, router, and tensor geometry fixed at model load.
pub(crate) struct LayerWeights {
    /// Page-locked registrations of `cpu`'s stacks; keeps each stack alive while registered.
    pub(crate) _pinned: Vec<PinnedStack>,
    pub(crate) cpu: MoEExperts,
    pub(crate) slot_device: Device,
    pub(crate) layer: usize,
    pub(crate) hidden: usize,
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) act: Activation,
    /// `[num_experts, hidden]` F32 on the CPU; `None` disables lookahead for this layer.
    pub(crate) router: Option<Tensor>,
}

/// Mutable cache directory and VRAM loan coordination for one layer.
pub(crate) struct LayerCache {
    pub(crate) state: Mutex<LayerState>,
    pub(crate) layers: Arc<Layers>,
    pub(crate) borrow_gate: Arc<BorrowGate>,
    /// A background fill of this layer is running; at most one at a time.
    pub(crate) filling: AtomicBool,
}

/// Async DMA and doorbell transport endpoints for one layer.
pub(crate) struct LayerTransport {
    /// Uploads fills alongside compute; `None` off CUDA.
    pub(crate) copy_stream: Option<Arc<CopyStream>>,
    pub(crate) staging: Arc<StagingRings>,
    /// Staging DMA completion event on the copy stream; `None` disables staging for this layer.
    pub(crate) staging_done: Option<Arc<CopyDone>>,
    /// Preload completion handoff between the lookahead thread and the token thread.
    pub(crate) preload: Option<PassCell>,
    /// Advisory GPU -> host activation channel for lookahead prediction.
    pub(crate) lookahead_chan: Option<Arc<LookaheadChannel>>,
    pub(crate) pcie_frac: f32,
    pub(crate) lookahead_swaps: usize,
    pub(crate) lookahead: bool,
}

pub(crate) struct CachedLayer {
    pub(crate) weights: LayerWeights,
    pub(crate) cache: LayerCache,
    pub(crate) transport: LayerTransport,
    pub(crate) prefill: Option<Arc<PrefillStream>>,
}

/// Wrapper implementing [`RoutedExperts`] over `Arc<CachedLayer>`.
pub(crate) struct CachedExperts(pub(crate) Arc<CachedLayer>);

impl RoutedExperts for CachedExperts {
    fn forward(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let layer = &self.0;
        let (batch, seq, hidden_dim) = xs.dims3()?;
        crate::dbg_note(
            "moe_forward",
            &format!("layer {}: batch {batch} seq {seq}", layer.weights.layer),
        );
        debug_assert_eq!(hidden_dim, layer.weights.hidden);
        let num_tokens = batch * seq;

        if layer.weights.layer == 0 && crate::config::diag().layer_dbg {
            eprintln!(
                "LAYER_DBG rows={num_tokens} prefill_threshold={} slab_borrowed={} prefill_engine={}",
                prefill_threshold(),
                layer.cache.borrow_gate.borrowed(),
                layer.prefill.is_some(),
            );
        }
        if num_tokens > prefill_threshold() {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.cache.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            if !crate::config::runtime().no_prefill_fill && !crate::moe::slab::Slab::borrowing() {
                if let Some(next) = layer.cache.layers.get(layer.weights.layer + 1) {
                    next.spawn_fill();
                }
                layer.spawn_fill();
            }
            if let Some(ps) = &layer.prefill {
                if let Some(out) = ps.forward(layer.weights.layer, xs, &topk_weights, topk_ids)? {
                    return Ok(out);
                }
            }
            let out = layer.weights.cpu.forward(
                &xs.to_device(&Device::Cpu)?,
                topk_weights.to_device(&Device::Cpu)?,
                &topk_ids.to_device(&Device::Cpu)?,
            )?;
            return out.to_device(&layer.weights.slot_device);
        }

        // Decode regime: ensure slab loan is released and any displaced slots are refilled at layer 0.
        layer
            .cache
            .layers
            .enter_decode(&layer.cache.borrow_gate, layer.weights.layer == 0);
        super::decode::forward(layer, xs, &topk_weights, topk_ids, layer.weights.act)
    }
}

impl CachedLayer {
    #[inline]
    pub(crate) fn state(&self) -> &Mutex<LayerState> {
        &self.cache.state
    }

    #[inline]
    pub(crate) fn cpu(&self) -> &MoEExperts {
        &self.weights.cpu
    }

    #[inline]
    pub(crate) fn slot_device(&self) -> &Device {
        &self.weights.slot_device
    }

    #[inline]
    pub(crate) fn hidden(&self) -> usize {
        self.weights.hidden
    }

    #[inline]
    pub(crate) fn num_experts(&self) -> usize {
        self.weights.num_experts
    }

    #[inline]
    pub(crate) fn top_k(&self) -> usize {
        self.weights.top_k
    }

    #[inline]
    pub(crate) fn act(&self) -> Activation {
        self.weights.act
    }

    #[inline]
    pub(crate) fn router(&self) -> Option<&Tensor> {
        self.weights.router.as_ref()
    }

    #[inline]
    pub(crate) fn lookahead(&self) -> bool {
        self.transport.lookahead
    }

    #[inline]
    pub(crate) fn lookahead_chan(&self) -> Option<&Arc<LookaheadChannel>> {
        self.transport.lookahead_chan.as_ref()
    }

    #[inline]
    pub(crate) fn copy_stream(&self) -> Option<&Arc<CopyStream>> {
        self.transport.copy_stream.as_ref()
    }

    #[inline]
    pub(crate) fn pcie_frac(&self) -> f32 {
        self.transport.pcie_frac
    }

    #[inline]
    pub(crate) fn staging_done(&self) -> Option<&Arc<CopyDone>> {
        self.transport.staging_done.as_ref()
    }

    #[inline]
    pub(crate) fn layers(&self) -> &Arc<Layers> {
        &self.cache.layers
    }

    #[inline]
    pub(crate) fn staging(&self) -> &StagingRings {
        &self.transport.staging
    }

    #[inline]
    pub(crate) fn layer(&self) -> usize {
        self.weights.layer
    }

    /// True only when staging copies have a completion event (`--pcie-frac > 0` and event created).
    #[inline]
    pub(crate) fn staging_enabled(&self) -> bool {
        self.transport.staging_done.is_some()
    }

    /// Wait on the calling stream for this layer's staging DMA (GPU-side event wait).
    #[inline]
    pub(crate) fn wait_staging(&self) -> candle_core::Result<()> {
        let Some(done) = &self.transport.staging_done else {
            return Ok(());
        };
        stats::record_cache_sync_wait();
        done.wait_on(&self.weights.slot_device)
    }

    /// Wait on the calling stream for the latest preload claim, if any.
    #[inline]
    pub(crate) fn wait_preload(&self) -> candle_core::Result<()> {
        let Some(cell) = &self.transport.preload else {
            return Ok(());
        };
        if cell.wait_latest(&self.weights.slot_device)?.is_some() {
            stats::record_cache_sync_wait();
        }
        Ok(())
    }

    /// Create this layer's top-k doorbell at warmup, sized for the largest decode window.
    pub(crate) fn warmup_doorbell(&self) -> candle_core::Result<()> {
        let cap = MAX_DECODE_ROWS * self.weights.top_k;
        if let Some(bell) = TopkDoorbell::acquire(&self.weights.slot_device, cap)? {
            self.cache.state.lock().unwrap().doorbell = Some(Arc::new(bell));
        }
        Ok(())
    }

    /// Create this layer's staging ring at warmup so all ring addresses stay fixed across decode.
    pub(super) fn warmup_staging_ring(&self) -> candle_core::Result<()> {
        let Some(cs) = &self.transport.copy_stream else {
            return Ok(());
        };
        let Some((gate, up, down)) = self.weights.cpu.expert_qtensors() else {
            return Ok(());
        };
        self.transport.staging.ring(
            cs,
            &(Arc::clone(&gate), Arc::clone(&up), Arc::clone(&down)),
            &self.weights.slot_device,
        )?;
        Ok(())
    }
}
