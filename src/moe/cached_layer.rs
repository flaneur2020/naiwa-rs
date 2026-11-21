//! Core `CachedLayer` state, `CachedExperts` forward entry point, and lightweight accessors /
//! CUDA stream sync helpers.
//!
//! Companion modules split the surrounding responsibilities:
//! - [`super::layer_registry`] — model-wide `Layers` registry and loan empty/refill.
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
use super::{Slot, SlotBuf, StagingRings};

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

pub(crate) struct CachedLayer {
    /// Page-locked registrations of `cpu`'s stacks; keeps each stack alive while registered.
    pub(crate) _pinned: Vec<PinnedStack>,
    pub(crate) cpu: MoEExperts,
    pub(crate) slot_device: Device,
    pub(crate) layer: usize,
    pub(crate) layers: Arc<Layers>,
    pub(crate) staging: Arc<StagingRings>,
    pub(crate) borrow_gate: Arc<BorrowGate>,
    /// This step's staged experts, drained unconditionally within one step on the token thread.
    pub(crate) staged_now: Mutex<Vec<(u32, Arc<SlotBuf>)>>,
    pub(crate) hidden: usize,
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) act: Activation,
    /// `[num_experts, hidden]` F32 on the CPU; `None` disables lookahead for this layer.
    pub(crate) router: Option<Tensor>,
    /// Advisory GPU -> host activation channel for lookahead prediction.
    pub(crate) lookahead_chan: Option<Arc<LookaheadChannel>>,
    pub(crate) state: Mutex<LayerState>,
    /// Uploads fills alongside compute; `None` off CUDA.
    pub(crate) copy_stream: Option<Arc<CopyStream>>,
    pub(crate) lookahead_swaps: usize,
    pub(crate) lookahead: bool,
    pub(crate) pcie_frac: f32,
    pub(crate) prefill: Option<Arc<PrefillStream>>,
    /// A background fill of this layer is running; at most one at a time.
    pub(crate) filling: AtomicBool,
    /// Preload completion handoff between the lookahead thread and the token thread.
    pub(crate) preload: Option<PassCell>,
    /// Staging DMA completion event on the copy stream; `None` disables staging for this layer.
    pub(crate) staging_done: Option<Arc<CopyDone>>,
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
            &format!("layer {}: batch {batch} seq {seq}", layer.layer),
        );
        debug_assert_eq!(hidden_dim, layer.hidden);
        let num_tokens = batch * seq;

        if layer.layer == 0
            && num_tokens <= prefill_threshold()
            && crate::stats::vram_peak::enabled()
        {
            crate::stats::vram_peak::note_step(&layer.slot_device);
        }
        if layer.layer == 0 && crate::config::diag().layer_dbg {
            eprintln!(
                "LAYER_DBG rows={num_tokens} prefill_threshold={} slab_borrowed={} prefill_engine={}",
                prefill_threshold(),
                layer.borrow_gate.borrowed(),
                layer.prefill.is_some(),
            );
        }
        if num_tokens > prefill_threshold() {
            let ids_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
            {
                let mut st = layer.state.lock().unwrap();
                for &id in &ids_vec {
                    st.hit_counts[id as usize] += 1;
                }
            }
            if !crate::config::runtime().no_prefill_fill && !crate::moe::slab::Slab::borrowing() {
                if let Some(next) = layer.layers.get(layer.layer + 1) {
                    next.spawn_fill();
                }
                layer.spawn_fill();
            }
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

        // Decode: release the borrow gate and refill any displaced loan slots at layer 0.
        layer.borrow_gate.set_borrowed(false);
        if layer.layer == 0 {
            super::quiesce(false);
            if layer.layers.refill_lent() > 0 {
                crate::dbg_note("prefill_borrow", "the loan was given back");
            }
        }
        super::decode::forward(layer, xs, &topk_weights, topk_ids, layer.act)
    }
}

impl CachedLayer {
    #[inline]
    pub(crate) fn state(&self) -> &Mutex<LayerState> {
        &self.state
    }

    #[inline]
    pub(crate) fn cpu(&self) -> &MoEExperts {
        &self.cpu
    }

    #[inline]
    pub(crate) fn slot_device(&self) -> &Device {
        &self.slot_device
    }

    #[inline]
    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    #[inline]
    pub(crate) fn num_experts(&self) -> usize {
        self.num_experts
    }

    #[inline]
    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    #[inline]
    pub(crate) fn act(&self) -> Activation {
        self.act
    }

    #[inline]
    pub(crate) fn router(&self) -> Option<&Tensor> {
        self.router.as_ref()
    }

    #[inline]
    pub(crate) fn lookahead(&self) -> bool {
        self.lookahead
    }

    #[inline]
    pub(crate) fn lookahead_chan(&self) -> Option<&Arc<LookaheadChannel>> {
        self.lookahead_chan.as_ref()
    }

    #[inline]
    pub(crate) fn copy_stream(&self) -> Option<&Arc<CopyStream>> {
        self.copy_stream.as_ref()
    }

    #[inline]
    pub(crate) fn pcie_frac(&self) -> f32 {
        self.pcie_frac
    }

    #[inline]
    pub(crate) fn set_staged_now(&self, v: Vec<(u32, Arc<SlotBuf>)>) {
        *self.staged_now.lock().unwrap() = v;
    }

    #[inline]
    pub(crate) fn staging_done(&self) -> Option<&Arc<CopyDone>> {
        self.staging_done.as_ref()
    }

    #[inline]
    pub(crate) fn layers(&self) -> &Arc<Layers> {
        &self.layers
    }

    #[inline]
    pub(crate) fn staging(&self) -> &StagingRings {
        &self.staging
    }

    #[inline]
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    /// True only when staging copies have a completion event (`--pcie-frac > 0` and event created).
    #[inline]
    pub(crate) fn staging_enabled(&self) -> bool {
        self.staging_done.is_some()
    }

    /// Wait on the calling stream for this layer's staging DMA (GPU-side event wait).
    #[inline]
    pub(crate) fn wait_staging(&self) -> candle_core::Result<()> {
        let Some(done) = &self.staging_done else {
            return Ok(());
        };
        stats::record_cache_sync_wait();
        done.wait_on(&self.slot_device)
    }

    /// Wait on the calling stream for the latest preload claim, if any.
    #[inline]
    pub(crate) fn wait_preload(&self) -> candle_core::Result<()> {
        let Some(cell) = &self.preload else {
            return Ok(());
        };
        if cell.wait_latest(&self.slot_device)?.is_some() {
            stats::record_cache_sync_wait();
        }
        Ok(())
    }

    /// Create this layer's top-k doorbell at warmup, sized for the largest decode window.
    pub(crate) fn warmup_doorbell(&self) -> candle_core::Result<()> {
        let cap = MAX_DECODE_ROWS * self.top_k;
        if let Some(bell) = TopkDoorbell::acquire(&self.slot_device, cap)? {
            self.state.lock().unwrap().doorbell = Some(Arc::new(bell));
        }
        Ok(())
    }

    /// Create this layer's staging ring at warmup so all ring addresses stay fixed across decode.
    pub(super) fn warmup_staging_ring(&self) -> candle_core::Result<()> {
        let Some(cs) = &self.copy_stream else {
            return Ok(());
        };
        let Some((gate, up, down)) = self.cpu.expert_qtensors() else {
            return Ok(());
        };
        self.staging.ring(
            cs,
            &(Arc::clone(&gate), Arc::clone(&up), Arc::clone(&down)),
            &self.slot_device,
        )?;
        Ok(())
    }
}
