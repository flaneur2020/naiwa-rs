//! Unified `HybridCache` holding Paged KV blocks, GDN recurrent states, and PLE n-gram states,
//! with exact prefix rollback for Built-in MTP speculative decoding.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use naiwa_attn::{alloc_kv_cache, GdnSeqStash, GdnSeqState};

use crate::qwen4_exp::config::{LayerType, PleConfig, Qwen4ExpConfig};

/// Per-sequence PLE state (conv history + token history).
#[derive(Clone, Debug)]
pub struct PleSeqState {
    /// `[hc * hidden, conv_state_len]` in model dtype.
    pub conv_state: Tensor,
    /// `[context_len]` (`ngram_size - 1`) in F32, storing `token_id + 1.0` (`0.0` = empty/EOS).
    pub history: Vec<Option<u32>>,
}

impl PleSeqState {
    pub fn new(
        hc_hidden: usize,
        ple: &PleConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv_state: Tensor::zeros((hc_hidden, ple.conv_state_len()), dtype, device)?,
            history: vec![None; ple.context_len()],
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.history.fill(None);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            conv_state: self.conv_state.copy()?,
            history: self.history.clone(),
        })
    }
}

/// Stashed PLE pre-step state and conv inputs for exact rollback after partial MTP rejection.
#[derive(Clone, Debug)]
pub struct PleVerifyStash {
    /// Per-sequence pre-verify `PleSeqState` (aligned with `slots`).
    pub pre_states: Vec<PleSeqState>,
    /// Per-sequence normalized gated conv inputs `[len, hc * hidden]` in model dtype.
    pub conv_inputs: Vec<Tensor>,
    /// Per-sequence token IDs processed in this verify step.
    pub tokens: Vec<Vec<u32>>,
}

/// Complete recurrent stash captured during an MTP verification forward pass so any sequence
/// rejecting a suffix of draft tokens can be rolled back bit-exactly to `keep` tokens.
#[derive(Clone, Debug)]
pub struct RecurrentVerifyStash {
    /// Cache slot IDs of the sequences in the verify batch.
    pub slots: Vec<usize>,
    /// `(layer_idx, per-seq GdnSeqStash)` for every linear-attention layer.
    pub gdn_layers: Vec<(usize, Vec<GdnSeqStash>)>,
    /// Optional PLE stash if the model has a PLE layer.
    pub ple: Option<PleVerifyStash>,
}

/// Model-wide inference cache for Qwen3.8-Flash-Next:
/// - Paged KV caches for every `FullAttention` layer (plus the MTP layer at index `num_hidden_layers`).
/// - Per-slot `GdnSeqState` for every `LinearAttention` layer.
/// - Per-slot `PleSeqState` for the PLE layer (if enabled).
pub struct HybridCache {
    /// Paged `(key_cache, value_cache)` indexed by layer (`0..num_hidden_layers`, plus MTP at `num_hidden_layers`).
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
    /// Per-layer GDN states: `gdn_states[layer][slot]`.
    gdn_states: Vec<Option<Vec<GdnSeqState>>>,
    /// Per-slot PLE states (if the model has a PLE layer).
    ple_states: Option<Vec<PleSeqState>>,
    /// Free recurrent slot indices (`0..max_seqs`).
    free_slots: Vec<usize>,
    max_seqs: usize,
    block_size: usize,
    num_blocks: usize,
}

impl HybridCache {
    /// Allocate the full paged KV pool and recurrent state slots on `device`.
    pub fn new(
        cfg: &Qwen4ExpConfig,
        has_mtp: bool,
        num_blocks: usize,
        block_size: usize,
        max_seqs: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let total_kv_layers = cfg.num_hidden_layers + usize::from(has_mtp);
        let mut kv_caches = Vec::with_capacity(total_kv_layers);
        let mut gdn_states = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_idx in 0..cfg.num_hidden_layers {
            match cfg.layer_types[layer_idx] {
                LayerType::FullAttention => {
                    let kv = alloc_kv_cache(
                        num_blocks,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        block_size,
                        dtype,
                        device,
                    )?;
                    kv_caches.push(Some(kv));
                    gdn_states.push(None);
                }
                LayerType::LinearAttention => {
                    kv_caches.push(None);
                    let mut slots = Vec::with_capacity(max_seqs);
                    // Slot 0 gets 4 checkpoint lanes when MTP is active so single-sequence MTP
                    // verification (up to n_predict=3) runs zero-copy/zero-alloc/zero-replay!
                    for s in 0..max_seqs {
                        let lanes = if has_mtp && s == 0 && device.is_cuda() {
                            4
                        } else {
                            1
                        };
                        slots.push(GdnSeqState::new_with_lanes(
                            &cfg.gdn,
                            dtype,
                            DType::F32,
                            lanes,
                            device,
                        )?);
                    }
                    gdn_states.push(Some(slots));
                }
            }
        }
        if has_mtp {
            let kv = alloc_kv_cache(
                num_blocks,
                cfg.num_key_value_heads,
                cfg.head_dim,
                block_size,
                dtype,
                device,
            )?;
            kv_caches.push(Some(kv));
        }

        let ple_states = match &cfg.ple {
            Some(ple) => {
                let mut slots = Vec::with_capacity(max_seqs);
                for _ in 0..max_seqs {
                    slots.push(PleSeqState::new(cfg.hc_hidden_size(), ple, dtype, device)?);
                }
                Some(slots)
            }
            None => None,
        };

        let free_slots = (0..max_seqs).rev().collect();

        Ok(Self {
            kv_caches,
            gdn_states,
            ple_states,
            free_slots,
            max_seqs,
            block_size,
            num_blocks,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// Allocate and zero-reset a recurrent state slot for a new sequence.
    pub fn alloc_slot(&mut self) -> Result<Option<usize>> {
        let Some(slot) = self.free_slots.pop() else {
            return Ok(None);
        };
        self.reset_slot(slot)?;
        Ok(Some(slot))
    }

    /// Return a recurrent slot to the free pool.
    pub fn free_slot(&mut self, slot: usize) {
        if slot < self.max_seqs && !self.free_slots.contains(&slot) {
            self.free_slots.push(slot);
        }
    }

    /// Zero-reset all GDN and PLE states for `slot`.
    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        for layer in self.gdn_states.iter_mut().flatten() {
            layer[slot].reset()?;
        }
        if let Some(ple) = self.ple_states.as_mut() {
            ple[slot].reset()?;
        }
        Ok(())
    }

    /// Paged `(key_cache, value_cache)` for `layer_idx` (including MTP at `num_hidden_layers`).
    pub fn kv_layer(&self, layer_idx: usize) -> Option<&(Tensor, Tensor)> {
        self.kv_caches.get(layer_idx).and_then(|opt| opt.as_ref())
    }

    /// Mutable borrow of all GDN slot states for `layer_idx`.
    pub fn gdn_layer_mut(&mut self, layer_idx: usize) -> Option<&mut [GdnSeqState]> {
        self.gdn_states
            .get_mut(layer_idx)
            .and_then(|opt| opt.as_deref_mut())
    }

    /// Mutable borrow of all PLE slot states.
    pub fn ple_states_mut(&mut self) -> Option<&mut [PleSeqState]> {
        self.ple_states.as_deref_mut()
    }

    /// Roll back PLE states for the given `(batch_idx, keep_rows)` pairs using `stash`.
    pub fn rollback_ple(
        &mut self,
        stash: &PleVerifyStash,
        slots: &[usize],
        rows: &[(usize, usize)],
        ple: &PleConfig,
    ) -> Result<()> {
        let Some(states) = self.ple_states.as_mut() else {
            return Ok(());
        };
        let state_len = ple.conv_state_len();
        let context_len = ple.context_len();
        for &(batch_idx, keep) in rows {
            let slot = slots[batch_idx];
            let pre = &stash.pre_states[batch_idx];
            let mut restored = pre.snapshot()?;
            if keep > 0 {
                let x = stash.conv_inputs[batch_idx].narrow(0, 0, keep)?; // [keep, hc*hidden]
                let hist_t = restored.conv_state.t()?; // [state_len, hc*hidden]
                let padded = Tensor::cat(&[&hist_t, &x], 0)?;
                let total = padded.dim(0)?;
                restored.conv_state = padded
                    .narrow(0, total - state_len, state_len)?
                    .t()?
                    .contiguous()?;

                let mut hist = pre.history.clone();
                for &tok in &stash.tokens[batch_idx][..keep] {
                    hist.push(Some(tok));
                }
                restored.history = hist[hist.len() - context_len..].to_vec();
            }
            states[slot] = restored;
        }
        Ok(())
    }

    /// Copy `slot` into a new `GdnSeqState` slice helper (used internally when batching distinct slots).
    pub(crate) fn collect_gdn_mut<'a>(
        layer_slots: &'a mut [GdnSeqState],
        slots: &[usize],
    ) -> Result<Vec<&'a mut GdnSeqState>> {
        // Verify slots are distinct and in bounds before building disjoint mutable references.
        for (i, &s) in slots.iter().enumerate() {
            if s >= layer_slots.len() {
                candle_core::bail!("GDN slot {s} out of bounds (max {})", layer_slots.len());
            }
            if slots[..i].contains(&s) {
                candle_core::bail!("duplicate GDN slot {s} in batch");
            }
        }
        let ptr = layer_slots.as_mut_ptr();
        let mut refs = Vec::with_capacity(slots.len());
        for &s in slots {
            // SAFETY: all indices in `slots` were verified distinct and in bounds above.
            refs.push(unsafe { &mut *ptr.add(s) });
        }
        Ok(refs)
    }
}

/// Keep compiler happy if `IndexOp` is used in future helpers.
#[allow(dead_code)]
fn _unused_index_op_marker(t: &Tensor) -> Result<Tensor> {
    t.i(0)
}
