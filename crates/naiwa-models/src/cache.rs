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

/// Host-RAM snapshot of a completed prompt prefix (`KV + QSA aux + GDN + PLE + MTP KV`),
/// consuming zero GPU VRAM and supporting both exact prompt reuse and multi-turn extensions.
pub struct HybridPrefixSnapshot {
    /// Prompt token IDs represented by this snapshot (length `P >= 2`).
    pub tokens: Vec<u32>,
    /// Number of paged KV blocks captured (`P.div_ceil(block_size)`).
    pub num_blocks: usize,
    /// Per-layer `(layer_idx, k_blocks_cpu, v_blocks_cpu)` for every FullAttention + MTP layer.
    pub kv_layers: Vec<(usize, Tensor, Tensor)>,
    /// Per-layer `(layer_idx, raw_cpu, block_keys_cpu)` for QSA aux caches (including MTP).
    pub qsa_aux_layers: Vec<(usize, Tensor, Tensor)>,
    /// Per-layer `(layer_idx, conv_lane_cpu, rec_lane_cpu)` for every GDN layer.
    pub gdn_layers: Vec<(usize, Tensor, Tensor)>,
    /// Optional PLE state `(conv_state_cpu, history)`.
    pub ple_state: Option<(Tensor, Vec<Option<u32>>)>,
    /// Final collapsed hidden state `[1, hidden]` at position `P - 1` (on CPU, 5 KiB) so exact
    /// prompt matches can project `lm_head` directly for any sampling temperature/seed.
    pub last_final_hidden: Tensor,
    /// Optional pre-final-mixer 4-stream residual `[1, 1, hc * hidden]` at position `P - 1` (on
    /// CPU, 20 KiB) so MTP can warm up position `P - 1` with the newly sampled or next prompt token.
    pub last_spec_hidden: Option<Tensor>,
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
        let mtp_lanes = if has_mtp { 4 } else { 1 };
        Self::new_with_mtp_lanes(
            cfg,
            has_mtp,
            mtp_lanes,
            num_blocks,
            block_size,
            max_seqs,
            dtype,
            device,
        )
    }

    /// Allocate the full paged KV pool and recurrent state slots with an explicit GDN checkpoint
    /// lane count on slot 0 (e.g. `mtp_n_predict + 1` so MTP-2 uses 3 lanes instead of 4, saving
    /// 72 MiB of F32 recurrent state VRAM).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_mtp_lanes(
        cfg: &Qwen4ExpConfig,
        has_mtp: bool,
        mtp_checkpoint_lanes: usize,
        num_blocks: usize,
        block_size: usize,
        max_seqs: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let total_kv_layers = cfg.num_hidden_layers + usize::from(has_mtp);
        let mut kv_caches = Vec::with_capacity(total_kv_layers);
        let mut gdn_states = Vec::with_capacity(cfg.num_hidden_layers);
        let slot0_lanes = if has_mtp && device.is_cuda() {
            mtp_checkpoint_lanes.max(1)
        } else {
            1
        };

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
                    // Slot 0 gets `slot0_lanes` checkpoint lanes when MTP is active so single-seq
                    // MTP verification (up to n_predict = slot0_lanes - 1) runs zero-copy/zero-replay.
                    for s in 0..max_seqs {
                        let lanes = if s == 0 { slot0_lanes } else { 1 };
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

    /// Estimate exact GPU VRAM bytes required by `HybridCache` + QSA auxiliary caches.
    pub fn estimate_vram_bytes(
        cfg: &Qwen4ExpConfig,
        has_mtp: bool,
        mtp_checkpoint_lanes: usize,
        num_blocks: usize,
        block_size: usize,
        max_seqs: usize,
        dtype: DType,
    ) -> usize {
        let elem = dtype.size_in_bytes();
        let full_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count()
            + usize::from(has_mtp);
        let linear_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count();

        // Paged K + V caches: 2 * num_blocks * block_size * kv_heads * head_dim * elem
        let kv_per_layer = 2 * num_blocks * block_size * cfg.num_key_value_heads * cfg.head_dim * elem;
        // QSA aux cache: per block, block_size tokens * head_dim * elem (plus block keys)
        let qsa_aux_per_layer = num_blocks * block_size * cfg.qsa.head_dim * elem * 2;
        let kv_total = full_layers * (kv_per_layer + qsa_aux_per_layer);

        // GDN states: slot 0 gets `mtp_checkpoint_lanes` when MTP is on; other slots get 1.
        let slot0_lanes = if has_mtp {
            mtp_checkpoint_lanes.max(1)
        } else {
            1
        };
        let total_lanes = if max_seqs == 0 {
            0
        } else {
            slot0_lanes + (max_seqs - 1)
        };
        let gdn_rec_per_lane = cfg.gdn.num_v_heads * cfg.gdn.head_v_dim * cfg.gdn.head_k_dim * 4;
        let gdn_conv_per_lane = cfg.gdn.conv_dim() * cfg.gdn.conv_kernel_dim * elem;
        let gdn_total = linear_layers * total_lanes * (gdn_rec_per_lane + gdn_conv_per_lane);

        // PLE conv states
        let ple_total = cfg.ple.as_ref().map_or(0, |ple| {
            max_seqs * cfg.hc_hidden_size() * ple.conv_state_len() * elem
        });

        kv_total + gdn_total + ple_total
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

    /// Immutable borrow of all PLE slot states.
    pub fn ple_states(&self) -> Option<&[PleSeqState]> {
        self.ple_states.as_deref()
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
                let x_t = stash.conv_inputs[batch_idx].narrow(0, 0, keep)?.t()?; // [hc*hidden, keep]
                let padded = Tensor::cat(&[&restored.conv_state, &x_t], 1)?;
                let total = padded.dim(1)?;
                restored.conv_state = padded
                    .narrow(1, total - state_len, state_len)?
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

    /// Snapshot paged KV blocks, GDN active lane, and PLE state for `(slot, block_table)` into
    /// host CPU RAM (zero GPU VRAM).
    pub(crate) fn snapshot_state_to_cpu(
        &self,
        slot: usize,
        block_table: &[u32],
    ) -> Result<(
        Vec<(usize, Tensor, Tensor)>,
        Vec<(usize, Tensor, Tensor)>,
        Option<(Tensor, Vec<Option<u32>>)>,
    )> {
        let mut kv_layers = Vec::new();
        for (layer_idx, kv_opt) in self.kv_caches.iter().enumerate() {
            if let Some((kc, vc)) = kv_opt {
                let (k_cpu, v_cpu) = if block_table.len() == 1 {
                    let b = block_table[0] as usize;
                    (
                        kc.narrow(0, b, 1)?.to_device(&Device::Cpu)?,
                        vc.narrow(0, b, 1)?.to_device(&Device::Cpu)?,
                    )
                } else {
                    let mut ks = Vec::with_capacity(block_table.len());
                    let mut vs = Vec::with_capacity(block_table.len());
                    for &blk in block_table {
                        let b = blk as usize;
                        ks.push(kc.narrow(0, b, 1)?.to_device(&Device::Cpu)?);
                        vs.push(vc.narrow(0, b, 1)?.to_device(&Device::Cpu)?);
                    }
                    (Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?)
                };
                kv_layers.push((layer_idx, k_cpu, v_cpu));
            }
        }
        let mut gdn_layers = Vec::new();
        for (layer_idx, slots_opt) in self.gdn_states.iter().enumerate() {
            if let Some(slots) = slots_opt {
                let (conv, rec) = slots[slot].snapshot_lane_to_cpu()?;
                gdn_layers.push((layer_idx, conv, rec));
            }
        }
        let ple_state = match &self.ple_states {
            Some(states) => {
                let s = &states[slot];
                Some((s.conv_state.to_device(&Device::Cpu)?, s.history.clone()))
            }
            None => None,
        };
        Ok((kv_layers, gdn_layers, ple_state))
    }

    /// Restore host-RAM KV blocks, GDN lane 0, and PLE state into `(slot, block_table)`.
    pub(crate) fn restore_state_from_cpu(
        &mut self,
        slot: usize,
        block_table: &[u32],
        kv_layers: &[(usize, Tensor, Tensor)],
        gdn_layers: &[(usize, Tensor, Tensor)],
        ple_state: Option<&(Tensor, Vec<Option<u32>>)>,
    ) -> Result<()> {
        for (layer_idx, k_cpu, v_cpu) in kv_layers {
            if let Some(Some((kc, vc))) = self.kv_caches.get(*layer_idx) {
                let dev = kc.device();
                let k_dev = k_cpu.to_device(dev)?;
                let v_dev = v_cpu.to_device(dev)?;
                for (i, &blk) in block_table.iter().enumerate() {
                    kc.slice_set(&k_dev.narrow(0, i, 1)?, 0, blk as usize)?;
                    vc.slice_set(&v_dev.narrow(0, i, 1)?, 0, blk as usize)?;
                }
            }
        }
        for (layer_idx, conv_cpu, rec_cpu) in gdn_layers {
            if let Some(Some(slots)) = self.gdn_states.get_mut(*layer_idx) {
                slots[slot].restore_lane_from_cpu(conv_cpu, rec_cpu)?;
            }
        }
        if let (Some(states), Some((conv_cpu, hist))) = (self.ple_states.as_mut(), ple_state) {
            let dev = states[slot].conv_state.device().clone();
            states[slot].conv_state = conv_cpu.to_device(&dev)?;
            states[slot].history = hist.clone();
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
