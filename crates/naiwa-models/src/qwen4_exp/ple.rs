//! Hashed n-gram Per-Layer Embedding (PLE) for Qwen3.8-Flash-Next (GGUF IQ4_NL mmap table).

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use naiwa_attn::grouped_rms_norm;
use naiwa_quant::{GgufArchive, QLinear};
use rayon::prelude::*;

use crate::cache::{PleSeqState, PleVerifyStash};
use crate::qwen4_exp::config::PleConfig;

const GATE_MAGNITUDE_FLOOR: f64 = 1e-6;
pub(crate) const GGUF_PLE_TABLE_NAME: &str = "per_layer_token_embd.weight";
const GGML_TYPE_IQ4_NL: u32 = 20;
const IQ4_NL_BLOCK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn ple_gather_probe(nanos: usize, rows: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("PLE_DBG").is_some()) {
        return;
    }
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static NS: AtomicUsize = AtomicUsize::new(0);
    static ROWS: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Relaxed) + 1;
    NS.fetch_add(nanos, Relaxed);
    ROWS.fetch_add(rows, Relaxed);
    eprintln!(
        "PLE probe: {n} gather calls, cumulative {:.1} ms, {:.2} ms/call, {:.0} rows/call",
        NS.load(Relaxed) as f64 / 1e6,
        NS.load(Relaxed) as f64 / n as f64 / 1e6,
        ROWS.load(Relaxed) as f64 / n as f64,
    );
}

#[derive(Clone, Debug)]
pub(crate) struct PleHash {
    pub multipliers: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    pub eos: u32,
}

impl PleHash {
    pub(crate) fn from_config(ple: &PleConfig) -> Self {
        Self {
            multipliers: ple.hash.layer_multipliers.clone(),
            vocab_sizes: ple.hash.head_vocab_sizes.clone(),
            offsets: ple.hash.head_offsets.clone(),
            heads_per_ngram: ple.heads_per_ngram,
            eos: ple.eos_token_id,
        }
    }

    pub(crate) fn num_heads(&self) -> usize {
        self.vocab_sizes.len()
    }

    /// Compute table row indices for `tokens[local]` given oldest-first `history`.
    pub(crate) fn rows(
        &self,
        tokens: &[u32],
        local: usize,
        history: &[Option<u32>],
        out: &mut [u64],
    ) {
        let ngram = self.multipliers.len();
        let mut ctx = vec![self.eos as u64; ngram];
        ctx[0] = tokens[local] as u64;
        let mut cut = false;
        for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
            let tok = if local >= s {
                Some(tokens[local - s])
            } else {
                let back = s - local;
                history.len().checked_sub(back).and_then(|idx| history[idx])
            };
            cut = cut || tok.is_none_or(|tok| tok == self.eos);
            *slot = if cut {
                self.eos as u64
            } else {
                tok.unwrap() as u64
            };
        }
        for n in 2..=ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&self.multipliers)
                .fold(0u64, |mixed, (tok, mult)| mixed ^ tok.wrapping_mul(*mult));
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                out[h] = mixed % self.vocab_sizes[h] + self.offsets[h];
            }
        }
    }
}

fn iq4_nl_row_bytes(head_dim: usize) -> usize {
    head_dim / IQ4_NL_BLOCK * IQ4_NL_BLOCK_BYTES
}

fn dequantize_iq4_nl_row(src: &[u8], out: &mut [f32]) {
    let (blocks, _) = src.as_chunks::<IQ4_NL_BLOCK_BYTES>();
    let (dsts, _) = out.as_chunks_mut::<IQ4_NL_BLOCK>();
    for (block, dst) in blocks.iter().zip(dsts) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let (lo, hi) = dst.split_at_mut(IQ4_NL_BLOCK / 2);
        for (j, &byte) in block[2..].iter().enumerate() {
            lo[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte & 0xf)]);
            hi[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte >> 4)]);
        }
    }
}

/// Memory-mapped IQ4_NL n-gram embedding table gathered on the host.
pub(crate) struct PleTable {
    archive: Arc<GgufArchive>,
    head_dim: usize,
}

impl PleTable {
    pub(crate) fn load_gguf(archive: Arc<GgufArchive>, head_dim: usize) -> Result<Self> {
        let info = archive.tensor_info(GGUF_PLE_TABLE_NAME)?;
        if info.dtype().raw() != GGML_TYPE_IQ4_NL {
            candle_core::bail!(
                "Qwen4-Exp PLE table `{GGUF_PLE_TABLE_NAME}` is {}, only IQ4_NL is supported",
                info.dtype().name()
            );
        }
        match info.shape() {
            &[_, dim] if dim == head_dim && head_dim.is_multiple_of(IQ4_NL_BLOCK) => {}
            shape => candle_core::bail!(
                "Qwen4-Exp PLE table has shape {shape:?}, expected [_, {head_dim}]"
            ),
        }
        let table = Self { archive, head_dim };
        table.advise_random_access()?;
        tracing::info!(
            "Qwen4-Exp PLE rows are gathered from the memory-mapped IQ4_NL table on the host"
        );
        Ok(table)
    }

    fn advise_random_access(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let data = self.archive.tensor_data(GGUF_PLE_TABLE_NAME)?.bytes();
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
            let start = data.as_ptr() as usize / page * page;
            let len = data.as_ptr() as usize + data.len() - start;
            // SAFETY: range covers mapped pages of `data`; advice does not alter contents.
            unsafe {
                libc::madvise(start as *mut libc::c_void, len, libc::MADV_RANDOM);
            }
        }
        Ok(())
    }

    fn gather_host(&self, rows: &[u64], device: &Device) -> Result<Tensor> {
        let row_bytes = iq4_nl_row_bytes(self.head_dim);
        let bytes = self.archive.tensor_data(GGUF_PLE_TABLE_NAME)?.bytes();
        let mut out = vec![0f32; rows.len() * self.head_dim];
        out.par_chunks_exact_mut(self.head_dim)
            .zip(rows.par_iter())
            .try_for_each(|(dst, &row)| {
                let offset = row as usize * row_bytes;
                let src = bytes.get(offset..offset + row_bytes).ok_or_else(|| {
                    candle_core::Error::msg(format!("PLE row {row} out of range"))
                })?;
                dequantize_iq4_nl_row(src, dst);
                Ok::<_, candle_core::Error>(())
            })?;
        Tensor::from_vec(out, (rows.len(), self.head_dim), device)
    }
}

pub(crate) struct PleLayer {
    pub(crate) cfg: PleConfig,
    pub(crate) hash: PleHash,
    pub(crate) table: PleTable,
    pub(crate) key_proj: QLinear,
    pub(crate) value_proj: QLinear,
    /// Already-shifted `(1 + w)` F32 weights `[hc * hidden]`.
    pub(crate) norm_key: Tensor,
    pub(crate) norm_query: Tensor,
    pub(crate) norm_conv: Tensor,
    /// Depthwise dilated conv1d weight `[hc * hidden, conv_kernel_size]` in F32.
    pub(crate) conv_weight: Tensor,
    pub(crate) hc: usize,
    pub(crate) hidden: usize,
    pub(crate) eps: f64,
}

impl PleLayer {
    /// `hidden_states + ple(hidden_states, tokens)`, advancing per-sequence `states`.
    /// When `stash_out` is `Some`, populates a [`PleVerifyStash`] for MTP speculative rollback.
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        tokens_host: &[u32],
        spans: &[(usize, usize)],
        slots: &[usize],
        states: &mut [PleSeqState],
        stash_out: Option<&mut Option<PleVerifyStash>>,
    ) -> Result<Tensor> {
        let dims = hidden_states.dims().to_vec();
        let width = dims[dims.len() - 1];
        let flat = hidden_states.reshape(((), width))?;
        let n_tokens = flat.dim(0)?;

        let pre_states = if stash_out.is_some() {
            let mut v = Vec::with_capacity(slots.len());
            for &slot in slots {
                v.push(states[slot].snapshot()?);
            }
            Some(v)
        } else {
            None
        };

        // 1. Hash token n-grams on host, update history, and gather IQ4_NL embeddings
        let heads = self.hash.num_heads();
        let context = self.cfg.context_len();
        let mut rows = vec![self.hash.offsets[0]; n_tokens * heads];
        let mut seq_tokens_copy = if stash_out.is_some() {
            Some(Vec::with_capacity(spans.len()))
        } else {
            None
        };

        for (seq_idx, &(start, len)) in spans.iter().enumerate() {
            let slot = slots[seq_idx];
            let seq_tokens = &tokens_host[start..start + len];
            if let Some(copy) = seq_tokens_copy.as_mut() {
                copy.push(seq_tokens.to_vec());
            }
            let hist = &mut states[slot].history;
            for local in 0..len {
                let t = start + local;
                self.hash
                    .rows(seq_tokens, local, hist, &mut rows[t * heads..(t + 1) * heads]);
            }
            let mut next_hist = hist.clone();
            next_hist.extend(seq_tokens.iter().map(|&t| Some(t)));
            *hist = next_hist[next_hist.len() - context..].to_vec();
        }

        let gather_start = std::time::Instant::now();
        let emb = self.table.gather_host(&rows, flat.device())?;
        ple_gather_probe(gather_start.elapsed().as_nanos() as usize, rows.len());
        let emb = emb
            .reshape((n_tokens, heads * self.table.head_dim))?
            .to_dtype(flat.dtype())?;

        // 2. Gated projection + grouped RMSNorm
        let (gated, normed) = self.gate(&emb, &flat)?;

        if let (Some(dst), Some(pre), Some(toks)) = (stash_out, pre_states, seq_tokens_copy) {
            let mut conv_inputs = Vec::with_capacity(spans.len());
            for &(start, len) in spans {
                conv_inputs.push(normed.narrow(0, start, len)?.contiguous()?);
            }
            *dst = Some(PleVerifyStash {
                pre_states: pre,
                conv_inputs,
                tokens: toks,
            });
        }

        // 3. Dilated depthwise causal conv1d + residual add
        let out = self.conv_step(&flat, &gated, &normed, spans, slots, states)?;
        out.reshape(dims)
    }

    fn gate(&self, emb: &Tensor, flat: &Tensor) -> Result<(Tensor, Tensor)> {
        let key = self.key_proj.forward(emb)?;
        let value = self.value_proj.forward(emb)?;
        #[cfg(feature = "cuda")]
        if flat.device().is_cuda() {
            return naiwa_attn::qwen4_exp::ple_gate(
                &key,
                flat,
                &value,
                naiwa_attn::qwen4_exp::PleGateWeights {
                    key: &self.norm_key,
                    query: &self.norm_query,
                    conv: &self.norm_conv,
                },
                self.hc,
                self.eps,
            );
        }
        let n = flat.dim(0)?;
        let key = grouped_rms_norm(&key, &self.norm_key, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let query = grouped_rms_norm(flat, &self.norm_query, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (self.hidden as f64).sqrt())?;
        let magnitude = gate.abs()?.maximum(GATE_MAGNITUDE_FLOOR)?.sqrt()?;
        let gate = candle_nn::ops::sigmoid(&(magnitude * gate.sign()?)?)?;
        let gated = gate
            .broadcast_mul(&value.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .reshape((n, self.hc * self.hidden))?
            .to_dtype(flat.dtype())?;
        let normed = grouped_rms_norm(&gated, &self.norm_conv, self.hc, self.eps)?;
        Ok((gated, normed))
    }

    fn conv_step(
        &self,
        flat: &Tensor,
        gated: &Tensor,
        normed: &Tensor,
        spans: &[(usize, usize)],
        slots: &[usize],
        states: &mut [PleSeqState],
    ) -> Result<Tensor> {
        let kernel = self.cfg.conv_kernel_size;
        let dilation = self.cfg.conv_dilation();
        let state_len = self.cfg.conv_state_len();
        let mut conv_out = Vec::with_capacity(spans.len());

        for (seq_idx, &(start, len)) in spans.iter().enumerate() {
            let slot = slots[seq_idx];
            let state_dtype = states[slot].conv_state.dtype();
            let history = states[slot].conv_state.to_dtype(DType::F32)?.t()?; // [state_len, hc*hidden]
            let x = normed.narrow(0, start, len)?.to_dtype(DType::F32)?;
            let padded = Tensor::cat(&[&history, &x], 0)?;
            let mut acc = Tensor::zeros(x.shape(), DType::F32, x.device())?;
            for k in 0..kernel {
                let offset = state_len - (kernel - 1 - k) * dilation;
                let tap = padded.narrow(0, offset, len)?;
                acc = (acc + tap.broadcast_mul(&self.conv_weight.i((.., k))?.unsqueeze(0)?)?)?;
            }
            conv_out.push(candle_nn::ops::silu(&acc)?);
            let total = padded.dim(0)?;
            states[slot].conv_state = padded
                .narrow(0, total - state_len, state_len)?
                .t()?
                .to_dtype(state_dtype)?
                .contiguous()?;
        }

        let conv = if conv_out.len() == 1 {
            conv_out.pop().unwrap()
        } else {
            Tensor::cat(&conv_out, 0)?
        };
        if conv.dim(0)? != flat.dim(0)? {
            candle_core::bail!("Qwen4-Exp PLE requires unpadded contiguous sequences");
        }
        ((flat.to_dtype(DType::F32)? + gated.to_dtype(DType::F32)?)? + conv)?.to_dtype(flat.dtype())
    }
}
