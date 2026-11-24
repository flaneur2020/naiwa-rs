//! Sparse MoE block with softmax top-k routing, pluggable `RoutedExperts`, and gated shared expert.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use naiwa_quant::{QLinear, RoutedExperts};

use crate::probe;

pub(crate) struct SharedMlp {
    pub(crate) gate_proj: QLinear,
    pub(crate) up_proj: QLinear,
    pub(crate) down_proj: QLinear,
}

impl SharedMlp {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

pub(crate) struct SparseMoeBlock {
    pub(crate) gate: QLinear,
    pub(crate) experts: Arc<dyn RoutedExperts>,
    pub(crate) shared_expert: SharedMlp,
    pub(crate) shared_expert_gate: QLinear,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let block_start = std::time::Instant::now();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let (topk_weights, topk_ids) =
            router_softmax_topk(&router_logits, self.num_experts_per_tok, self.norm_topk_prob)?;

        let hook_start = std::time::Instant::now();
        let y = self
            .experts
            .forward(xs, topk_weights, &topk_ids)?
            .reshape((b_size, seq_len, hidden_dim))?;
        probe::boundary(&y, usize::MAX, "routed_end");
        let hook_nanos = hook_start.elapsed().as_nanos() as usize;
        let block_nanos = block_start.elapsed().as_nanos() as usize;
        moe_block_probe(block_nanos, hook_nanos);

        let shared_gate = candle_nn::ops::sigmoid(&self.shared_expert_gate.forward(&xs_flat)?)?
            .reshape((b_size, seq_len, 1))?;
        let shared_out = self
            .shared_expert
            .forward(xs)?
            .broadcast_mul(&shared_gate)?;
        probe::boundary(&shared_out, usize::MAX, "shared_end");
        y + shared_out
    }
}

/// F32 softmax + descending top-k selection with deterministic ascending-index tie-breaking
/// (+ optional renormalization).
/// Returns `(topk_weights: F32 [tokens, k], topk_ids: U32 [tokens, k])`.
pub(crate) fn router_softmax_topk(
    logits: &Tensor,
    top_k: usize,
    renormalize: bool,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        return naiwa_attn::qwen4_exp::router_softmax_topk(logits, top_k, renormalize);
    }
    let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
    let (rows, n_experts) = probs.dims2()?;
    let data = probs.flatten_all()?.to_vec1::<f32>()?;
    let mut out_vals = Vec::with_capacity(rows * top_k);
    let mut out_ids = Vec::with_capacity(rows * top_k);
    let mut scratch: Vec<(u32, f32)> = Vec::with_capacity(n_experts);
    for r in 0..rows {
        scratch.clear();
        let row = &data[r * n_experts..(r + 1) * n_experts];
        for (i, &v) in row.iter().enumerate() {
            scratch.push((i as u32, v));
        }
        scratch.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let sum: f32 = if renormalize {
            scratch[..top_k].iter().map(|(_, v)| *v).sum::<f32>().max(1e-30)
        } else {
            1.0
        };
        for &(id, v) in &scratch[..top_k] {
            out_ids.push(id);
            out_vals.push(v / sum);
        }
    }
    let vals = Tensor::from_vec(out_vals, (rows, top_k), logits.device())?;
    let ids = Tensor::from_vec(out_ids, (rows, top_k), logits.device())?;
    Ok((vals, ids))
}

fn moe_block_probe(block_nanos: usize, hook_nanos: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("MOE_DBG").is_some()) {
        return;
    }
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static BLOCK_NS: AtomicUsize = AtomicUsize::new(0);
    static HOOK_NS: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Relaxed) + 1;
    BLOCK_NS.fetch_add(block_nanos, Relaxed);
    HOOK_NS.fetch_add(hook_nanos, Relaxed);
    if n % 480 == 0 {
        let block = BLOCK_NS.load(Relaxed) as f64 / 1e6;
        let hook = HOOK_NS.load(Relaxed) as f64 / 1e6;
        eprintln!(
            "MoE probe: {n} calls, cumulative block {block:.1} ms (experts {hook:.1} ms), {:.3} ms/call block",
            block / n as f64 * 1000.0,
        );
    }
}
