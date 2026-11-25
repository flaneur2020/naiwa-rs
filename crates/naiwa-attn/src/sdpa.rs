//! Scaled Dot-Product Attention (SDPA) with GQA and causal/offset masking.

use candle_core::{DType, Result, Tensor};

/// Parameters controlling SDPA scaling and head grouping.
#[derive(Clone, Copy, Debug)]
pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softmax_scale: f32,
    pub softcap: Option<f32>,
}

/// Repeat KV heads `[b, n_kv_heads, s, d]` to `[b, n_kv_heads * n_rep, s, d]`.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv_heads, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv_heads, n_rep, s, d))?
        .reshape((b, n_kv_heads * n_rep, s, d))
}

/// Build an additive causal mask `[1, 1, q_len, kv_len]` (in F32: `0.0` allowed, `-inf` masked)
/// where query `i` attends to keys `0 ..= (kv_len - q_len) + i`.
pub fn causal_mask(q_len: usize, kv_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let offset = kv_len.saturating_sub(q_len);
    let mut data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_k = offset + i;
        for j in (max_k + 1)..kv_len {
            data[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)
}

/// Compute scaled dot-product attention for:
/// - `q`: `[b, n_heads, q_len, d]`
/// - `k`: `[b, n_kv_heads, kv_len, d]`
/// - `v`: `[b, n_kv_heads, kv_len, d]`
/// Returns `[b, n_heads, q_len, d]` in `q.dtype()`.
pub fn sdpa_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    params: SdpaParams,
) -> Result<Tensor> {
    let orig_dtype = q.dtype();
    let (_, _, q_len, _) = q.dims4()?;
    let (_, _, kv_len, _) = k.dims4()?;

    let k = repeat_kv(k.clone(), params.n_kv_groups)?.contiguous()?;
    let v = repeat_kv(v.clone(), params.n_kv_groups)?.contiguous()?;
    let q_c = q.contiguous()?;

    // Compute attention scores in F32 for numerical parity across BF16/F16/F32.
    let scores = (q_c
        .to_dtype(DType::F32)?
        .matmul(&k.to_dtype(DType::F32)?.t()?)?
        * (params.softmax_scale as f64))?;

    let scores = if let Some(cap) = params.softcap {
        ((scores / (cap as f64))?.tanh()? * (cap as f64))?
    } else {
        scores
    };

    let scores = if let Some(m) = mask {
        let m = if m.dtype() != DType::F32 {
            m.to_dtype(DType::F32)?
        } else {
            m.clone()
        };
        scores.broadcast_add(&m)?
    } else if causal && q_len > 1 {
        let m = causal_mask(q_len, kv_len, q.device())?;
        scores.broadcast_add(&m)?
    } else {
        scores
    };

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v.to_dtype(DType::F32)?)?;
    out.to_dtype(orig_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn sdpa_causal_single_and_multi_token() {
        let q = Tensor::ones((1, 4, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::from_vec(
            (0..2 * 3 * 8).map(|i| i as f32 * 0.1).collect(),
            (1, 2, 3, 8),
            &Device::Cpu,
        )
        .unwrap();
        let out = sdpa_forward(
            &q,
            &k,
            &v,
            None,
            true,
            SdpaParams {
                n_kv_groups: 2,
                softmax_scale: 1.0 / (8.0f32).sqrt(),
                softcap: None,
            },
        )
        .unwrap();
        assert_eq!(out.dims(), &[1, 4, 3, 8]);
    }
}
