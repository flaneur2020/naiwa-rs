//! 1D text Rotary Positional Embedding (RoPE) with partial rotary factor support.

use candle_core::{DType, Device, Result, Tensor, D};

/// Text-only rotary embedding for Qwen3.8-Flash-Next (supports `rotary_dim <= head_dim`).
#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    rotary_dim: usize,
    attention_factor: f32,
}

impl RotaryEmbedding {
    /// Construct standard RoPE over `rotary_dim` (even) with base `rope_theta`.
    pub fn new(rope_theta: f64, rotary_dim: usize, device: &Device) -> Result<Self> {
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0f64 / rope_theta.powf((2 * i) as f64 / rotary_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;
        Ok(Self {
            inv_freq,
            rotary_dim,
            attention_factor: 1.0,
        })
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub fn half_rotary_dim(&self) -> usize {
        self.rotary_dim / 2
    }

    /// Compute `(cos, sin)` of shape `[batch, seq_len, rotary_dim / 2]` in `dtype` for
    /// `position_ids` of shape `[batch, seq_len]` (or `[seq_len]`).
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let pos = position_ids.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let inv = self.inv_freq.reshape((1, self.inv_freq.dim(0)?))?;
        let freqs = pos.broadcast_mul(&inv)?;
        let mut cos = freqs.cos()?;
        let mut sin = freqs.sin()?;
        if self.attention_factor != 1.0 {
            cos = (cos * self.attention_factor as f64)?;
            sin = (sin * self.attention_factor as f64)?;
        }
        Ok((
            cos.to_dtype(dtype)?.contiguous()?,
            sin.to_dtype(dtype)?.contiguous()?,
        ))
    }

    /// Apply NeoX-style rotary embedding to `x` of shape `[batch, heads, seq_len, head_dim]`
    /// using `(cos, sin)` of shape `[batch, seq_len, rotary_dim / 2]` (or `[seq_len, rotary_dim / 2]`).
    ///
    /// When `rotary_dim < head_dim` (partial RoPE), only the first `rotary_dim` channels are rotated
    /// and the remaining pass-through channels are concatenated unchanged.
    pub fn apply_bhsd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, _h, s, d) = x.dims4()?;
        let orig_dtype = x.dtype();
        let rot = self.rotary_dim;
        let half = rot / 2;
        let cos = match cos.rank() {
            2 => cos.unsqueeze(0)?.broadcast_as((b, s, half))?.contiguous()?,
            3 => cos.contiguous()?,
            r => candle_core::bail!("RotaryEmbedding::apply_bhsd unexpected cos rank {r}"),
        }
        .to_dtype(DType::F32)?
        .unsqueeze(1)?; // [b, 1, s, rot/2]
        let sin = match sin.rank() {
            2 => sin.unsqueeze(0)?.broadcast_as((b, s, half))?.contiguous()?,
            3 => sin.contiguous()?,
            r => candle_core::bail!("RotaryEmbedding::apply_bhsd unexpected sin rank {r}"),
        }
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
        let xf = x.to_dtype(DType::F32)?;
        let x1 = xf.narrow(D::Minus1, 0, half)?;
        let x2 = xf.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let out = if rot == d {
            Tensor::cat(&[&r1, &r2], D::Minus1)?
        } else {
            let x_pass = xf.narrow(D::Minus1, rot, d - rot)?;
            Tensor::cat(&[&r1, &r2, &x_pass], D::Minus1)?
        };
        out.to_dtype(orig_dtype)
    }

    /// Apply NeoX-style partial RoPE to `x` of shape `[batch, seq_len, heads, head_dim]`,
    /// with `(cos, sin)` of shape `[batch, seq_len, rotary_dim / 2]`.
    pub fn apply_bshd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let half = cos.dim(D::Minus1)?;
        let d = x.dim(D::Minus1)?;
        let cos = cos.to_dtype(DType::F32)?.unsqueeze(2)?;
        let sin = sin.to_dtype(DType::F32)?.unsqueeze(2)?;
        let xf = x.to_dtype(DType::F32)?;
        let x1 = xf.narrow(D::Minus1, 0, half)?;
        let x2 = xf.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        let out = if d > 2 * half {
            Tensor::cat(&[&r1, &r2, &xf.narrow(D::Minus1, 2 * half, d - 2 * half)?], D::Minus1)?
        } else {
            Tensor::cat(&[&r1, &r2], D::Minus1)?
        };
        out.to_dtype(orig_dtype)
    }

    /// Apply per-head RMSNorm + partial NeoX RoPE to `q` and `k` (both `[batch, heads, seq_len, head_dim]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qk_norm(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        q_eps: f64,
        k_eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let qn = candle_nn::ops::rms_norm(&q.contiguous()?, q_weight, q_eps as f32)?;
        let kn = candle_nn::ops::rms_norm(&k.contiguous()?, k_weight, k_eps as f32)?;
        let qr = self.apply_bhsd(&qn, cos, sin)?;
        let kr = self.apply_bhsd(&kn, cos, sin)?;
        Ok((qr, kr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rope_leaves_tail_unchanged() {
        let rope = RotaryEmbedding::new(10000.0, 4, &Device::Cpu).unwrap();
        let pos = Tensor::new(&[[0u32, 1u32]], &Device::Cpu).unwrap();
        let (cos, sin) = rope.compute_cos_sin(&pos, DType::F32).unwrap();
        let x = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // pos 0
                1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, // pos 1
            ],
            (1, 1, 2, 8),
            &Device::Cpu,
        )
        .unwrap();
        let y = rope.apply_bhsd(&x, &cos, &sin).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Pos 0: cos=1, sin=0 -> identical
        assert!((yv[0] - 1.0).abs() < 1e-5);
        assert!((yv[3] - 4.0).abs() < 1e-5);
        // Tail channels untouched at both positions
        assert_eq!(&yv[4..8], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&yv[12..16], &[9.0, 10.0, 11.0, 12.0]);
    }
}
