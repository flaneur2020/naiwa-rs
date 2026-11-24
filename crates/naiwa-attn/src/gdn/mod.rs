//! GatedDeltaNet (GDN) linear-attention layer & per-sequence recurrent state.

#[cfg(feature = "cuda")]
pub mod cuda;

use candle_core::{DType, Device, Result, Tensor, D};
use naiwa_quant::QLinear;

/// Output gate activation applied after RMSNorm in GDN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnGateActivation {
    Silu,
    Sigmoid,
}

/// How V-heads are grouped relative to K-heads in the checkpoint projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnVHeadLayout {
    Grouped,
    Tiled,
}

/// Hyperparameters for one GatedDeltaNet linear-attention layer.
#[derive(Clone, Debug)]
pub struct GdnConfig {
    pub hidden_size: usize,
    pub rms_norm_eps: f64,
    pub conv_kernel_dim: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub v_head_layout: GdnVHeadLayout,
    pub gate_activation: GdnGateActivation,
}

impl GdnConfig {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Per-sequence recurrent state for one GDN layer (supports clone/restore for MTP rollback).
#[derive(Clone, Debug)]
pub struct GdnSeqState {
    /// `[1, conv_dim, conv_kernel_dim]` in activation dtype (BF16/F16/F32).
    pub conv_state: Tensor,
    /// `[1, num_v_heads, head_v_dim, head_k_dim]` (value-major) in F32 (or BF16).
    pub recurrent_state: Tensor,
    /// True when `recurrent_state` is stored value-major `[1, H_v, V, K]`.
    pub value_major: bool,
}

impl GdnSeqState {
    pub fn new(
        cfg: &GdnConfig,
        act_dtype: DType,
        state_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let value_major = device.is_cuda();
        let conv_state = Tensor::zeros((1, cfg.conv_dim(), cfg.conv_kernel_dim), act_dtype, device)?;
        let recurrent_state = if value_major {
            Tensor::zeros(
                (1, cfg.num_v_heads, cfg.head_v_dim, cfg.head_k_dim),
                state_dtype,
                device,
            )?
        } else {
            Tensor::zeros(
                (1, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim),
                state_dtype,
                device,
            )?
        };
        Ok(Self {
            conv_state,
            recurrent_state,
            value_major,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        Ok(())
    }

    /// Deep-copy snapshot (used for MTP speculative verify rollback).
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            conv_state: self.conv_state.copy()?,
            recurrent_state: self.recurrent_state.copy()?,
            value_major: self.value_major,
        })
    }
}

/// Single-device GatedDeltaNet linear-attention layer (Qwen3.8-Flash-Next split-projection layout).
pub struct GatedDeltaNet {
    pub cfg: GdnConfig,
    /// `[conv_dim, hidden]` (or merged QKV projection `in_proj_qkv`).
    pub in_proj_qkv: QLinear,
    /// `[value_dim, hidden]` output gate `z` projection (`in_proj_z`).
    pub in_proj_z: QLinear,
    /// `[num_v_heads, hidden]` beta projection (`in_proj_b`).
    pub in_proj_b: QLinear,
    /// `[num_v_heads, hidden]` alpha/decay projection (`in_proj_a`).
    pub in_proj_a: QLinear,
    /// `[conv_dim, conv_kernel_dim]` depthwise conv1d weight (in activation dtype).
    pub conv_weight: Tensor,
    /// `[num_v_heads]` F32 `A_log`.
    pub a_log: Tensor,
    /// `[num_v_heads]` F32 `dt_bias`.
    pub dt_bias: Tensor,
    /// `[head_v_dim]` gated RMSNorm weight.
    pub norm_weight: Tensor,
    /// `[hidden, value_dim]` output projection (`out_proj`).
    pub out_proj: QLinear,
}

impl GatedDeltaNet {
    /// Forward pass for a single sequence `[1, seq_len, hidden]`, updating `state` in place.
    pub fn forward_seq(&self, xs: &Tensor, state: &mut GdnSeqState) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = xs.dims3()?;
        if batch != 1 {
            candle_core::bail!("GatedDeltaNet::forward_seq expects batch=1, got {batch}");
        }
        let dtype = xs.dtype();
        let qkv_raw = self.in_proj_qkv.forward(xs)?; // [1, S, conv_dim]
        let z = self.in_proj_z.forward(xs)?.reshape((
            1,
            seq_len,
            self.cfg.num_v_heads,
            self.cfg.head_v_dim,
        ))?;
        let b = self.in_proj_b.forward(xs)?; // [1, S, H_v]
        let a = self.in_proj_a.forward(xs)?; // [1, S, H_v]

        let tiled = self.cfg.v_head_layout == GdnVHeadLayout::Tiled;
        let sigmoid_gate = self.cfg.gate_activation == GdnGateActivation::Sigmoid;

        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() && matches!(dtype, DType::BF16 | DType::F16) {
            let mixed_qkv =
                cuda::causal_conv1d_cuda(&qkv_raw, &self.conv_weight, &mut state.conv_state)?;
            let core_out = if seq_len == 1 {
                cuda::decode_recurrence_cuda(
                    &mixed_qkv,
                    &b,
                    &a,
                    &self.a_log,
                    &self.dt_bias,
                    &mut state.recurrent_state,
                    self.cfg.num_k_heads,
                    self.cfg.num_v_heads,
                    self.cfg.head_k_dim,
                    self.cfg.head_v_dim,
                    tiled,
                    state.value_major,
                )?
            } else {
                cuda::prefill_recurrence_cuda(
                    &mixed_qkv,
                    &b,
                    &a,
                    &self.a_log,
                    &self.dt_bias,
                    &mut state.recurrent_state,
                    self.cfg.num_k_heads,
                    self.cfg.num_v_heads,
                    self.cfg.head_k_dim,
                    self.cfg.head_v_dim,
                    tiled,
                    state.value_major,
                )?
            };
            let normed = cuda::rmsnorm_gated_cuda(
                &core_out,
                &z,
                &self.norm_weight,
                self.cfg.rms_norm_eps,
                sigmoid_gate,
            )?;
            let flat = normed.reshape((1, seq_len, self.cfg.value_dim()))?;
            return self.out_proj.forward(&flat);
        }

        // Pure candle CPU / F32 reference fallback
        let mixed_qkv = causal_conv1d_ref(&qkv_raw, &self.conv_weight, &mut state.conv_state)?;
        let core_out = recurrence_ref(
            &mixed_qkv,
            &b,
            &a,
            &self.a_log,
            &self.dt_bias,
            &mut state.recurrent_state,
            &self.cfg,
            state.value_major,
        )?;
        let normed = rmsnorm_gated_ref(
            &core_out,
            &z,
            &self.norm_weight,
            self.cfg.rms_norm_eps,
            sigmoid_gate,
        )?;
        let flat = normed.reshape((1, seq_len, self.cfg.value_dim()))?;
        self.out_proj.forward(&flat)
    }
}

fn causal_conv1d_ref(x: &Tensor, weight: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let (_b, s, _d) = x.dims3()?;
    let (_, k) = weight.dims2()?;
    // state: [B, D, K], x_t: [B, D, S]
    let x_t = x.transpose(1, 2)?.to_dtype(DType::F32)?;
    let st_f32 = conv_state.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let combined = Tensor::cat(&[&st_f32, &x_t], 2)?; // [B, D, K + S]
    // New state = last K positions
    *conv_state = combined
        .narrow(2, s, k)?
        .contiguous()?
        .to_dtype(orig_dtype)?;
    // Depthwise causal conv: for each output pos t in 0..S, dot combined[:, :, t+1 .. t+1+K] with w
    let mut outs = Vec::with_capacity(s);
    for t in 0..s {
        let win = combined.narrow(2, t + 1, k)?; // [B, D, K]
        let y = win.broadcast_mul(&w_f32)?.sum(2)?; // [B, D]
        outs.push(y);
    }
    let out = Tensor::stack(&outs, 1)?; // [B, S, D]
    candle_nn::ops::silu(&out)?.to_dtype(orig_dtype)
}

#[allow(clippy::too_many_arguments)]
fn recurrence_ref(
    mixed_qkv: &Tensor,
    b_proj: &Tensor,
    a_proj: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    cfg: &GdnConfig,
    value_major: bool,
) -> Result<Tensor> {
    let orig_dtype = mixed_qkv.dtype();
    let state_dtype = state.dtype();
    let dev = mixed_qkv.device();
    let (_, s, _) = mixed_qkv.dims3()?;
    let kd = cfg.key_dim();
    let vd = cfg.value_dim();
    let hk = cfg.num_k_heads;
    let hv = cfg.num_v_heads;
    let dk = cfg.head_k_dim;
    let dv = cfg.head_v_dim;
    let rep = hv / hk;

    let mq = mixed_qkv
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let bv = b_proj.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let av = a_proj.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let al = a_log.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let dtb = dt_bias.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut st = state
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let scale = 1.0f32 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; s * hv * dv];

    for t in 0..s {
        let row = &mq[t];
        let q_all = &row[0..kd];
        let k_all = &row[kd..2 * kd];
        let v_all = &row[2 * kd..2 * kd + vd];

        for h in 0..hv {
            let kh = match cfg.v_head_layout {
                GdnVHeadLayout::Tiled => h % hk,
                GdnVHeadLayout::Grouped => h / rep,
            };
            let q_raw = &q_all[kh * dk..(kh + 1) * dk];
            let k_raw = &k_all[kh * dk..(kh + 1) * dk];
            let v_raw = &v_all[h * dv..(h + 1) * dv];

            let q_norm = (q_raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let k_norm = (k_raw.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();

            let x_a = av[t][h] + dtb[h];
            let softplus = if x_a > 20.0 {
                x_a
            } else {
                (1.0 + x_a.exp()).ln()
            };
            let g = (-al[h].exp() * softplus).exp();
            let beta = 1.0 / (1.0 + (-bv[t][h]).exp());

            let head_st = &mut st[h * dk * dv..(h + 1) * dk * dv];
            let out_row = &mut out[(t * hv + h) * dv..(t * hv + h + 1) * dv];

            for vi in 0..dv {
                let mut kv_mem = 0.0f32;
                for ki in 0..dk {
                    let idx = if value_major {
                        vi * dk + ki
                    } else {
                        ki * dv + vi
                    };
                    head_st[idx] *= g;
                    kv_mem += head_st[idx] * (k_raw[ki] / k_norm);
                }
                let delta = (v_raw[vi] - kv_mem) * beta;
                let mut acc = 0.0f32;
                for ki in 0..dk {
                    let idx = if value_major {
                        vi * dk + ki
                    } else {
                        ki * dv + vi
                    };
                    head_st[idx] += (k_raw[ki] / k_norm) * delta;
                    acc += head_st[idx] * (q_raw[ki] / q_norm) * scale;
                }
                out_row[vi] = acc;
            }
        }
    }

    *state = Tensor::from_vec(st, state.shape().clone(), dev)?.to_dtype(state_dtype)?;
    Tensor::from_vec(out, (1, s, hv, dv), dev)?.to_dtype(orig_dtype)
}

fn rmsnorm_gated_ref(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
    sigmoid_gate: bool,
) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let g_f32 = gate.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let d = x_f32.dim(D::Minus1)? as f64;
    let var = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / d)?;
    let normed = x_f32.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(&w_f32)?;
    let act_g = if sigmoid_gate {
        candle_nn::ops::sigmoid(&g_f32)?
    } else {
        candle_nn::ops::silu(&g_f32)?
    };
    normed.mul(&act_g)?.to_dtype(orig_dtype)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn gdn_cuda_prefill_and_decode_match_cpu_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let cfg = GdnConfig {
            hidden_size: 256,
            rms_norm_eps: 1e-6,
            conv_kernel_dim: 4,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 128,
            head_v_dim: 128,
            v_head_layout: GdnVHeadLayout::Tiled,
            gate_activation: GdnGateActivation::Sigmoid,
        };
        let conv_dim = cfg.conv_dim();
        let x_pre = Tensor::randn(0f32, 0.2, (1, 6, conv_dim), &dev)?.to_dtype(DType::BF16)?;
        let x_dec = Tensor::randn(0f32, 0.2, (1, 1, conv_dim), &dev)?.to_dtype(DType::BF16)?;
        let w_conv = Tensor::randn(0f32, 0.2, (conv_dim, 4), &dev)?.to_dtype(DType::BF16)?;
        let b_pre = Tensor::randn(0f32, 0.5, (1, 6, cfg.num_v_heads), &dev)?.to_dtype(DType::BF16)?;
        let a_pre = Tensor::randn(0f32, 0.5, (1, 6, cfg.num_v_heads), &dev)?.to_dtype(DType::BF16)?;
        let b_dec = Tensor::randn(0f32, 0.5, (1, 1, cfg.num_v_heads), &dev)?.to_dtype(DType::BF16)?;
        let a_dec = Tensor::randn(0f32, 0.5, (1, 1, cfg.num_v_heads), &dev)?.to_dtype(DType::BF16)?;
        let a_log = Tensor::randn(-1f32, 0.2, (cfg.num_v_heads,), &dev)?;
        let dt_bias = Tensor::randn(0f32, 0.2, (cfg.num_v_heads,), &dev)?;
        let z_dec = Tensor::randn(0f32, 0.5, (1, 1, cfg.num_v_heads, cfg.head_v_dim), &dev)?.to_dtype(DType::BF16)?;
        let nw = Tensor::ones((cfg.head_v_dim,), DType::BF16, &dev)?;

        let mut st_cuda = GdnSeqState::new(&cfg, DType::BF16, DType::F32, &dev)?;
        let mut st_ref = st_cuda.snapshot()?;

        // Prefill (6 tokens)
        let c_cuda = cuda::causal_conv1d_cuda(&x_pre, &w_conv, &mut st_cuda.conv_state)?;
        let c_ref = causal_conv1d_ref(&x_pre, &w_conv, &mut st_ref.conv_state)?;
        let o_cuda = cuda::prefill_recurrence_cuda(
            &c_cuda, &b_pre, &a_pre, &a_log, &dt_bias, &mut st_cuda.recurrent_state,
            cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, true, true,
        )?;
        let o_ref = recurrence_ref(
            &c_ref, &b_pre, &a_pre, &a_log, &dt_bias, &mut st_ref.recurrent_state, &cfg, true,
        )?;
        let diff_pre = (o_cuda.to_dtype(DType::F32)? - o_ref.to_dtype(DType::F32)?)?
            .abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff_pre < 5e-2, "GDN prefill CUDA vs ref max diff = {diff_pre}");

        // Decode (1 token) + rmsnorm_gated
        let st_c_diff = (st_cuda.conv_state.to_dtype(DType::F32)? - st_ref.conv_state.to_dtype(DType::F32)?)?
            .abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(st_c_diff < 1e-2, "GDN conv_state diff = {st_c_diff}");
        let c_cuda = cuda::causal_conv1d_cuda(&x_dec, &w_conv, &mut st_cuda.conv_state)?;
        let c_ref = causal_conv1d_ref(&x_dec, &w_conv, &mut st_ref.conv_state)?;
        let c_diff = (c_cuda.to_dtype(DType::F32)? - c_ref.to_dtype(DType::F32)?)?
            .abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(c_diff < 1e-2, "GDN conv decode diff = {c_diff}");
        let d_cuda = cuda::decode_recurrence_cuda(
            &c_cuda, &b_dec, &a_dec, &a_log, &dt_bias, &mut st_cuda.recurrent_state,
            cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, true, true,
        )?;
        let d_ref = recurrence_ref(
            &c_ref, &b_dec, &a_dec, &a_log, &dt_bias, &mut st_ref.recurrent_state, &cfg, true,
        )?;
        let d_diff = (d_cuda.to_dtype(DType::F32)? - d_ref.to_dtype(DType::F32)?)?
            .abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(d_diff < 1e-2, "GDN decode recurrence CUDA vs ref max diff = {d_diff}");
        // Verify rmsnorm_gated CUDA vs ref on identical input (since tiny BF16 recurrence delta
        // is amplified 1/rms when norming near-zero random outputs).
        let n_cuda = cuda::rmsnorm_gated_cuda(&d_cuda, &z_dec, &nw, cfg.rms_norm_eps, true)?;
        let n_ref = rmsnorm_gated_ref(&d_cuda, &z_dec, &nw, cfg.rms_norm_eps, true)?;
        let diff_norm = (n_cuda.to_dtype(DType::F32)? - n_ref.to_dtype(DType::F32)?)?
            .abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff_norm < 2e-2, "GDN rmsnorm_gated CUDA vs ref max diff = {diff_norm}");
        Ok(())
    }
}
