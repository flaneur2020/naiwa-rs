//! Clean CUDA kernel launchers for GatedDeltaNet (`kernels/gdn/gdn.cu`).

#![allow(dead_code)]

use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};
use core::ffi::c_void;

pub const GDN_STATE_DTYPE_F16: i32 = 0;
pub const GDN_STATE_DTYPE_BF16: i32 = 1;
pub const GDN_STATE_DTYPE_F32: i32 = 2;

pub const GDN_DECODE_KERNEL_COOPERATIVE: i32 = 1;
pub const GDN_DECODE_KERNEL_PIPELINED: i32 = 2;
pub const GDN_DECODE_KERNEL_VALUE_MAJOR_4: i32 = 3;
pub const GDN_DECODE_KERNEL_VALUE_MAJOR_32: i32 = 4;

extern "C" {
    #[link_name = "naiwa_causal_conv1d_update"]
    fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_causal_conv1d_full"]
    fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_in: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_fused_gdn_gating"]
    fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_gdn_prepare_recurrence"]
    fn gdn_prepare_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        g_out: *mut f32,
        beta_out: *mut f32,
        batch_size: i32,
        seq_len: i32,
        token_stride: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        dtype: i32,
        out_bf16: i32,
        stream: i64,
    );

    #[link_name = "naiwa_gdn_decode_recurrence"]
    fn gdn_decode_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        b_batch_stride: i64,
        b_head_stride: i64,
        a_batch_stride: i64,
        a_head_stride: i64,
        slot_indices: *const i32,
        kernel_kind: i32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_gated_delta_rule_recurrence"]
    fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_chunked_gated_delta_rule_recurrence"]
    fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_vmajor_warp_gated_delta_rule_recurrence"]
    fn vmajor_warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    #[link_name = "naiwa_vmajor_chunked_gated_delta_rule_recurrence"]
    fn vmajor_chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    ) -> i32;

    #[link_name = "naiwa_gdn_rmsnorm_gated"]
    fn gdn_rmsnorm_gated(
        x: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden_dim: i32,
        outer_dim_1: i32,
        outer_dim_2: i32,
        x_stride_0: i64,
        x_stride_1: i64,
        x_stride_2: i64,
        x_stride_3: i64,
        gate_stride_0: i64,
        gate_stride_1: i64,
        gate_stride_2: i64,
        gate_stride_3: i64,
        eps: f32,
        sigmoid_gate: i32,
        dtype: i32,
        stream: i64,
    );
}

fn act_dtype_code(dt: DType) -> Result<i32> {
    match dt {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        other => candle_core::bail!("gdn cuda kernel expects F16 or BF16 activations, got {other:?}"),
    }
}

fn state_dtype_code(dt: DType) -> Result<i32> {
    match dt {
        DType::F16 => Ok(GDN_STATE_DTYPE_F16),
        DType::BF16 => Ok(GDN_STATE_DTYPE_BF16),
        DType::F32 => Ok(GDN_STATE_DTYPE_F32),
        other => candle_core::bail!("gdn cuda kernel unsupported state dtype {other:?}"),
    }
}

fn raw_ptr(
    t: &Tensor,
    stream: &candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<*const c_void> {
    let (s, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*s else {
        candle_core::bail!("gdn cuda expects CUDA tensor");
    };
    let (p, _g) = match t.dtype() {
        DType::F16 => {
            naiwa_quant::utils::slice_ptr_on_stream(c.as_cuda_slice::<half::f16>()?, l.start_offset(), stream)
        }
        DType::BF16 => {
            naiwa_quant::utils::slice_ptr_on_stream(c.as_cuda_slice::<half::bf16>()?, l.start_offset(), stream)
        }
        DType::F32 => {
            naiwa_quant::utils::slice_ptr_on_stream(c.as_cuda_slice::<f32>()?, l.start_offset(), stream)
        }
        other => candle_core::bail!("gdn raw_ptr unsupported dtype {other:?}"),
    };
    Ok(p as *const c_void)
}

fn raw_ptr_f32(
    t: &Tensor,
    stream: &candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<*const f32> {
    let (s, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*s else {
        candle_core::bail!("gdn cuda expects CUDA f32 tensor");
    };
    let (p, _g) =
        naiwa_quant::utils::slice_ptr_on_stream(c.as_cuda_slice::<f32>()?, l.start_offset(), stream);
    Ok(p as *const f32)
}

fn alloc_like_dtype(
    dev: &candle_core::CudaDevice,
    shape: Shape,
    dtype: DType,
) -> Result<Tensor> {
    let n = shape.elem_count();
    let storage = match dtype {
        DType::F16 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<half::f16>(n)? }, dev.clone()),
        DType::BF16 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<half::bf16>(n)? }, dev.clone()),
        DType::F32 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<f32>(n)? }, dev.clone()),
        other => candle_core::bail!("gdn alloc unsupported dtype {other:?}"),
    };
    Ok(Tensor::from((Storage::Cuda(storage), shape)))
}

/// Causal depthwise 1D convolution with SiLU activation (`x`: `[B, S, conv_dim]`,
/// `weight`: `[conv_dim, kernel_size]`, `conv_state`: `[B, conv_dim, kernel_size]` updated in place).
pub fn causal_conv1d_cuda(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let (b, s, conv_dim) = x.dims3()?;
    let (_, kernel_size) = weight.dims2()?;
    let dtype = x.dtype();
    let dt_code = act_dtype_code(dtype)?;
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream() as i64;

    let out = alloc_like_dtype(dev, Shape::from((b, s, conv_dim)), dtype)?;
    let x_stride_b = (s * conv_dim) as i64;
    let x_stride_s = conv_dim as i64;
    let x_stride_c = 1i64;

    if s == 1 {
        let xp = raw_ptr(&x, &stream)?;
        let wp = raw_ptr(&weight, &stream)?;
        let sp = raw_ptr(conv_state, &stream)?;
        let op = raw_ptr(&out, &stream)?;
        unsafe {
            causal_conv1d_update(
                xp,
                wp,
                sp as *mut c_void,
                op as *mut c_void,
                b as i32,
                conv_dim as i32,
                kernel_size as i32,
                x_stride_b,
                x_stride_s,
                x_stride_c,
                std::ptr::null(),
                dt_code,
                cu_stream,
            );
        }
    } else {
        let new_state = alloc_like_dtype(dev, conv_state.shape().clone(), dtype)?;
        {
            let xp = raw_ptr(&x, &stream)?;
            let wp = raw_ptr(&weight, &stream)?;
            let sin_p = raw_ptr(conv_state, &stream)?;
            let sout_p = raw_ptr(&new_state, &stream)?;
            let op = raw_ptr(&out, &stream)?;
            unsafe {
                causal_conv1d_full(
                    xp,
                    wp,
                    sin_p,
                    sout_p as *mut c_void,
                    op as *mut c_void,
                    b as i32,
                    conv_dim as i32,
                    s as i32,
                    kernel_size as i32,
                    x_stride_b,
                    x_stride_s,
                    x_stride_c,
                    std::ptr::null(),
                    dt_code,
                    cu_stream,
                );
            }
        }
        *conv_state = new_state;
    }

    Ok(out)
}

/// Fused single-token decode recurrence directly on `mixed_qkv` (`[B, 1, conv_dim]`),
/// `b` (`[B, 1, H_v]`), `a` (`[B, 1, H_v]`), updating `state` (`[B, H_v, V, K]` value-major
/// or `[B, H_v, K, V]` key-major) in place and returning `[B, 1, H_v, V]` in `mixed_qkv.dtype()`.
#[allow(clippy::too_many_arguments)]
pub fn decode_recurrence_cuda(
    mixed_qkv: &Tensor,
    b_proj: &Tensor,
    a_proj: &Tensor,
    a_log_f32: &Tensor,
    dt_bias_f32: &Tensor,
    state: &mut Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
    value_major_state: bool,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let b_proj = b_proj.contiguous()?;
    let a_proj = a_proj.contiguous()?;
    let a_log = a_log_f32.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias = dt_bias_f32.to_dtype(DType::F32)?.contiguous()?;

    let batch = mixed_qkv.dim(0)?;
    let dtype = mixed_qkv.dtype();
    let dt_code = act_dtype_code(dtype)?;
    let st_code = state_dtype_code(state.dtype())?;
    let dev = mixed_qkv.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let out = alloc_like_dtype(dev, Shape::from((batch, 1, num_v_heads, head_v_dim)), dtype)?;
    let kernel_kind = if value_major_state && head_k_dim == 128 && head_v_dim == 128 {
        GDN_DECODE_KERNEL_VALUE_MAJOR_4
    } else if !value_major_state && head_k_dim == 128 && head_v_dim == 128 {
        GDN_DECODE_KERNEL_COOPERATIVE
    } else {
        0
    };

    {
        let qp = raw_ptr(&mixed_qkv, &stream)?;
        let bp = raw_ptr(&b_proj, &stream)?;
        let ap = raw_ptr(&a_proj, &stream)?;
        let alp = raw_ptr_f32(&a_log, &stream)?;
        let dtp = raw_ptr_f32(&dt_bias, &stream)?;
        let sp = raw_ptr(state, &stream)?;
        let op = raw_ptr(&out, &stream)?;
        unsafe {
            gdn_decode_recurrence(
                qp,
                bp,
                ap,
                alp,
                dtp,
                sp as *mut c_void,
                op as *mut c_void,
                batch as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                i32::from(tiled_v_heads),
                num_v_heads as i64,
                1,
                num_v_heads as i64,
                1,
                std::ptr::null(),
                kernel_kind,
                dt_code,
                st_code,
                stream.cu_stream() as i64,
            );
        }
    }
    Ok(out)
}

/// Multi-token prefill/verify recurrence on CUDA using `gdn_prepare_recurrence` +
/// `vmajor_chunked_gated_delta_rule_recurrence` / `chunked_gated_delta_rule_recurrence`.
///
/// Updates `state` (`[B, H_v, V, K]` if `value_major_state`, else `[B, H_v, K, V]`) in place
/// and returns `[B, S, H_v, V]` in `mixed_qkv.dtype()`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_recurrence_cuda(
    mixed_qkv: &Tensor,
    b_proj: &Tensor,
    a_proj: &Tensor,
    a_log_f32: &Tensor,
    dt_bias_f32: &Tensor,
    state: &mut Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
    value_major_state: bool,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let b_proj = b_proj.contiguous()?;
    let a_proj = a_proj.contiguous()?;
    let a_log = a_log_f32.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias = dt_bias_f32.to_dtype(DType::F32)?.contiguous()?;

    let (batch, seq_len, _conv_dim) = mixed_qkv.dims3()?;
    let token_stride = seq_len;
    let dtype = mixed_qkv.dtype();
    let dt_code = act_dtype_code(dtype)?;
    let st_code = state_dtype_code(state.dtype())?;
    let dev = mixed_qkv.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream() as i64;
    let bh = batch * num_v_heads;

    // 1. Prepare F32 q/k/v/g/beta laid out as [B*H_v, S, dim]
    let q_f32 = alloc_like_dtype(dev, Shape::from((bh, seq_len, head_k_dim)), DType::F32)?;
    let k_f32 = alloc_like_dtype(dev, Shape::from((bh, seq_len, head_k_dim)), DType::F32)?;
    let v_f32 = alloc_like_dtype(dev, Shape::from((bh, seq_len, head_v_dim)), DType::F32)?;
    let g_f32 = alloc_like_dtype(dev, Shape::from((bh, seq_len)), DType::F32)?;
    let beta_f32 = alloc_like_dtype(dev, Shape::from((bh, seq_len)), DType::F32)?;

    {
        let mqp = raw_ptr(&mixed_qkv, &stream)?;
        let bp = raw_ptr(&b_proj, &stream)?;
        let ap = raw_ptr(&a_proj, &stream)?;
        let alp = raw_ptr_f32(&a_log, &stream)?;
        let dtp = raw_ptr_f32(&dt_bias, &stream)?;
        let qp = raw_ptr_f32(&q_f32, &stream)?;
        let kp = raw_ptr_f32(&k_f32, &stream)?;
        let vp = raw_ptr_f32(&v_f32, &stream)?;
        let gp = raw_ptr_f32(&g_f32, &stream)?;
        let btp = raw_ptr_f32(&beta_f32, &stream)?;
        unsafe {
            gdn_prepare_recurrence(
                mqp,
                bp,
                ap,
                alp,
                dtp,
                qp as *mut c_void,
                kp as *mut c_void,
                vp as *mut c_void,
                gp as *mut f32,
                btp as *mut f32,
                batch as i32,
                seq_len as i32,
                token_stride as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                i32::from(tiled_v_heads),
                dt_code,
                0, // out_bf16 = 0 -> f32
                cu_stream,
            );
        }
    }

    // 2. Recurrence -> [B*H_v, S, V] in F32
    let out_bh = alloc_like_dtype(dev, Shape::from((bh, seq_len, head_v_dim)), DType::F32)?;
    {
        let qp = raw_ptr_f32(&q_f32, &stream)?;
        let kp = raw_ptr_f32(&k_f32, &stream)?;
        let vp = raw_ptr_f32(&v_f32, &stream)?;
        let gp = raw_ptr_f32(&g_f32, &stream)?;
        let btp = raw_ptr_f32(&beta_f32, &stream)?;
        let sp = raw_ptr(state, &stream)?;
        let op = raw_ptr_f32(&out_bh, &stream)?;
        unsafe {
            if value_major_state {
                let rc = if seq_len >= 16 {
                    vmajor_chunked_gated_delta_rule_recurrence(
                        qp,
                        kp,
                        vp,
                        gp,
                        btp,
                        sp as *mut c_void,
                        op as *mut f32,
                        bh as i32,
                        seq_len as i32,
                        head_k_dim as i32,
                        head_v_dim as i32,
                        std::ptr::null(),
                        num_v_heads as i32,
                        st_code,
                        cu_stream,
                    )
                } else {
                    0
                };
                if rc == 0 {
                    vmajor_warp_gated_delta_rule_recurrence(
                        qp,
                        kp,
                        vp,
                        gp,
                        btp,
                        sp as *mut c_void,
                        op as *mut f32,
                        bh as i32,
                        seq_len as i32,
                        head_k_dim as i32,
                        head_v_dim as i32,
                        std::ptr::null(),
                        num_v_heads as i32,
                        st_code,
                        cu_stream,
                    );
                }
            } else if seq_len >= 16 {
                chunked_gated_delta_rule_recurrence(
                    qp,
                    kp,
                    vp,
                    gp,
                    btp,
                    sp as *mut c_void,
                    op as *mut f32,
                    bh as i32,
                    seq_len as i32,
                    head_k_dim as i32,
                    head_v_dim as i32,
                    std::ptr::null(),
                    num_v_heads as i32,
                    st_code,
                    cu_stream,
                );
            } else {
                gated_delta_rule_recurrence(
                    qp,
                    kp,
                    vp,
                    gp,
                    btp,
                    sp as *mut c_void,
                    op as *mut f32,
                    bh as i32,
                    seq_len as i32,
                    head_k_dim as i32,
                    head_v_dim as i32,
                    std::ptr::null(),
                    num_v_heads as i32,
                    st_code,
                    cu_stream,
                );
            }
        }
    }

    // [B*H_v, S, V] -> [B, H_v, S, V] -> [B, S, H_v, V] in activation dtype
    out_bh
        .reshape((batch, num_v_heads, seq_len, head_v_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

/// Fused RMSNorm + SiLU/Sigmoid output gate: `x` & `gate` of shape `[B, S, H_v, V]`,
/// `weight` of shape `[V]`. Returns `[B, S, H_v, V]` in `x.dtype()`.
pub fn rmsnorm_gated_cuda(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
    sigmoid_gate: bool,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let gate = gate.contiguous()?;
    let weight = weight.to_dtype(x.dtype())?.contiguous()?;
    let (b, s, h, v) = x.dims4()?;
    let rows = b * s * h;
    let dtype = x.dtype();
    let dt_code = act_dtype_code(dtype)?;
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let out = alloc_like_dtype(dev, Shape::from((b, s, h, v)), dtype)?;
    let s0 = (s * h * v) as i64;
    let s1 = (h * v) as i64;
    let s2 = v as i64;
    let s3 = 1i64;

    {
        let xp = raw_ptr(&x, &stream)?;
        let gp = raw_ptr(&gate, &stream)?;
        let wp = raw_ptr(&weight, &stream)?;
        let op = raw_ptr(&out, &stream)?;
        unsafe {
            gdn_rmsnorm_gated(
                xp,
                gp,
                wp,
                op as *mut c_void,
                rows as i32,
                v as i32,
                s as i32,
                h as i32,
                s0,
                s1,
                s2,
                s3,
                s0,
                s1,
                s2,
                s3,
                eps as f32,
                i32::from(sigmoid_gate),
                dt_code,
                stream.cu_stream() as i64,
            );
        }
    }
    Ok(out)
}
