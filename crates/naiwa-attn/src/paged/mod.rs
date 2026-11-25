//! Single-device PagedAttention primitives (`reshape_and_cache`, `paged_attention_decode`,
//! `gather_kv_cache`, and `PagedAttention`).

#[cfg(feature = "cuda")]
mod ffi;

use candle_core::{DType, Device, Result, Tensor};

use crate::sdpa::{sdpa_forward, SdpaParams};

/// Per-step metadata describing paged KV blocks for a batch of sequences.
#[derive(Clone, Debug)]
pub struct PagedAttnMeta {
    /// `[num_tokens]` i64 slot index (`block_id * block_size + block_offset`) for every token.
    pub slot_mapping: Tensor,
    /// `[num_seqs, max_blocks_per_seq]` u32/i32 physical block IDs.
    pub block_tables: Tensor,
    /// `[num_seqs]` u32/i32 total context length (including current tokens) per sequence.
    pub context_lens: Tensor,
    /// `[num_seqs + 1]` u32/i32 cumulative context lengths (for `gather_kv_cache` on prefill).
    pub cu_context_lens: Tensor,
    /// Host-side context lengths per sequence (avoids D2H sync).
    pub host_context_lens: Vec<usize>,
    /// Host-side query lengths per sequence in this step.
    pub host_query_lens: Vec<usize>,
    /// Maximum context length across the batch.
    pub max_context_len: usize,
    /// True when every sequence in the batch has `query_len == 1` (single-token decode).
    pub is_decode: bool,
    /// Optional per-query-row `[total_q, max_blocks_per_seq]` block tables for fused multi-row
    /// `paged_attention_decode` (used by MTP verify/refresh when `total_q <= 8`).
    pub row_block_tables: Option<Tensor>,
    /// Optional per-query-row `[total_q]` causal context lengths paired with `row_block_tables`.
    pub row_context_lens: Option<Tensor>,
}

/// Number of elements packed in the innermost `x` dimension of the paged key cache (`16 / sizeof(T)`).
pub fn key_cache_x(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 8,
        _ => 8,
    }
}

/// Allocate one layer's `(key_cache, value_cache)` paged KV tensors:
/// - `key_cache`:   `[num_blocks, num_kv_heads, head_dim / x, block_size, x]`
/// - `value_cache`: `[num_blocks, num_kv_heads, head_dim, block_size]`
pub fn alloc_kv_cache(
    num_blocks: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let x = key_cache_x(dtype);
    let k = Tensor::zeros(
        (num_blocks, num_kv_heads, head_dim / x, block_size, x),
        dtype,
        device,
    )?;
    let v = Tensor::zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype, device)?;
    Ok((k, v))
}

#[cfg(feature = "cuda")]
fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("paged-attn: unsupported dtype {other:?}"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_storage_ptr<'a>(
    c: &'a candle_core::CudaStorage,
    offset: usize,
    dtype: DType,
    stream: &'a candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<(
    *const core::ffi::c_void,
    candle_core::cuda::cudarc::driver::SyncOnDrop<'a>,
)> {
    match dtype {
        DType::F16 => {
            let (p, g) = naiwa_quant::utils::slice_ptr_on_stream(
                c.as_cuda_slice::<half::f16>()?,
                offset,
                stream,
            );
            Ok((p as *const core::ffi::c_void, g))
        }
        DType::BF16 => {
            let (p, g) = naiwa_quant::utils::slice_ptr_on_stream(
                c.as_cuda_slice::<half::bf16>()?,
                offset,
                stream,
            );
            Ok((p as *const core::ffi::c_void, g))
        }
        DType::F32 => {
            let (p, g) = naiwa_quant::utils::slice_ptr_on_stream(
                c.as_cuda_slice::<f32>()?,
                offset,
                stream,
            );
            Ok((p as *const core::ffi::c_void, g))
        }
        other => candle_core::bail!("paged-attn unsupported dtype {other:?}"),
    }
}

/// Write `key` and `value` (`[num_tokens, num_kv_heads, head_dim]`) into the paged KV cache
/// at the positions specified by `slot_mapping` (`[num_tokens]` i64).
#[cfg(feature = "cuda")]
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::Storage;
    use core::ffi::{c_int, c_long};

    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let slot_mapping = slot_mapping.to_dtype(DType::I64)?.contiguous()?;
    let (num_tokens, num_kv_heads, head_dim) = key.dims3()?;
    if num_tokens == 0 {
        return Ok(());
    }
    let (_, _, _, block_size, x) = key_cache.dims5()?;
    let dt = dtype_code(key.dtype())?;
    let cdt = dtype_code(key_cache.dtype())?;
    let dev = key.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let key_stride = (num_kv_heads * head_dim) as c_int;
    let val_stride = (num_kv_heads * head_dim) as c_int;

    let (ks, kl) = key.storage_and_layout();
    let (vs, vl) = value.storage_and_layout();
    let (kcs, kcl) = key_cache.storage_and_layout();
    let (vcs, vcl) = value_cache.storage_and_layout();
    let (ss, sl) = slot_mapping.storage_and_layout();

    let (
        Storage::Cuda(k_c),
        Storage::Cuda(v_c),
        Storage::Cuda(kc_c),
        Storage::Cuda(vc_c),
        Storage::Cuda(s_c),
    ) = (&*ks, &*vs, &*kcs, &*vcs, &*ss)
    else {
        candle_core::bail!("reshape_and_cache requires CUDA tensors");
    };

    let (k_ptr, _kg) = cuda_storage_ptr(k_c, kl.start_offset(), key.dtype(), &stream)?;
    let (v_ptr, _vg) = cuda_storage_ptr(v_c, vl.start_offset(), value.dtype(), &stream)?;
    let (kc_ptr, _kcg) = cuda_storage_ptr(kc_c, kcl.start_offset(), key_cache.dtype(), &stream)?;
    let (vc_ptr, _vcg) = cuda_storage_ptr(vc_c, vcl.start_offset(), value_cache.dtype(), &stream)?;
    let (s_ptr, _sg) = s_c.as_cuda_slice::<i64>()?.device_ptr(&stream);
    let s_ptr = (s_ptr as *const c_long).wrapping_add(sl.start_offset());

    unsafe {
        ffi::reshape_and_cache(
            k_ptr,
            v_ptr,
            kc_ptr,
            vc_ptr,
            s_ptr,
            num_tokens as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            val_stride,
            stream.cu_stream(),
            dt,
            cdt,
            std::ptr::null(),
            std::ptr::null(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn reshape_and_cache(
    _key: &Tensor,
    _value: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _slot_mapping: &Tensor,
) -> Result<()> {
    candle_core::bail!("reshape_and_cache requires CUDA")
}

/// Gather contiguous `(k_out, v_out)` of shape `[total_tokens, num_kv_heads, head_dim]` from paged KV cache.
#[cfg(feature = "cuda")]
pub fn gather_kv_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    total_tokens: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    use candle_core::Storage;
    use core::ffi::c_int;

    let (_, num_kv_heads, head_over_x, block_size, x) = key_cache.dims5()?;
    let head_dim = head_over_x * x;
    if total_tokens == 0 {
        let k = Tensor::zeros((0, num_kv_heads, head_dim), out_dtype, key_cache.device())?;
        let v = Tensor::zeros((0, num_kv_heads, head_dim), out_dtype, key_cache.device())?;
        return Ok((k, v));
    }

    let bt = block_tables.to_dtype(DType::U32)?.contiguous()?;
    let cu = cu_seq_lens.to_dtype(DType::U32)?.contiguous()?;
    let (_, max_blocks) = bt.dims2()?;
    let num_seqs = cu.dims1()?.saturating_sub(1);

    let k_out = Tensor::zeros(
        (total_tokens, num_kv_heads, head_dim),
        out_dtype,
        key_cache.device(),
    )?;
    let v_out = Tensor::zeros(
        (total_tokens, num_kv_heads, head_dim),
        out_dtype,
        key_cache.device(),
    )?;

    let out_code = dtype_code(out_dtype)?;
    let cache_code = dtype_code(key_cache.dtype())?;
    let dev = key_cache.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    {
        let (kcs, kcl) = key_cache.storage_and_layout();
        let (vcs, vcl) = value_cache.storage_and_layout();
        let (kos, kol) = k_out.storage_and_layout();
        let (vos, vol) = v_out.storage_and_layout();
        let (bts, btl) = bt.storage_and_layout();
        let (cus, cul) = cu.storage_and_layout();

        let (
            Storage::Cuda(kc_c),
            Storage::Cuda(vc_c),
            Storage::Cuda(ko_c),
            Storage::Cuda(vo_c),
            Storage::Cuda(bt_c),
            Storage::Cuda(cu_c),
        ) = (&*kcs, &*vcs, &*kos, &*vos, &*bts, &*cus)
        else {
            candle_core::bail!("gather_kv_cache requires CUDA tensors");
        };

        let (kc_ptr, _kcg) = cuda_storage_ptr(kc_c, kcl.start_offset(), key_cache.dtype(), &stream)?;
        let (vc_ptr, _vcg) = cuda_storage_ptr(vc_c, vcl.start_offset(), value_cache.dtype(), &stream)?;
        let (ko_ptr, _kog) = cuda_storage_ptr(ko_c, kol.start_offset(), out_dtype, &stream)?;
        let (vo_ptr, _vog) = cuda_storage_ptr(vo_c, vol.start_offset(), out_dtype, &stream)?;
        let (bt_ptr, _btg) = naiwa_quant::utils::slice_ptr_on_stream(
            bt_c.as_cuda_slice::<u32>()?,
            btl.start_offset(),
            &stream,
        );
        let (cu_ptr, _cug) = naiwa_quant::utils::slice_ptr_on_stream(
            cu_c.as_cuda_slice::<u32>()?,
            cul.start_offset(),
            &stream,
        );

        unsafe {
            ffi::gather_kv_cache(
                kc_ptr,
                vc_ptr,
                ko_ptr,
                vo_ptr,
                std::ptr::null(),
                std::ptr::null(),
                bt_ptr as *const c_int,
                cu_ptr as *const c_int,
                total_tokens as c_int,
                num_seqs as c_int,
                block_size as c_int,
                max_blocks as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                x as c_int,
                stream.cu_stream(),
                out_code,
                cache_code,
            );
        }
    }

    Ok((k_out, v_out))
}

#[cfg(not(feature = "cuda"))]
pub fn gather_kv_cache(
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _block_tables: &Tensor,
    _cu_seq_lens: &Tensor,
    _total_tokens: usize,
    _out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("gather_kv_cache requires CUDA")
}

/// Single-token decode PagedAttention (`q`: `[num_seqs, num_heads, head_dim]`) via v1/v2 CUDA kernel.
#[cfg(feature = "cuda")]
pub fn paged_attention_decode(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr};
    use candle_core::{CudaStorage, Shape, Storage};
    use core::ffi::{c_int, c_void};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    struct WsSlot {
        slice: CudaSlice<u8>,
        cap: usize,
    }
    static WS: OnceLock<Mutex<HashMap<candle_core::cuda_backend::DeviceId, &'static Mutex<WsSlot>>>> =
        OnceLock::new();

    let q = q.contiguous()?;
    let bt = block_tables.to_dtype(DType::U32)?.contiguous()?;
    let cl = context_lens.to_dtype(DType::U32)?.contiguous()?;
    let (num_seqs, num_heads, head_size) = q.dims3()?;
    let (_, num_kv_heads, _, block_size, _) = key_cache.dims5()?;
    let (_, max_num_blocks_per_seq) = bt.dims2()?;
    let dtype = q.dtype();
    let cache_dtype = dtype_code(key_cache.dtype())?;
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let q_stride = (num_heads * head_size) as c_int;
    let kc_strides = key_cache.layout().stride();
    let kv_block_stride = kc_strides[0] as c_int;
    let kv_head_stride = kc_strides[1] as c_int;

    let partition_size = 512usize;
    let effective_max_context_len = (max_num_blocks_per_seq * block_size).min(max_context_len);
    let max_num_partitions = effective_max_context_len.div_ceil(partition_size).max(1);
    let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
        && partition_size.is_multiple_of(block_size);

    let elem_count = num_seqs * num_heads * head_size;
    let out_shape = Shape::from((num_seqs, num_heads, head_size));

    let (qs, ql) = q.storage_and_layout();
    let (kcs, kcl) = key_cache.storage_and_layout();
    let (vcs, vcl) = value_cache.storage_and_layout();
    let (bts, btl) = bt.storage_and_layout();
    let (cls, cll) = cl.storage_and_layout();
    let (
        Storage::Cuda(q_c),
        Storage::Cuda(kc_c),
        Storage::Cuda(vc_c),
        Storage::Cuda(bt_c),
        Storage::Cuda(cl_c),
    ) = (&*qs, &*kcs, &*vcs, &*bts, &*cls)
    else {
        candle_core::bail!("paged_attention_decode requires CUDA tensors");
    };

    let (bt_ptr, _btg) = naiwa_quant::utils::slice_ptr_on_stream(
        bt_c.as_cuda_slice::<u32>()?,
        btl.start_offset(),
        &stream,
    );
    let (cl_ptr, _clg) = naiwa_quant::utils::slice_ptr_on_stream(
        cl_c.as_cuda_slice::<u32>()?,
        cll.start_offset(),
        &stream,
    );

    macro_rules! launch_typed {
        ($ty:ty, $v1:path, $v2:path) => {{
            let (q_ptr, _qg) = naiwa_quant::utils::slice_ptr_on_stream(
                q_c.as_cuda_slice::<$ty>()?,
                ql.start_offset(),
                &stream,
            );
            let (kc_ptr, _kcg) = naiwa_quant::utils::slice_ptr_on_stream(
                kc_c.as_cuda_slice::<$ty>()?,
                kcl.start_offset(),
                &stream,
            );
            let (vc_ptr, _vcg) = naiwa_quant::utils::slice_ptr_on_stream(
                vc_c.as_cuda_slice::<$ty>()?,
                vcl.start_offset(),
                &stream,
            );
            let out_slice = unsafe { dev.alloc::<$ty>(elem_count)? };
            let (out_ptr, out_guard) = out_slice.device_ptr(&stream);
            if use_v1 {
                unsafe {
                    $v1(
                        out_ptr as *const c_void,
                        q_ptr as *const c_void,
                        kc_ptr as *const c_void,
                        vc_ptr as *const c_void,
                        std::ptr::null(),
                        num_kv_heads as c_int,
                        softmax_scale,
                        softcapping,
                        bt_ptr as *const c_int,
                        cl_ptr as *const c_int,
                        block_size as c_int,
                        effective_max_context_len as c_int,
                        num_seqs as c_int,
                        num_heads as c_int,
                        head_size as c_int,
                        max_num_blocks_per_seq as c_int,
                        q_stride,
                        kv_block_stride,
                        kv_head_stride,
                        stream.cu_stream(),
                        cache_dtype,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    );
                }
            } else {
                let tmp_out_bytes =
                    num_seqs * num_heads * max_num_partitions * head_size * std::mem::size_of::<$ty>();
                let exp_sums_bytes = num_seqs * num_heads * max_num_partitions * std::mem::size_of::<f32>();
                let align16 = |n: usize| n.div_ceil(16) * 16;
                let exp_off = align16(tmp_out_bytes);
                let max_off = align16(exp_off + exp_sums_bytes);
                let total_ws = (max_off + exp_sums_bytes).max(1);

                let map = WS.get_or_init(|| Mutex::new(HashMap::new()));
                let dev_mtx: &'static Mutex<WsSlot> = {
                    let mut g = map.lock().unwrap();
                    match g.get(&dev.id()).copied() {
                        Some(m) => m,
                        None => {
                            let slice = unsafe { dev.alloc::<u8>(total_ws)? };
                            let leaked = Box::leak(Box::new(Mutex::new(WsSlot { slice, cap: total_ws })));
                            g.insert(dev.id(), leaked);
                            leaked
                        }
                    }
                };
                let mut slot = dev_mtx.lock().unwrap();
                if slot.cap < total_ws {
                    slot.slice = unsafe { dev.alloc::<u8>(total_ws)? };
                    slot.cap = total_ws;
                }
                let (ws_ptr, _wsg) = slot.slice.device_ptr(&stream);
                unsafe {
                    $v2(
                        out_ptr as *const c_void,
                        (ws_ptr + exp_off as u64) as *const f32,
                        (ws_ptr + max_off as u64) as *const f32,
                        ws_ptr as *const c_void,
                        q_ptr as *const c_void,
                        kc_ptr as *const c_void,
                        vc_ptr as *const c_void,
                        std::ptr::null(),
                        num_kv_heads as c_int,
                        softmax_scale,
                        softcapping,
                        bt_ptr as *const c_int,
                        cl_ptr as *const c_int,
                        block_size as c_int,
                        effective_max_context_len as c_int,
                        num_seqs as c_int,
                        num_heads as c_int,
                        head_size as c_int,
                        max_num_blocks_per_seq as c_int,
                        q_stride,
                        kv_block_stride,
                        kv_head_stride,
                        stream.cu_stream(),
                        cache_dtype,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    );
                }
            }
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out_slice, dev.clone())),
                out_shape,
            ))
        }};
    }

    let out = match dtype {
        DType::BF16 => launch_typed!(half::bf16, ffi::paged_attention_v1_bf16, ffi::paged_attention_v2_bf16),
        DType::F16 => launch_typed!(half::f16, ffi::paged_attention_v1_f16, ffi::paged_attention_v2_f16),
        DType::F32 => launch_typed!(f32, ffi::paged_attention_v1_f32, ffi::paged_attention_v2_f32),
        other => candle_core::bail!("paged_attention_decode unsupported dtype {other:?}"),
    };
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn paged_attention_decode(
    _q: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _block_tables: &Tensor,
    _context_lens: &Tensor,
    _max_context_len: usize,
    _softmax_scale: f32,
    _softcapping: f32,
) -> Result<Tensor> {
    candle_core::bail!("paged_attention_decode requires CUDA")
}

/// High-level PagedAttention layer handling both decode (`paged_attention_v1/v2`) and
/// prefill/multi-token verify (`reshape_and_cache` + `gather_kv_cache` + `sdpa_forward`).
#[derive(Clone, Debug)]
pub struct PagedAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
    pub softcapping: f32,
}

impl PagedAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, softmax_scale: f32) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            softcapping: 1.0,
        }
    }

    /// Write `[total_tokens, num_kv_heads, head_dim]` K/V into the paged cache without running attention.
    pub fn write_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    }

    /// Write K/V into the paged cache and run attention for either decode or prefill/verify.
    ///
    /// - `q`: `[total_tokens, num_heads, head_dim]`
    /// - `k`, `v`: `[total_tokens, num_kv_heads, head_dim]`
    /// Returns `[total_tokens, num_heads, head_dim]`.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        meta: &PagedAttnMeta,
    ) -> Result<Tensor> {
        reshape_and_cache(k, v, key_cache, value_cache, &meta.slot_mapping)?;

        if meta.is_decode {
            return paged_attention_decode(
                q,
                key_cache,
                value_cache,
                &meta.block_tables,
                &meta.context_lens,
                meta.max_context_len,
                self.softmax_scale,
                self.softcapping,
            );
        }

        // Multi-token verify / small refresh fast path: each query row runs as a virtual
        // causal decode sequence directly against paged KV (no gather_kv_cache, no SDPA)!
        if let (Some(rbt), Some(rcl)) = (&meta.row_block_tables, &meta.row_context_lens) {
            if q.device().is_cuda() {
                return paged_attention_decode(
                    q,
                    key_cache,
                    value_cache,
                    rbt,
                    rcl,
                    meta.max_context_len,
                    self.softmax_scale,
                    self.softcapping,
                );
            }
        }

        // Prefill or multi-token verify: gather full KV per sequence from paged cache (supports prefix/chunked prefill + verify!)
        let total_kv: usize = meta.host_context_lens.iter().sum();
        let (all_k, all_v) = gather_kv_cache(
            key_cache,
            value_cache,
            &meta.block_tables,
            &meta.cu_context_lens,
            total_kv,
            q.dtype(),
        )?;

        let sdpa_params = SdpaParams {
            n_kv_groups: self.num_heads / self.num_kv_heads,
            softmax_scale: self.softmax_scale,
            softcap: (self.softcapping != 1.0 && self.softcapping != 0.0).then_some(self.softcapping),
        };

        let mut outs = Vec::with_capacity(meta.host_query_lens.len());
        let mut q_off = 0usize;
        let mut kv_off = 0usize;
        for (&q_len, &kv_len) in meta
            .host_query_lens
            .iter()
            .zip(meta.host_context_lens.iter())
        {
            if q_len == 0 {
                kv_off += kv_len;
                continue;
            }
            let qi = q
                .narrow(0, q_off, q_len)?
                .transpose(0, 1)?
                .unsqueeze(0)?; // [1, H, q_len, D]
            let ki = all_k
                .narrow(0, kv_off, kv_len)?
                .transpose(0, 1)?
                .unsqueeze(0)?; // [1, H_kv, kv_len, D]
            let vi = all_v
                .narrow(0, kv_off, kv_len)?
                .transpose(0, 1)?
                .unsqueeze(0)?;
            let yi = sdpa_forward(&qi, &ki, &vi, None, true, sdpa_params)?
                .squeeze(0)?
                .transpose(0, 1)?; // [q_len, H, D]
            outs.push(yi);
            q_off += q_len;
            kv_off += kv_len;
        }
        if outs.len() == 1 {
            Ok(outs.pop().unwrap())
        } else {
            Tensor::cat(&outs, 0)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn paged_decode_and_gather_match_sdpa() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (num_heads, num_kv_heads, head_dim, block_size) = (8, 2, 128, 16);
        let num_blocks = 8;
        let ctx_len = 23usize;
        let (kc, vc) = alloc_kv_cache(num_blocks, num_kv_heads, head_dim, block_size, DType::BF16, &dev)?;

        // Write 23 tokens into blocks [2, 5] (slots 32..48, 80..87)
        let k_all = Tensor::randn(0f32, 0.5, (ctx_len, num_kv_heads, head_dim), &dev)?.to_dtype(DType::BF16)?;
        let v_all = Tensor::randn(0f32, 0.5, (ctx_len, num_kv_heads, head_dim), &dev)?.to_dtype(DType::BF16)?;
        let mut slots = Vec::with_capacity(ctx_len);
        for t in 0..ctx_len {
            let blk = if t < block_size { 2i64 } else { 5i64 };
            slots.push(blk * block_size as i64 + (t % block_size) as i64);
        }
        let slot_t = Tensor::from_vec(slots, (ctx_len,), &dev)?;
        reshape_and_cache(&k_all, &v_all, &kc, &vc, &slot_t)?;

        let bt = Tensor::from_vec(vec![2u32, 5u32], (1, 2), &dev)?;
        let cu = Tensor::from_vec(vec![0u32, ctx_len as u32], (2,), &dev)?;
        let (kg, vg) = gather_kv_cache(&kc, &vc, &bt, &cu, ctx_len, DType::BF16)?;
        let k_diff = (kg.to_dtype(DType::F32)? - k_all.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let v_diff = (vg.to_dtype(DType::F32)? - v_all.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(k_diff < 1e-4 && v_diff < 1e-4, "gather diff k={k_diff} v={v_diff}");

        // Compare paged_attention_decode against SDPA for 1 query token (softcapping=1.0 disables softcap)
        let q = Tensor::randn(0f32, 0.5, (1, num_heads, head_dim), &dev)?.to_dtype(DType::BF16)?;
        let cl = Tensor::from_vec(vec![ctx_len as u32], (1,), &dev)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let got = paged_attention_decode(&q, &kc, &vc, &bt, &cl, ctx_len, scale, 1.0)?;

        let q4 = q.transpose(0, 1)?.unsqueeze(0)?;
        let k4 = k_all.transpose(0, 1)?.unsqueeze(0)?;
        let v4 = v_all.transpose(0, 1)?.unsqueeze(0)?;
        let want = sdpa_forward(
            &q4,
            &k4,
            &v4,
            None,
            false,
            SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softmax_scale: scale,
                softcap: None,
            },
        )?
        .squeeze(0)?
        .transpose(0, 1)?;
        let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 2e-2, "paged decode vs SDPA max diff = {diff}");

        // Also test multi-row causal paged_attention_decode (MTP verify path: 3 query rows
        // with causal context_lens [21, 22, 23] sharing physical blocks [2, 5]).
        let q3 = Tensor::randn(0f32, 0.5, (3, num_heads, head_dim), &dev)?.to_dtype(DType::BF16)?;
        let rbt = Tensor::from_vec(vec![2u32, 5, 2, 5, 2, 5], (3, 2), &dev)?;
        let rcl = Tensor::from_vec(vec![21u32, 22, 23], (3,), &dev)?;
        let got3 = paged_attention_decode(&q3, &kc, &vc, &rbt, &rcl, ctx_len, scale, 1.0)?;
        let q3_4 = q3.transpose(0, 1)?.unsqueeze(0)?;
        let want3 = sdpa_forward(
            &q3_4,
            &k4,
            &v4,
            None,
            true,
            SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softmax_scale: scale,
                softcap: None,
            },
        )?
        .squeeze(0)?
        .transpose(0, 1)?;
        let diff3 = (got3.to_dtype(DType::F32)? - want3.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff3 < 2e-2, "multi-row causal paged decode vs SDPA max diff = {diff3}");
        Ok(())
    }
}
