//! Fast CUDA GEMV for dense BF16/F16/F32 linear layers (`batch_size` 1..=8).

#[cfg(feature = "cuda")]
mod cuda_impl {
    use core::ffi::c_void;

    use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    extern "C" {
        fn naiwa_launch_gemv_bf16(
            a: *const c_void,
            x: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            m: i32,
            k: i32,
            batch_size: i32,
            has_bias: bool,
            stream: *mut c_void,
        );
        fn naiwa_launch_gemv_f16(
            a: *const c_void,
            x: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            m: i32,
            k: i32,
            batch_size: i32,
            has_bias: bool,
            stream: *mut c_void,
        );
        fn naiwa_launch_gemv_f32(
            a: *const c_void,
            x: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            m: i32,
            k: i32,
            batch_size: i32,
            has_bias: bool,
            stream: *mut c_void,
        );
    }

    /// Returns `true` when `x` (`[.., K]`) and `w` (`[M, K]`) should dispatch to our custom
    /// CUDA GEMV kernel instead of cuBLAS.
    pub fn should_use_gemv(x: &Tensor, w: &Tensor) -> bool {
        if !x.device().is_cuda() || !w.device().is_cuda() {
            return false;
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32) || x.dtype() != w.dtype() {
            return false;
        }
        if !w.is_contiguous() {
            return false;
        }
        let dims = x.dims();
        if dims.len() < 2 {
            return false;
        }
        let k = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows == 0 || rows > 8 || k == 0 {
            return false;
        }
        let Ok((m, wk)) = w.dims2() else {
            return false;
        };
        // Custom warp-shuffle GEMV beats cuBLAS when K is wide (e.g. K=2560, 10240) and M is
        // small-to-medium; for tall-narrow matrices like hc_up [10240, 320] cuBLAS is faster.
        wk == k && m > 0 && k >= 1024
    }

    /// Run `Y = X @ W^T (+ bias)` using `naiwa_gemv_kernel_batched`.
    pub fn gemv(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.dims();
        let k = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let (m, _) = w.dims2()?;

        let x = x.contiguous()?;
        let dev = x.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        let stream_ptr = stream.cu_stream() as *mut c_void;

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(m);
        let out_shape = Shape::from(out_dims);

        let (xs, xl) = x.storage_and_layout();
        let (ws, wl) = w.storage_and_layout();
        let (Storage::Cuda(xc), Storage::Cuda(wc)) = (&*xs, &*ws) else {
            candle_core::bail!("gemv expects CUDA tensors");
        };

        let bias_contig = match bias {
            Some(b) if b.dtype() == x.dtype() => Some(b.contiguous()?),
            Some(b) => Some(b.to_dtype(x.dtype())?.contiguous()?),
            None => None,
        };
        let has_bias = bias_contig.is_some();

        macro_rules! dispatch {
            ($ty:ty, $launch:ident) => {{
                let mut out = unsafe { dev.alloc::<$ty>(rows * m)? };
                {
                    let (ap, _ag) = slice_ptr_on_stream(
                        wc.as_cuda_slice::<$ty>()?,
                        wl.start_offset(),
                        &stream,
                    );
                    let (xp, _xg) = slice_ptr_on_stream(
                        xc.as_cuda_slice::<$ty>()?,
                        xl.start_offset(),
                        &stream,
                    );
                    let (yp, _yg) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                    if let Some(ref b) = bias_contig {
                        let (bs, bl) = b.storage_and_layout();
                        let Storage::Cuda(bc) = &*bs else {
                            candle_core::bail!("gemv bias expects CUDA tensor");
                        };
                        let (bp, _bg) = slice_ptr_on_stream(
                            bc.as_cuda_slice::<$ty>()?,
                            bl.start_offset(),
                            &stream,
                        );
                        unsafe {
                            $launch(
                                ap as *const c_void,
                                xp as *const c_void,
                                bp as *const c_void,
                                yp as *mut c_void,
                                m as i32,
                                k as i32,
                                rows as i32,
                                has_bias,
                                stream_ptr,
                            );
                        }
                    } else {
                        unsafe {
                            $launch(
                                ap as *const c_void,
                                xp as *const c_void,
                                std::ptr::null(),
                                yp as *mut c_void,
                                m as i32,
                                k as i32,
                                rows as i32,
                                false,
                                stream_ptr,
                            );
                        }
                    }
                }
                let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
                Ok(Tensor::from((Storage::Cuda(out_storage), out_shape)))
            }};
        }

        match x.dtype() {
            DType::BF16 => dispatch!(half::bf16, naiwa_launch_gemv_bf16),
            DType::F16 => dispatch!(half::f16, naiwa_launch_gemv_f16),
            DType::F32 => dispatch!(f32, naiwa_launch_gemv_f32),
            other => candle_core::bail!("gemv unsupported dtype {other:?}"),
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{gemv, should_use_gemv};
