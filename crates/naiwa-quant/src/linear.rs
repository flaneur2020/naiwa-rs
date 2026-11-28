//! Single-device linear layer supporting both GGUF quantized weights (`QTensor`) and dense `Tensor`s.

use std::sync::{Arc, OnceLock};

use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Device, Result, Tensor,
};

use crate::gguf::{dequant, qtensor_matmul};

/// A single-device linear projection `y = x @ W^T (+ b)`, also supporting quantized embedding lookup.
#[derive(Clone)]
pub enum QLinear {
    Quantized {
        weight: Arc<QTensor>,
        bias: Option<Tensor>,
        iq_host_bytes: Arc<OnceLock<Option<Vec<u8>>>>,
    },
    /// Host-resident quantized embedding table (e.g. `IQ4_XS` `token_embd.weight`), avoiding
    /// uploading hundreds of MB to VRAM only to copy it back to host on the first token.
    HostQuantizedEmbedding {
        raw: Arc<Vec<u8>>,
        rows: usize,
        cols: usize,
        dtype: candle_core::quantized::GgmlDType,
        device: Device,
    },
    Dense {
        weight: Tensor,
        bias: Option<Tensor>,
    },
}

impl QLinear {
    pub fn from_qtensor(weight: Arc<QTensor>, bias: Option<Tensor>) -> Self {
        Self::Quantized {
            weight,
            bias,
            iq_host_bytes: Arc::new(OnceLock::new()),
        }
    }

    pub fn from_host_iq_embedding(
        raw: Vec<u8>,
        rows: usize,
        cols: usize,
        dtype: candle_core::quantized::GgmlDType,
        device: Device,
    ) -> Self {
        Self::HostQuantizedEmbedding {
            raw: Arc::new(raw),
            rows,
            cols,
            dtype,
            device,
        }
    }

    pub fn from_dense(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Dense { weight, bias }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Quantized { weight, .. } => weight.device(),
            Self::HostQuantizedEmbedding { device, .. } => device.clone(),
            Self::Dense { weight, .. } => weight.device().clone(),
        }
    }

    pub fn qtensor(&self) -> Option<&Arc<QTensor>> {
        match self {
            Self::Quantized { weight, .. } => Some(weight),
            Self::HostQuantizedEmbedding { .. } | Self::Dense { .. } => None,
        }
    }

    /// Returns the dense weight tensor when this projection is an unbiased `QLinear::Dense`.
    pub fn dense_weight(&self) -> Option<&Tensor> {
        match self {
            Self::Dense { weight, bias: None } => Some(weight),
            _ => None,
        }
    }

    /// Embedding lookup using host-resident token IDs directly (avoids a synchronous GPU->CPU
    /// copy of `input_ids` when the embedding table is host-dequantized, e.g. `IQ4_XS`).
    pub fn embedding_forward_host_ids(
        &self,
        ids: &[u32],
        prefix_shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match self {
            Self::HostQuantizedEmbedding {
                raw,
                rows,
                cols,
                dtype: qdtype,
                device,
            } => Self::dequant_host_iq_rows(raw, *rows, *cols, *qdtype, ids, prefix_shape, dtype, device),
            Self::Quantized {
                weight,
                iq_host_bytes,
                ..
            } if dequant::is_fork_iq(weight.dtype()) => {
                let qdtype = weight.dtype();
                let raw = iq_host_bytes.get_or_init(|| weight.data().ok().map(|b| b.into_owned()));
                let raw = raw.as_deref().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "failed to read host bytes for {qdtype:?} embedding"
                    ))
                })?;
                let (rows, cols) = weight.shape().dims2()?;
                Self::dequant_host_iq_rows(
                    raw,
                    rows,
                    cols,
                    qdtype,
                    ids,
                    prefix_shape,
                    dtype,
                    &weight.device(),
                )
            }
            _ => {
                let ids_t = Tensor::from_slice(ids, prefix_shape, &self.device())?;
                self.embedding_forward(&ids_t, dtype)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_host_iq_rows(
        raw: &[u8],
        rows: usize,
        cols: usize,
        qdtype: candle_core::quantized::GgmlDType,
        ids: &[u32],
        prefix_shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let row_bytes = (cols / qdtype.block_size()) * qdtype.type_size();
        if raw.len() < rows * row_bytes {
            candle_core::bail!(
                "IQ embedding: {rows}x{cols} {qdtype:?} needs {} bytes, got {}",
                rows * row_bytes,
                raw.len()
            );
        }
        let mut out_shape = prefix_shape.to_vec();
        out_shape.push(cols);
        let mut f32_row = vec![0f32; cols];
        match dtype {
            DType::BF16 => {
                let mut out = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    let idx = id as usize;
                    if idx >= rows {
                        candle_core::bail!("embedding id {idx} out of range for {rows} rows");
                    }
                    let src = &raw[idx * row_bytes..(idx + 1) * row_bytes];
                    dequant::dequantize(src, qdtype, &mut f32_row)?;
                    out.extend(f32_row.iter().map(|&v| half::bf16::from_f32(v)));
                }
                Tensor::from_vec(out, out_shape, device)
            }
            DType::F16 => {
                let mut out = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    let idx = id as usize;
                    if idx >= rows {
                        candle_core::bail!("embedding id {idx} out of range for {rows} rows");
                    }
                    let src = &raw[idx * row_bytes..(idx + 1) * row_bytes];
                    dequant::dequantize(src, qdtype, &mut f32_row)?;
                    out.extend(f32_row.iter().map(|&v| half::f16::from_f32(v)));
                }
                Tensor::from_vec(out, out_shape, device)
            }
            _ => {
                let mut out = vec![0f32; ids.len() * cols];
                for (i, &id) in ids.iter().enumerate() {
                    let idx = id as usize;
                    if idx >= rows {
                        candle_core::bail!("embedding id {idx} out of range for {rows} rows");
                    }
                    let src = &raw[idx * row_bytes..(idx + 1) * row_bytes];
                    dequant::dequantize(src, qdtype, &mut out[i * cols..(i + 1) * cols])?;
                }
                Tensor::from_vec(out, out_shape, device)?.to_dtype(dtype)
            }
        }
    }

    /// Embedding lookup for `ids` of any integer shape `[..]` -> `[.., cols]` in `dtype` on `self.device()`.
    pub fn embedding_forward(&self, ids: &Tensor, dtype: DType) -> Result<Tensor> {
        match self {
            Self::Dense { weight, .. } => {
                let mut out_shape = ids.dims().to_vec();
                let (_, cols) = weight.dims2()?;
                out_shape.push(cols);
                let flat = ids.flatten_all()?.to_device(weight.device())?;
                weight
                    .index_select(&flat, 0)?
                    .reshape(out_shape)?
                    .to_dtype(dtype)
            }
            Self::HostQuantizedEmbedding {
                raw,
                rows,
                cols,
                dtype: qdtype,
                device,
            } => {
                let row_bytes = (*cols / qdtype.block_size()) * qdtype.type_size();
                if raw.len() < *rows * row_bytes {
                    candle_core::bail!(
                        "Host IQ embedding: {rows}x{cols} {qdtype:?} needs {} bytes, got {}",
                        *rows * row_bytes,
                        raw.len()
                    );
                }
                let mut out_shape = ids.dims().to_vec();
                out_shape.push(*cols);
                let id_vec = ids
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::U32)?
                    .flatten_all()?
                    .contiguous()?
                    .to_vec1::<u32>()?;
                let mut out = vec![0f32; id_vec.len() * *cols];
                for (i, &id) in id_vec.iter().enumerate() {
                    let idx = id as usize;
                    if idx >= *rows {
                        candle_core::bail!("embedding id {idx} out of range for {rows} rows");
                    }
                    let src = &raw[idx * row_bytes..(idx + 1) * row_bytes];
                    dequant::dequantize(src, *qdtype, &mut out[i * *cols..(i + 1) * *cols])?;
                }
                Tensor::from_vec(out, out_shape, device)?.to_dtype(dtype)
            }
            Self::Quantized {
                weight,
                iq_host_bytes,
                ..
            } => {
                let qdtype = weight.dtype();
                if dequant::is_fork_iq(qdtype) {
                    let raw = iq_host_bytes.get_or_init(|| weight.data().ok().map(|b| b.into_owned()));
                    let raw = raw.as_deref().ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "failed to read host bytes for {qdtype:?} embedding"
                        ))
                    })?;
                    let (rows, cols) = weight.shape().dims2()?;
                    let row_bytes = (cols / qdtype.block_size()) * qdtype.type_size();
                    if raw.len() < rows * row_bytes {
                        candle_core::bail!(
                            "IQ embedding: {rows}x{cols} {qdtype:?} needs {} bytes, got {}",
                            rows * row_bytes,
                            raw.len()
                        );
                    }
                    let mut out_shape = ids.dims().to_vec();
                    out_shape.push(cols);
                    let id_vec = ids
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::U32)?
                        .flatten_all()?
                        .contiguous()?
                        .to_vec1::<u32>()?;
                    let mut out = vec![0f32; id_vec.len() * cols];
                    for (i, &id) in id_vec.iter().enumerate() {
                        let idx = id as usize;
                        if idx >= rows {
                            candle_core::bail!("embedding id {idx} out of range for {rows} rows");
                        }
                        let src = &raw[idx * row_bytes..(idx + 1) * row_bytes];
                        dequant::dequantize(src, qdtype, &mut out[i * cols..(i + 1) * cols])?;
                    }
                    return Tensor::from_vec(out, out_shape, &weight.device())?.to_dtype(dtype);
                }
                let qmm = QMatMul::from_arc(Arc::clone(weight))?;
                let ids_dev = ids.to_device(&weight.device())?;
                qmm.embedding(&ids_dev)?.to_dtype(dtype)
            }
        }
    }

    /// Forward pass: `xs` has shape `[.., in_features]`, output has shape `[.., out_features]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Self::Dense { weight, bias } = self {
            if crate::gemv::should_use_gemv(xs, weight) {
                return crate::gemv::gemv(xs, weight, bias.as_ref());
            }
        }

        let mut out = match self {
            Self::Quantized { weight, .. } => qtensor_matmul(weight, xs)?,
            Self::HostQuantizedEmbedding { .. } => {
                candle_core::bail!("HostQuantizedEmbedding only supports embedding_forward")
            }
            Self::Dense { weight, .. } => {
                let w = if weight.dtype() != xs.dtype() {
                    weight.to_dtype(xs.dtype())?
                } else {
                    weight.clone()
                };
                let wt = w.t()?;
                let dims = xs.dims();
                if dims.len() > 2 {
                    let k = dims[dims.len() - 1];
                    let rows: usize = dims[..dims.len() - 1].iter().product();
                    let out_features = w.dim(0)?;
                    let flat = xs.contiguous()?.reshape((rows, k))?;
                    let y = flat.matmul(&wt)?;
                    let mut out_dims = dims[..dims.len() - 1].to_vec();
                    out_dims.push(out_features);
                    y.reshape(out_dims)?
                } else {
                    xs.contiguous()?.matmul(&wt)?
                }
            }
        };
        if let Some(bias) = match self {
            Self::Quantized { bias, .. } | Self::Dense { bias, .. } => bias.as_ref(),
            Self::HostQuantizedEmbedding { .. } => None,
        } {
            let b = if bias.dtype() != out.dtype() {
                bias.to_dtype(out.dtype())?
            } else {
                bias.clone()
            };
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}
