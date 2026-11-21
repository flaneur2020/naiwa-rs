//! Expert sub-range dispatch (`--prefill-split g`): runs one uploaded layer as `g` masked launches
//! over zero-copy views of its buffer, falling back to the whole layer on any view mismatch.

use std::sync::{Arc, OnceLock};

use candle_core::quantized::QTensor;
use candle_core::{DType, Shape, Tensor};
use mistralrs_core::MoEExperts;

use crate::copy_stream::SlotMemory;

use super::buffers::LayerBuf;
use super::PrefillStream;

impl PrefillStream {
    /// The MoE for one uploaded layer, either as one launch over the whole expert set or as `split`
    /// masked launches over expert sub-ranges (M34 S2).
    pub(super) fn dispatch(
        &self,
        experts: &MoEExperts,
        buf: &LayerBuf,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let whole = |experts: &MoEExperts| experts.forward(xs, topk_weights.clone(), topk_ids);
        let g = self.split;
        {
            static ONCE: OnceLock<()> = OnceLock::new();
            if ONCE.set(()).is_ok() {
                crate::dbg_note(
                    "prefill_split",
                    &format!("dispatch: g={g} experts={}", self.num_experts),
                );
            }
        }
        if g <= 1 || self.num_experts % g != 0 {
            return whole(experts);
        }
        let per = self.num_experts / g;
        let Some(memory) = buf.memory.as_ref() else {
            return whole(experts);
        };
        match self.dispatch_split(experts, memory, xs, topk_weights, topk_ids, g, per) {
            Ok(out) => Ok(out),
            Err(e) => {
                static ONCE: OnceLock<()> = OnceLock::new();
                if ONCE.set(()).is_ok() {
                    tracing::warn!(
                        "prefill split {g}: {e}; the chunk runs its experts as whole layers from here"
                    );
                }
                whole(experts)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_split(
        &self,
        experts: &MoEExperts,
        memory: &[SlotMemory; 3],
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        g: usize,
        per: usize,
    ) -> candle_core::Result<Tensor> {
        let mask = crate::config::diag().split_check;
        let whole = if mask {
            Some(experts.forward(xs, topk_weights.clone(), topk_ids)?)
        } else {
            None
        };
        let mut acc: Option<Tensor> = None;
        let mut built = 0usize;
        for k in 0..g {
            let a = k * per;
            let Some(sub) = self.sub_experts(memory, a, per)? else {
                return experts.forward(xs, topk_weights.clone(), topk_ids);
            };
            built += 1;
            let (w, ids) = self.mask_range(topk_weights, topk_ids, a, per)?;
            let part = sub.forward(xs, w, &ids)?;
            acc = Some(match acc {
                None => part,
                Some(prev) => prev.add(&part)?,
            });
        }
        let out = acc.expect("at least one sub-range");
        if let Some(whole) = whole {
            let (a, b) = (
                whole.flatten_all()?.to_dtype(DType::F32)?,
                out.flatten_all()?.to_dtype(DType::F32)?,
            );
            let (av, bv) = (
                a.to_vec1::<f32>().map_err(candle_core::Error::wrap)?,
                b.to_vec1::<f32>().map_err(candle_core::Error::wrap)?,
            );
            let mut worst = 0f32;
            let mut scale = 0f32;
            for (x, y) in av.iter().zip(&bv) {
                worst = worst.max((x - y).abs());
                scale = scale.max(x.abs());
            }
            crate::dbg_note(
                "prefill_split",
                &format!(
                    "split {g} in {built} sub-ranges: worst |diff| {worst:.3e} (scale {scale:.3e}, relative {:.2e})",
                    (worst / scale.max(1e-30)) as f64
                ),
            );
        }
        Ok(out)
    }

    /// `MoEExperts` over experts `[a, a + n)` of one uploaded layer, as views of the buffer's own bytes.
    fn sub_experts(
        &self,
        memory: &[SlotMemory; 3],
        a: usize,
        n: usize,
    ) -> candle_core::Result<Option<MoEExperts>> {
        let dev = self.device.as_cuda_device()?;
        let mut out: Vec<Arc<QTensor>> = Vec::with_capacity(3);
        for mem in memory {
            let dims = mem.tensor().shape().dims().to_vec();
            let rows = dims[0];
            if rows != self.num_experts || a + n > rows {
                crate::dbg_note(
                    "prefill_split",
                    &format!(
                        "no view: rows {rows} against {} experts, range {a}+{n}",
                        self.num_experts
                    ),
                );
                return Ok(None);
            }
            let cols: usize = dims[1..].iter().product();
            let dt = mem.tensor().dtype();
            let per_row = (cols + dt.block_size() - 1) / dt.block_size() * dt.type_size();
            let mut shape = dims.clone();
            shape[0] = n;
            match mem.sub_view(a * per_row, n * cols, Shape::from_dims(&shape), dev) {
                Ok(view) => out.push(Arc::new(view)),
                Err(e) => {
                    crate::dbg_note(
                        "prefill_split",
                        &format!("no view at {a}+{n} rows ({per_row} B/row, {cols} cols): {e}"),
                    );
                    return Ok(None);
                }
            }
        }
        let mut it = out.into_iter();
        Ok(Some(MoEExperts::from_qtensors(
            it.next().expect("gate"),
            it.next().expect("up"),
            it.next().expect("down"),
            n,
            self.top_k,
            &self.comm,
            self.act,
        )?))
    }

    /// `topk_weights` and `topk_ids` restricted to experts `[a, a + n)`: weights outside the range
    /// become 0 and ids are rebased. Uses F32 masking because candle lacks integer CUDA unary ops.
    fn mask_range(
        &self,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        a: usize,
        n: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let dev = topk_ids.device();
        let shape = topk_ids.shape().clone();
        let ids = topk_ids.to_dtype(DType::F32)?;
        let lo = Tensor::full(a as f32, &shape, dev)?;
        let hi = Tensor::full((a + n) as f32, &shape, dev)?;
        let inside = ids
            .broadcast_ge(&lo)?
            .to_dtype(DType::F32)?
            .mul(&ids.broadcast_lt(&hi)?.to_dtype(DType::F32)?)?;
        let w = topk_weights
            .to_dtype(DType::F32)?
            .mul(&inside)?
            .to_dtype(topk_weights.dtype())?;
        let local = ids.broadcast_sub(&lo)?.mul(&inside)?;
        Ok((w, local.to_dtype(topk_ids.dtype())?))
    }
}
