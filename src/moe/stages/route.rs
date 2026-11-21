//! Route: top-k doorbell read and hit/miss split.

use candle_core::{DType, Result, Tensor};

use crate::moe::CachedLayer;

use super::Hit;

pub(crate) struct RoutePlan {
    pub resident: Vec<Vec<Hit>>,
    pub staged: Vec<Vec<Hit>>,
    pub misses: Vec<Vec<(u32, f32)>>,
    /// Diagnostic-only payload; no extra device read. Emitted after staging assignment.
    pub work_trace: Option<serde_json::Value>,
}

pub(crate) struct Route;

impl Route {
    pub(crate) fn submit(
        layer: &CachedLayer,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        rows: usize,
    ) -> Result<RoutePlan> {
        let n = rows * layer.top_k();
        #[cfg(feature = "cuda")]
        let device_flip = layer.top_k() > 0
            && topk_ids.elem_count() % layer.top_k() == 0
            && topk_ids.elem_count() / layer.top_k() == rows;
        #[cfg(not(feature = "cuda"))]
        let device_flip = false;

        // Step 5 doorbell: queue the ring on the same stream the top-k kernels ran on, then spin
        // host-side for the payload — no stream sync, no D2H launch. The doorbell is sized at
        // warmup for the largest window, so a publish that does not fit is a bug and fails here
        // rather than silently falling back to a blocking read. Without a doorbell (allocation
        // failed) `to_vec2` is the fallback: correct, just slower.
        let db = if device_flip {
            None
        } else {
            layer.state().lock().unwrap().doorbell.clone()
        };
        let read = match db.as_ref() {
            Some(db) => Some(db.publish_and_read(topk_ids, topk_weights, n)?),
            None => None,
        };
        let (ids_rows, weight_rows): (Vec<Vec<u32>>, Vec<Vec<f32>>) = if let Some((ids, ws)) = read
        {
            // W106 sanitation: a garbage id means the doorbell payload was read stale or
            // corrupted; log everything needed and fall back to the blocking D2H rather than
            // indexing out of bounds.
            if ids.iter().any(|&id| id as usize >= layer.num_experts()) {
                let bad: Vec<u32> = ids
                    .iter()
                    .copied()
                    .filter(|id| *id as usize >= layer.num_experts())
                    .take(8)
                    .collect();
                candle_core::bail!(
                    "doorbell garbage: layer {} n {} rows {} top_k {} ids_shape {:?} bad_ids {:x?}",
                    layer.layer(),
                    n,
                    rows,
                    layer.top_k(),
                    topk_ids.shape(),
                    bad
                );
            }
            (
                ids.chunks(layer.top_k()).map(<[u32]>::to_vec).collect(),
                ws.chunks(layer.top_k()).map(<[f32]>::to_vec).collect(),
            )
        } else {
            (
                topk_ids.reshape((rows, ()))?.to_vec2::<u32>()?,
                topk_weights
                    .reshape((rows, ()))?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?,
            )
        };

        let mut resident: Vec<Vec<Hit>> = vec![Vec::new(); rows];
        let staged: Vec<Vec<Hit>> = vec![Vec::new(); rows];
        let mut misses: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        // M40: compare this pass's real demand against the lookahead's prediction for this layer.
        if crate::stats::predict_enabled() {
            crate::stats::record_actual(layer.layer(), &ids_rows);
        }
        let trace_enabled = crate::config::diag().work_trace;
        let mut work_trace = None;
        {
            let mut st = layer.state().lock().unwrap();
            st.tick += 1;
            let tick = st.tick;
            if trace_enabled {
                let residency: Vec<_> = st.slots.iter().enumerate()
                    .filter(|(_, s)| s.buf.is_some())
                    .filter_map(|(slot, s)| s.expert.map(|id| (slot, id))).collect();
                work_trace = Some(serde_json::json!({
                    "schema": 1, "layer": layer.layer(), "tick": tick, "rows": rows,
                    "ids": ids_rows,
                    "weight_bits": weight_rows.iter().map(|r| r.iter().map(|w| w.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "residency": residency,
                }));
            }
            if crate::config::diag().cpu_layer_trace || crate::config::diag().route_ids {
                use std::sync::atomic::{AtomicU64, Ordering};
                static SEQ: AtomicU64 = AtomicU64::new(0);
                let seq = SEQ.fetch_add(1, Ordering::Relaxed);
                let resident: Vec<_> = st.slots.iter()
                    .filter(|s| s.buf.is_some()).filter_map(|s| s.expert).collect();
                eprintln!("ROUTE_IDS seq={seq} layer={} rows={} ids={:?} resident={:?}",
                    layer.layer(), rows, ids_rows, resident);
            }
            for (t, (ids, ws)) in ids_rows.iter().zip(&weight_rows).enumerate() {
                for (&id, &w) in ids.iter().zip(ws) {
                    st.hit_counts[id as usize] += 1;
                    let found = st
                        .slots
                        .iter()
                        .position(|s| s.expert == Some(id as usize) && s.buf.is_some())
                        .map(|slot| (slot, st.slots[slot].buf.clone().unwrap()));
                    match found {
                        Some((slot, buf)) => {
                            st.slots[slot].last_use = tick;
                            crate::stats::schedule::detail(layer.layer(), tick, "resident_use", 0, Some(id as usize), Some(slot));
                            resident[t].push((buf, slot, w));
                        }
                        None => misses[t].push((id, w)),
                    }
                }
            }
        }

        // Probe `NAIWA_GPU_ONLY_EXPERTS`: once the cache is full, drop every expert that is not
        // resident and never stage or compute one. This is lossy; it bounds the perfect-cache path.
        #[cfg(feature = "cuda")]
        if crate::config::runtime().gpu_only_experts {
            let full = {
                let st = layer.state().lock().unwrap();
                st.slots[..st.usable].iter().all(|s| s.expert.is_some())
            };
            if full {
                for row in misses.iter_mut() {
                    row.clear();
                }
            }
        }

        Ok(RoutePlan { resident, staged, misses, work_trace })
    }
}
