//! Route: top-k doorbell read and hit/miss split.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

use crate::moe::CachedLayer;

use super::Hit;

pub(crate) struct RoutePlan {
    pub hits: Vec<Vec<Hit>>,
    pub misses: Vec<Vec<(u32, f32)>>,
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

        // Step 5 doorbell: lazily acquire (first decode), queue the ring on the same stream the top-k
        // kernels ran on, then spin host-side for the payload — no stream sync, no D2H launch.
        // Falls back to `to_vec2` when unavailable.
        let ring = if device_flip {
            None
        } else {
            let mut st = layer.state().lock().unwrap();
            if st.doorbell.is_none() {
                let cap = n.max(8 * layer.top_k()).max(64);
                st.doorbell = crate::doorbell::TopkDoorbell::acquire(layer.slot_device(), cap)
                    .ok()
                    .flatten()
                    .map(Arc::new);
            }
            let db = st.doorbell.clone().filter(|db| db.max_n() >= n);
            match db {
                Some(db) => match db.publish(topk_ids, topk_weights, n) {
                    Ok(()) => {
                        let seq = st.doorbell_seq;
                        st.doorbell_seq = seq.wrapping_add(1);
                        Some((db, seq))
                    }
                    Err(_) => None,
                },
                None => None,
            }
        };
        let (ids_rows, weight_rows): (Vec<Vec<u32>>, Vec<Vec<f32>>);
        if let Some((db, seq)) = ring {
            let (_, ids, ws) = db.wait_and_read(seq, n);
            // W106 sanitation: a garbage id means the doorbell payload was read stale or corrupted;
            // log everything needed and fall back to the blocking D2H rather than indexing out of
            // bounds.
            if ids.iter().any(|&id| id as usize >= layer.num_experts()) {
                let bad: Vec<u32> = ids
                    .iter()
                    .copied()
                    .filter(|id| *id as usize >= layer.num_experts())
                    .take(8)
                    .collect();
                candle_core::bail!(
                "doorbell garbage: layer {} seq {} n {} cap {} rows {} top_k {} ids_shape {:?} bad_ids {:x?}",
                layer.layer(),
                seq,
                n,
                db.max_n(),
                rows,
                layer.top_k(),
                topk_ids.shape(),
                bad
            );
            }
            ids_rows = ids.chunks(layer.top_k()).map(<[u32]>::to_vec).collect();
            weight_rows = ws.chunks(layer.top_k()).map(<[f32]>::to_vec).collect();
        } else {
            ids_rows = topk_ids.reshape((rows, ()))?.to_vec2::<u32>()?;
            weight_rows = topk_weights
                .reshape((rows, ()))?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
        }

        let mut hits: Vec<Vec<Hit>> = vec![Vec::new(); rows];
        let mut misses: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        {
            let mut st = layer.state().lock().unwrap();
            st.tick += 1;
            let tick = st.tick;
            for (t, (ids, ws)) in ids_rows.iter().zip(&weight_rows).enumerate() {
                for (&id, &w) in ids.iter().zip(ws) {
                    st.hit_counts[id as usize] += 1;
                    let resident = st
                        .slots
                        .iter()
                        .position(|s| s.expert == Some(id as usize) && s.buf.is_some())
                        .map(|slot| (slot, st.slots[slot].buf.clone().unwrap()));
                    match resident {
                        Some((slot, buf)) => {
                            st.slots[slot].last_use = tick;
                            hits[t].push((buf, slot, w));
                        }
                        None => misses[t].push((id, w)),
                    }
                }
            }
        }

        // Probe `NAIWA_GPU_ONLY_EXPERTS`: once the cache is full, drop every expert that is not
        // resident and never stage or compute one. This is lossy; it bounds the perfect-cache path.
        #[cfg(feature = "cuda")]
        if std::env::var_os("NAIWA_GPU_ONLY_EXPERTS").is_some() {
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

        Ok(RoutePlan { hits, misses })
    }
}
