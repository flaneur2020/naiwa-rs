//! Layer lookups and the prefill stream's memory loan against the cache.

use std::sync::{Arc, Weak};

use crate::prefill_stream::ExpertMemoryLender;

use super::{CachedLayer, LayerRegistry};

impl LayerRegistry {
    pub(super) fn insert(&self, layer: usize, cached: &Arc<CachedLayer>) {
        let mut layers = self.layers.write();
        if layers.len() <= layer {
            layers.resize_with(layer + 1, Weak::new);
        }
        layers[layer] = Arc::downgrade(cached);
    }

    pub(super) fn get(&self, layer: usize) -> Option<Arc<CachedLayer>> {
        self.layers.read().get(layer).and_then(Weak::upgrade)
    }

    pub(super) fn all(&self) -> Vec<Arc<CachedLayer>> {
        self.layers
            .read()
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
}

// Lends the cache's tail slots to the prefill stream while a prompt runs (see
// `prefill_stream`): each layer gives up an equal share of the bytes asked for.
impl ExpertMemoryLender for LayerRegistry {
    fn lend(&self, bytes: usize) -> candle_core::Result<usize> {
        let layers = self.all();
        let share = bytes.div_ceil(layers.len().max(1));
        Ok(layers
            .iter()
            .map(|l| {
                let per_slot = l.slot_bytes();
                if per_slot == 0 {
                    return 0;
                }
                l.lend(share.div_ceil(per_slot)) * per_slot
            })
            .sum())
    }

    fn give_back(&self) -> candle_core::Result<()> {
        for layer in self.all() {
            layer.give_back()?;
        }
        Ok(())
    }
}
