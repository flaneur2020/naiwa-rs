//! Optional boundary-probe hook for `naiwa-offload`'s `model_partition` diagnostics.

use std::sync::OnceLock;

use candle_core::Tensor;

/// Signature of a layer/section boundary callback `(tensor, layer_idx, tag)`.
pub type BoundaryHook = fn(&Tensor, usize, &'static str);

static BOUNDARY_HOOK: OnceLock<BoundaryHook> = OnceLock::new();

/// Install a process-wide boundary probe hook (idempotent; first registration wins).
pub fn set_boundary_probe(hook: BoundaryHook) {
    let _ = BOUNDARY_HOOK.set(hook);
}

#[inline]
pub(crate) fn boundary(t: &Tensor, layer: usize, tag: &'static str) {
    if let Some(hook) = BOUNDARY_HOOK.get() {
        hook(t, layer, tag);
    }
}
