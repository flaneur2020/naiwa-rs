//! `naiwa-models`: GGUF text model architectures built directly on `candle-core`, `naiwa-quant`,
//! and `naiwa-attn` (zero `mistral.rs` dependency).

pub mod cache;
pub mod probe;
pub mod qwen4_exp;

pub use cache::{HybridCache, PleSeqState, PleVerifyStash, RecurrentVerifyStash};
pub use probe::{set_boundary_probe, BoundaryHook};
pub use qwen4_exp::{
    load_qwen4_exp_from_archive, open_gguf_archive, ForwardInput, ForwardKind, ForwardOutput,
    LayerType, MtpForwardInput, MtpForwardOutput, PleConfig, PleHashConstants, QsaConfig,
    Qwen4ExpConfig, Qwen4ExpModel,
};
