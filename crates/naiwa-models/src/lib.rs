//! `naiwa-models`: GGUF text model architectures built directly on `candle-core`, `naiwa-quant`,
//! and `naiwa-attn` (zero `mistral.rs` dependency).

pub mod cache;
pub mod probe;
pub mod qwen4_exp;

pub use cache::{
    HybridCache, HybridPrefixSnapshot, PleSeqState, PleVerifyStash, RecurrentVerifyStash,
};
pub use probe::{set_boundary_probe, BoundaryHook};
pub use qwen4_exp::{
    estimate_model_vram, load_qwen4_exp_from_archive, open_gguf_archive, ForwardInput, ForwardKind,
    ForwardOutput, LayerType, ModelVramPlan, MtpForwardInput, MtpForwardOutput, PleConfig,
    PleHashConstants, QsaConfig, Qwen4ExpConfig, Qwen4ExpModel,
};
