//! `naiwa-quant`: GGUF archive reading, IQ/Q2_0/MMQ/MMVQ/MoE kernels, single-device `QLinear`,
//! and `RoutedExperts` offloading traits built directly on `candle-core`.

pub mod gguf;
pub mod linear;
pub mod moe;
pub mod utils;

pub use gguf::{dequant_qtensor_cpu, is_custom_iq_dtype, qtensor_matmul, GgufArchive};
#[cfg(feature = "cuda")]
pub use gguf::{
    indexed_moe_fused_decode, moe_workspace_bytes, qtensor_indexed_moe_forward,
    ACT_GELU_PYTORCH_TANH, ACT_SILU,
};
pub use gguf::{fused_moe_decode, install_task_dispatcher};
pub use linear::QLinear;
pub use moe::{
    Activation, ExpertLoadRequest, ExpertSite, MoEExperts, MoEExpertsConfig, RoutedExperts,
    RoutedExpertsProvider,
};
pub use utils::{fused_glu, GluActivationType};
