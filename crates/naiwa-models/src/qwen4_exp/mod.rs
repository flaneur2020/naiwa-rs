//! Qwen3.8-Flash-Next (`qwen4exp`) text model components.

pub mod config;
pub(crate) mod hyper;
pub mod loader;
pub(crate) mod moe;
pub mod mtp;
pub(crate) mod ple;
pub(crate) mod qsa;
pub mod text;

pub use config::{LayerType, PleConfig, PleHashConstants, QsaConfig, Qwen4ExpConfig};
pub use loader::{load_qwen4_exp_from_archive, open_gguf_archive};
pub use mtp::{MtpForwardInput, MtpForwardOutput};
pub use text::{ForwardInput, ForwardKind, ForwardOutput, Qwen4ExpModel};
