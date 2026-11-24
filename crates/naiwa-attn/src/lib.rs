//! `naiwa-attn`: Attention, GatedDeltaNet (GDN), QSA/PLE/Hyper-Connections, RoPE, and RMSNorm
//! operators built directly on `candle-core` / `candle-nn` and `naiwa-quant`.

pub mod gdn;
pub mod norm;
pub mod paged;
#[cfg(feature = "cuda")]
pub mod qwen4_exp;
pub mod rope;
pub mod sdpa;

pub use gdn::{GatedDeltaNet, GdnConfig, GdnGateActivation, GdnSeqState, GdnVHeadLayout};
pub use norm::{grouped_rms_norm, rms_norm_with_weight, GemmaRmsNorm, RmsNorm};
pub use paged::{alloc_kv_cache, PagedAttention, PagedAttnMeta};
pub use rope::RotaryEmbedding;
pub use sdpa::{causal_mask, repeat_kv, sdpa_forward, SdpaParams};
