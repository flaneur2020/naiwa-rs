//! `naiwa-runtime`: GGUF tokenizer & chat template, paged KV block manager, sampler,
//! Built-in MTP speculative decoding driver, and continuous-batching async inference engine.

pub mod block_manager;
pub mod engine;
pub mod mtp;
pub mod sampler;
pub mod tokenizer;

pub use block_manager::{compute_num_blocks, BlockManager, DEFAULT_BLOCK_SIZE};
pub use engine::{
    EngineConfig, EngineHandle, EngineMetrics, FinishReason, GenerationRequest, StreamEvent,
    DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
};
pub use mtp::{MtpConfig, MtpDraftSampling, MtpStats};
pub use sampler::{Sampler, SamplingParams};
pub use tokenizer::{ChatMessage, GgufTokenizer, StreamDecoder};
