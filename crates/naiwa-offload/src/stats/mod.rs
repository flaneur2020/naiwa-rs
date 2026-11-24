//! Process-wide instrumentation: cache counters and JSONL attribution sinks.

mod cache;
mod sink;
pub(crate) mod schedule;

#[cfg(feature = "cuda")]
pub(crate) mod gpu_partition;
#[cfg(feature = "cuda")]
pub(crate) mod head_partition;
#[cfg(feature = "cuda")]
pub mod model_partition;

pub use cache::{dump_cache_stats, CacheStats};
pub(crate) use cache::{
    record_cache_hits, record_cache_installed, record_cache_lookahead_dropped,
    record_cache_lookahead_load, record_cache_misses, record_cache_pcie, record_cache_result_stall,
    record_cache_staged, record_cache_swap, record_cache_sync_wait, record_promotions,
};
