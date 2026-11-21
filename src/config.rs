//! Centralized environment-variable configuration for `naiwa-rs`.
//!
//! Every `NAIWA_*` variable is read once into [`EnvConfig`] so hot paths never call `getenv`
//! per layer or per step, and the entire env surface is inspectable in one place.
//!
//! Parsing semantics intentionally preserve historical behavior verbatim:
//! - presence-only flags (`var_os().is_some()`) stay presence-only (e.g., `NAIWA_CPU_PHYSICAL_LAYOUT`)
//! - strict `=1` flags (`== "1"`) stay strict `=1` (`NAIWA_CPU_BEFORE_DMA`, `NAIWA_WORK_TRACE`)
//! - default-on flags (`0`/`off`/`false`/`no` disable) stay default-on (`NAIWA_CACHE_SLAB`, `NAIWA_PREFILL_BORROW`).

use std::ffi::OsString;
use std::sync::OnceLock;
use std::time::Duration;

use crate::moe::MAX_DECODE_ROWS;

/// Default row threshold above which MoE forward takes the whole-chunk prefill path.
const DEFAULT_PREFILL_THRESHOLD: usize = 32;

/// Behavior-affecting runtime overrides sourced from `NAIWA_*` environment variables.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    /// `NAIWA_CPU_BEFORE_DMA=1`: submit complementary CPU misses before demand DMA.
    pub cpu_before_dma: bool,
    /// `NAIWA_CPU_SPIN_US` (raw value when set): enables `spin_pool::dispatch` and sets spin budget.
    pub cpu_spin_us_raw: Option<OsString>,
    /// `NAIWA_CPU_MOE_THREADS` (usize, default 8): worker count for `spin_pool`.
    pub cpu_moe_threads: usize,
    /// `NAIWA_CPU_PHYSICAL_LAYOUT` (presence): pin spin-pool threads to physical cores + caller on core 0.
    pub cpu_physical_layout: bool,

    /// `NAIWA_PROMOTE_MAX` (usize, default 1): max staged experts promoted per layer per step.
    pub promote_max: usize,
    /// `NAIWA_PROMOTE_MIN_HITS` (u64, default 4): minimum cumulative routing hits before promotion.
    pub promote_min_hits: u64,
    /// `NAIWA_PREFILL_THRESHOLD` (usize, default 32, clamped to `1..=MAX_DECODE_ROWS`).
    pub prefill_threshold: usize,

    /// `NAIWA_CACHE_SLAB` (default true; `0`/`off`/`false`/`no` disables).
    pub cache_slab: bool,
    /// `NAIWA_SLAB_MIB` (usize, default 1536): initial lending slab window in MiB.
    pub slab_mib: usize,
    /// `NAIWA_SLAB_MIN_MIB` (usize, default 512): minimum lending slab window in MiB before fallback.
    pub slab_min_mib: usize,
    /// `NAIWA_PROFILE_FILL` (f64, default 1.0): fraction of each layer's slots seeded from `--expert-profile`.
    pub profile_fill: f64,

    /// `NAIWA_PREFILL_BORROW` (default true; `0`/`off`/`false`/`no` disables).
    pub prefill_borrow: bool,
    /// `NAIWA_PREFILL_ALLOW_CPU` (presence): allow fallback to CPU prefill if stream does not fit.
    pub prefill_allow_cpu: bool,
    /// `NAIWA_PREFILL_STRICT` (presence): fail rather than downgrade buffers on OOM.
    pub prefill_strict: bool,
    /// `NAIWA_NO_PREFILL_FILL` (presence): disable background empty-slot fills during prefill.
    pub no_prefill_fill: bool,
    /// `NAIWA_REFILL` (`== "0"` disables): whether displaced loan slots are refilled after a chunk.
    pub refill_lent: bool,

    /// `NAIWA_CHUNK_FIXED_MIB` (usize MiB, default 32 MiB) in bytes.
    pub chunk_fixed_bytes: usize,
    /// `NAIWA_CHUNK_PER_ROW_KIB` (usize KiB, default 354 KiB) in bytes.
    pub chunk_per_row_bytes: usize,
    /// `NAIWA_PREFILL_SLACK_MIB` (optional usize MiB) in bytes.
    pub prefill_slack_bytes: Option<usize>,
    /// `NAIWA_DECODE_RESERVE_MIB` (usize MiB, default 700 MiB) in bytes.
    pub decode_reserve_bytes: usize,
    /// `NAIWA_DEPTH_CAP` (usize, default 2): cap on borrowed buffer depth.
    pub depth_cap: usize,
    /// `NAIWA_KV_RESERVE_MIB` (usize MiB, default 384 MiB) in bytes.
    pub kv_reserve_bytes: usize,

    /// `NAIWA_GPU_ONLY_EXPERTS` (presence): lossy perfect-cache upper-bound probe.
    pub gpu_only_experts: bool,
}

impl RuntimeConfig {
    /// Parse `NAIWA_CPU_SPIN_US` for `spin_pool::dispatch` with identical error semantics.
    pub(crate) fn spin_budget(&self) -> Option<Duration> {
        match self.cpu_spin_us_raw.as_ref().and_then(|v| v.to_str()) {
            Some("forever") => None,
            Some(value) => Some(Duration::from_micros(
                value
                    .parse()
                    .expect("NAIWA_CPU_SPIN_US must be integer or forever"),
            )),
            None if self.cpu_spin_us_raw.is_some() => {
                panic!("NAIWA_CPU_SPIN_US must be integer or forever");
            }
            None => Some(Duration::from_micros(1000)),
        }
    }
}

/// Diagnostic, tracing, and profiling switches sourced from `NAIWA_*` environment variables.
#[derive(Debug, Clone)]
pub(crate) struct DiagConfig {
    /// `NAIWA_STREAM_DBG` (presence).
    pub stream_dbg: bool,
    /// `NAIWA_STATS` (presence): print cache totals on graceful `serve` shutdown.
    pub stats: bool,

    /// `NAIWA_XSUM` (presence).
    pub xsum: bool,
    /// `NAIWA_CPU_TIMELINE` (presence).
    pub cpu_timeline: bool,
    /// `NAIWA_WORK_TRACE=1` (strict `=1`).
    pub work_trace: bool,
    /// `NAIWA_WORK_TRACE_FILE` (optional output path).
    pub work_trace_file: Option<OsString>,
    /// `NAIWA_CPU_LAYER_TRACE` (presence).
    pub cpu_layer_trace: bool,
    /// `NAIWA_ROUTE_IDS` (presence).
    pub route_ids: bool,
    /// `NAIWA_STAGE_TRACE` (presence).
    pub stage_trace: bool,
    /// `NAIWA_ADMISSION_TRACE` (presence).
    pub admission_trace: bool,
    /// `NAIWA_PROMOTE_TRACE` (presence).
    pub promote_trace: bool,
    /// `NAIWA_LAYER_DBG` (presence).
    pub layer_dbg: bool,

    /// `NAIWA_PREFILL_PEAK` (presence).
    pub prefill_peak: bool,
    /// `NAIWA_COPY_TIME` (presence).
    pub copy_time: bool,
    /// `NAIWA_REQ_TIME` (presence).
    pub req_time: bool,

    /// `NAIWA_SCHEDULE_FILE` (optional output path).
    pub schedule_file: Option<OsString>,
    /// `NAIWA_GPU_PARTITION_FILE` (optional output path).
    pub gpu_partition_file: Option<OsString>,
    /// `NAIWA_HEAD_PARTITION_FILE` (optional output path).
    pub head_partition_file: Option<OsString>,
    /// `NAIWA_MODEL_PARTITION_FILE` (optional output path).
    pub model_partition_file: Option<OsString>,
    /// `NAIWA_MODEL_PARTITION_STRIDE` (usize, default 1, clamped `>= 1`).
    pub model_partition_stride: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EnvConfig {
    pub runtime: RuntimeConfig,
    pub diag: DiagConfig,
}

impl EnvConfig {
    fn from_lookup<F>(get: F) -> Self
    where
        F: Fn(&str) -> Option<OsString>,
    {
        let present = |k: &str| get(k).is_some();
        let str_val = |k: &str| get(k).and_then(|v| v.into_string().ok());
        let eq_1 = |k: &str| str_val(k).as_deref() == Some("1");
        let default_true = |k: &str| {
            !matches!(
                str_val(k).as_deref(),
                Some("0") | Some("off") | Some("false") | Some("no")
            )
        };
        let opt_usize = |k: &str| str_val(k).and_then(|v| v.parse::<usize>().ok());
        let usize_or = |k: &str, default: usize| opt_usize(k).unwrap_or(default);
        let u64_or = |k: &str, default: u64| {
            str_val(k)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(default)
        };
        let f64_or = |k: &str, default: f64| {
            str_val(k)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(default)
        };

        let runtime = RuntimeConfig {
            cpu_before_dma: eq_1("NAIWA_CPU_BEFORE_DMA"),
            cpu_spin_us_raw: get("NAIWA_CPU_SPIN_US"),
            cpu_moe_threads: usize_or("NAIWA_CPU_MOE_THREADS", 8),
            cpu_physical_layout: present("NAIWA_CPU_PHYSICAL_LAYOUT"),

            promote_max: usize_or("NAIWA_PROMOTE_MAX", 1),
            promote_min_hits: u64_or("NAIWA_PROMOTE_MIN_HITS", 4),
            prefill_threshold: usize_or("NAIWA_PREFILL_THRESHOLD", DEFAULT_PREFILL_THRESHOLD)
                .clamp(1, MAX_DECODE_ROWS),

            cache_slab: default_true("NAIWA_CACHE_SLAB"),
            slab_mib: usize_or("NAIWA_SLAB_MIB", 1536),
            slab_min_mib: usize_or("NAIWA_SLAB_MIN_MIB", 512),
            profile_fill: f64_or("NAIWA_PROFILE_FILL", 1.0),

            prefill_borrow: default_true("NAIWA_PREFILL_BORROW"),
            prefill_allow_cpu: present("NAIWA_PREFILL_ALLOW_CPU"),
            prefill_strict: present("NAIWA_PREFILL_STRICT"),
            no_prefill_fill: present("NAIWA_NO_PREFILL_FILL"),
            refill_lent: str_val("NAIWA_REFILL").as_deref() != Some("0"),

            chunk_fixed_bytes: usize_or("NAIWA_CHUNK_FIXED_MIB", 32) << 20,
            chunk_per_row_bytes: usize_or("NAIWA_CHUNK_PER_ROW_KIB", 354) << 10,
            prefill_slack_bytes: opt_usize("NAIWA_PREFILL_SLACK_MIB").map(|mib| mib << 20),
            decode_reserve_bytes: usize_or("NAIWA_DECODE_RESERVE_MIB", 700) << 20,
            depth_cap: usize_or("NAIWA_DEPTH_CAP", 2),
            kv_reserve_bytes: usize_or("NAIWA_KV_RESERVE_MIB", 384) << 20,

            gpu_only_experts: present("NAIWA_GPU_ONLY_EXPERTS"),
        };

        let diag = DiagConfig {
            stream_dbg: present("NAIWA_STREAM_DBG"),
            stats: present("NAIWA_STATS"),

            xsum: present("NAIWA_XSUM"),
            cpu_timeline: present("NAIWA_CPU_TIMELINE"),
            work_trace: eq_1("NAIWA_WORK_TRACE"),
            work_trace_file: get("NAIWA_WORK_TRACE_FILE"),
            cpu_layer_trace: present("NAIWA_CPU_LAYER_TRACE"),
            route_ids: present("NAIWA_ROUTE_IDS"),
            stage_trace: present("NAIWA_STAGE_TRACE"),
            admission_trace: present("NAIWA_ADMISSION_TRACE"),
            promote_trace: present("NAIWA_PROMOTE_TRACE"),
            layer_dbg: present("NAIWA_LAYER_DBG"),

            prefill_peak: present("NAIWA_PREFILL_PEAK"),
            copy_time: present("NAIWA_COPY_TIME"),
            req_time: present("NAIWA_REQ_TIME"),

            schedule_file: get("NAIWA_SCHEDULE_FILE"),
            gpu_partition_file: get("NAIWA_GPU_PARTITION_FILE"),
            head_partition_file: get("NAIWA_HEAD_PARTITION_FILE"),
            model_partition_file: get("NAIWA_MODEL_PARTITION_FILE"),
            model_partition_stride: usize_or("NAIWA_MODEL_PARTITION_STRIDE", 1).max(1),
        };

        Self { runtime, diag }
    }
}

static ENV: OnceLock<EnvConfig> = OnceLock::new();

#[inline]
pub(crate) fn env() -> &'static EnvConfig {
    ENV.get_or_init(|| EnvConfig::from_lookup(|k| std::env::var_os(k)))
}

#[inline]
pub(crate) fn runtime() -> &'static RuntimeConfig {
    &env().runtime
}

#[inline]
pub(crate) fn diag() -> &'static DiagConfig {
    &env().diag
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::time::Duration;

    use super::EnvConfig;

    fn cfg(pairs: &[(&str, &str)]) -> EnvConfig {
        let map: HashMap<&str, OsString> = pairs
            .iter()
            .map(|(k, v)| (*k, OsString::from(v)))
            .collect();
        EnvConfig::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn defaults_match_historical_values() {
        let c = cfg(&[]);
        assert!(!c.runtime.cpu_before_dma);
        assert!(c.runtime.cpu_spin_us_raw.is_none());
        assert_eq!(c.runtime.spin_budget(), Some(Duration::from_micros(1000)));
        assert_eq!(c.runtime.cpu_moe_threads, 8);
        assert!(!c.runtime.cpu_physical_layout);
        assert_eq!(c.runtime.promote_max, 1);
        assert_eq!(c.runtime.promote_min_hits, 4);
        assert_eq!(c.runtime.prefill_threshold, 32);
        assert!(c.runtime.cache_slab);
        assert_eq!(c.runtime.slab_mib, 1536);
        assert_eq!(c.runtime.slab_min_mib, 512);
        assert_eq!(c.runtime.profile_fill, 1.0);
        assert!(c.runtime.prefill_borrow);
        assert!(c.runtime.refill_lent);
        assert_eq!(c.runtime.chunk_fixed_bytes, 32 << 20);
        assert_eq!(c.runtime.chunk_per_row_bytes, 354 << 10);
        assert_eq!(c.runtime.prefill_slack_bytes, None);
        assert_eq!(c.runtime.decode_reserve_bytes, 700 << 20);
        assert_eq!(c.runtime.depth_cap, 2);
        assert_eq!(c.runtime.kv_reserve_bytes, 384 << 20);
        assert!(!c.diag.work_trace);
        assert_eq!(c.diag.model_partition_stride, 1);
    }

    #[test]
    fn presence_strict_and_default_true_semantics_are_preserved() {
        // Presence-only: even "0" enables NAIWA_CPU_PHYSICAL_LAYOUT.
        let c = cfg(&[
            ("NAIWA_CPU_PHYSICAL_LAYOUT", "0"),
            ("NAIWA_CPU_BEFORE_DMA", "0"),
            ("NAIWA_WORK_TRACE", "true"),
            ("NAIWA_CACHE_SLAB", "off"),
            ("NAIWA_PREFILL_BORROW", "0"),
            ("NAIWA_REFILL", "0"),
        ]);
        assert!(c.runtime.cpu_physical_layout);
        assert!(!c.runtime.cpu_before_dma);
        assert!(!c.diag.work_trace);
        assert!(!c.runtime.cache_slab);
        assert!(!c.runtime.prefill_borrow);
        assert!(!c.runtime.refill_lent);

        let c2 = cfg(&[
            ("NAIWA_CPU_BEFORE_DMA", "1"),
            ("NAIWA_WORK_TRACE", "1"),
            ("NAIWA_CPU_SPIN_US", "forever"),
            ("NAIWA_PREFILL_THRESHOLD", "9999"),
            ("NAIWA_MODEL_PARTITION_STRIDE", "0"),
        ]);
        assert!(c2.runtime.cpu_before_dma);
        assert!(c2.diag.work_trace);
        assert_eq!(c2.runtime.spin_budget(), None);
        assert_eq!(c2.runtime.prefill_threshold, crate::moe::MAX_DECODE_ROWS);
        assert_eq!(c2.diag.model_partition_stride, 1);
    }

    #[test]
    fn invalid_numeric_values_fall_back_to_defaults() {
        let c = cfg(&[
            ("NAIWA_PROMOTE_MAX", "abc"),
            ("NAIWA_PROMOTE_MIN_HITS", "-1"),
            ("NAIWA_CPU_MOE_THREADS", "x"),
            ("NAIWA_SLAB_MIB", "bad"),
            ("NAIWA_PROFILE_FILL", "nan?"),
        ]);
        assert_eq!(c.runtime.promote_max, 1);
        assert_eq!(c.runtime.promote_min_hits, 4);
        assert_eq!(c.runtime.cpu_moe_threads, 8);
        assert_eq!(c.runtime.slab_mib, 1536);
        assert_eq!(c.runtime.profile_fill, 1.0);
    }
}
