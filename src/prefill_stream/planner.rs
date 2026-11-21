//! Pure VRAM budget planning for `--prefill-stream`: picks the prompt chunk size and
//! buffer count together from measured free VRAM and optional cache-slab lending capacity.

use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::info;

use super::PrefillStream;

/// Default for `--prefill-stream-min-tokens`: chunks shorter than this stay on the CPU (Strata
/// streams every expert from 1024 tokens).
pub const DEFAULT_MIN_TOKENS: usize = 1024;

/// Default for `--prefill-depth`: the double buffer this file was built around (M18's 16-19% overlap).
pub const DEFAULT_DEPTH: usize = 2;

/// The most layer buffers `--prefill-depth` may ask for. Four is the point where the copy queue is fed by more than
/// one layer's worth of work; beyond that the plan's arithmetic is the limit, not the pipeline.
pub(crate) const MAX_DEPTH: usize = 4;

/// Free VRAM kept on top of what the plan prices: the KV blocks and the copy-stream staging a prompt
/// step grows into, plus the driver's own bookkeeping.
pub(super) const MAX_HEADROOM: usize = 512 << 20;

/// The slack a **lent** chunk keeps on top of its measured peak (256 MiB = half of `MAX_HEADROOM`).
pub(crate) const CHUNK_HEAD: usize = 256 << 20;

/// Grid granularity (in tokens) for chunk-size bisection.
pub(super) const PLAN_GRID: usize = 256;

/// Historical row-batch size used in `activation_bytes` shape estimates.
pub(crate) const SUB_ROWS: usize = 256;

/// The operator's ceiling on the prompt step (`min(--max-num-batched-tokens,
/// --max-prefill-chunk-tokens)`), published by the commands before the model loads. The plan below
/// only ever chooses at or below it.
static PROMPT_STEP_CEILING: AtomicUsize = AtomicUsize::new(0);

/// `0` clears the ceiling (tests, non-CUDA).
pub fn set_prompt_step_ceiling(tokens: usize) {
    PROMPT_STEP_CEILING.store(tokens, Ordering::Release);
}

pub(super) fn prompt_step_ceiling() -> usize {
    match PROMPT_STEP_CEILING.load(Ordering::Acquire) {
        0 => DEFAULT_MIN_TOKENS * 8,
        n => n,
    }
}

/// The prefill path's memory decision: how many layer buffers, and the prompt step they were
/// priced with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Plan {
    pub stride: usize,
    pub chunk: usize,
}

/// Device bytes a prompt chunk really takes on this card, measured rather than derived (M33 §12/§13):
/// `32 MiB + 354 KiB * rows` by default, overridable via `NAIWA_CHUNK_FIXED_MIB` / `NAIWA_CHUNK_PER_ROW_KIB`.
pub(crate) fn chunk_scratch(rows: usize) -> usize {
    let cfg = crate::config::runtime();
    cfg.chunk_fixed_bytes + cfg.chunk_per_row_bytes * rows
}

/// Shape-based activation estimate retained for peak logging and diagnostic comparisons.
pub(crate) fn activation_bytes(chunk: usize) -> usize {
    let moe = (160 << 20) + (680 << 20) * chunk.min(SUB_ROWS) / 1024;
    let rest = chunk * (24 << 10);
    moe + rest
}

/// The KV bytes the plan must leave alone: the paged-attention pool grows to the first prompt's length *after*
/// warmup's measurement (default 384 MiB; override via `NAIWA_KV_RESERVE_MIB`).
pub(super) fn kv_reserve() -> usize {
    crate::config::runtime().kv_reserve_bytes
}

#[cfg(test)]
pub(crate) fn plan(
    free: usize,
    layer_bytes: usize,
    ceiling: usize,
    borrowing: bool,
) -> Option<(usize, usize)> {
    plan_with_reserve(free, layer_bytes, ceiling, borrowing, 0)
}

/// Plan with an explicit `kv_reserve` subtracted from free VRAM before sizing.
pub(crate) fn plan_with_reserve(
    free: usize,
    layer_bytes: usize,
    ceiling: usize,
    borrowing: bool,
    kv_reserve: usize,
) -> Option<(usize, usize)> {
    plan_with_lend(free, layer_bytes, ceiling, borrowing, kv_reserve, 0)
}

/// Plan with explicit `lend_bytes` (the slab's lendable pool size) and default depth 2.
pub(crate) fn plan_with_lend(
    free: usize,
    layer_bytes: usize,
    ceiling: usize,
    borrowing: bool,
    kv_reserve: usize,
    lend_bytes: usize,
) -> Option<(usize, usize)> {
    plan_with_lend_depth(free, layer_bytes, ceiling, borrowing, kv_reserve, lend_bytes, 2)
}

/// The whole prefill memory decision in one pure function: largest prompt step on the 256-token
/// grid that fits, paired with the deepest buffer stride (`1..=max_depth`) that fits at that step.
pub(crate) fn plan_with_lend_depth(
    free: usize,
    layer_bytes: usize,
    ceiling: usize,
    borrowing: bool,
    kv_reserve: usize,
    lend_bytes: usize,
    max_depth: usize,
) -> Option<(usize, usize)> {
    const GRID: usize = PLAN_GRID;
    let free = free.saturating_sub(kv_reserve);
    let slack_override = crate::config::runtime().prefill_slack_bytes;
    let fits = |chunk: usize, stride: usize| {
        let head = slack_override.unwrap_or_else(|| MAX_HEADROOM.max(free / 8));
        if borrowing {
            free >= chunk_scratch(chunk) + head.min(CHUNK_HEAD)
        } else {
            free >= stride * layer_bytes + chunk_scratch(chunk) + head
        }
    };
    // Largest chunk with one buffer: `fits` is monotone in the chunk, so bisect.
    let (mut lo, mut hi, mut best) = (1usize, (ceiling.min(8192) / GRID).max(1), 0usize);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid * GRID, 1) {
            best = mid * GRID;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if best == 0 {
        return None;
    }
    let reserve = crate::config::runtime().decode_reserve_bytes;
    let depth_max = max_depth.clamp(1, MAX_DEPTH);
    let mut stride = 1;
    if borrowing {
        let cap = crate::config::runtime().depth_cap;
        stride = (lend_bytes / layer_bytes.max(1)).clamp(1, depth_max.min(cap));
    } else {
        for d in (2..=depth_max).rev() {
            let owns_fits = fits(best, d)
                && free.saturating_sub(d * layer_bytes + chunk_scratch(best) + MAX_HEADROOM)
                    >= reserve;
            if owns_fits {
                stride = d;
                break;
            }
        }
    }
    Some((stride, best))
}

/// One line with the whole decision, so the numbers are checkable in the log.
pub(super) fn log_plan(stream: &PrefillStream, plan: Plan, free: usize, when: &str) {
    let mib = |b: usize| (b as f64) / (1u64 << 20) as f64;
    let layer = stream.layer_bytes();
    let (buffers, scratch, head) = if stream.borrows() {
        (0, chunk_scratch(plan.chunk), MAX_HEADROOM / 2)
    } else {
        (plan.stride * layer, chunk_scratch(plan.chunk), MAX_HEADROOM)
    };
    info!(
        "prefill ({when}): chunk {}{} + {:.0} MiB chunk scratch + {:.0} MiB headroom = {:.0} MiB of \
         {:.0} MiB free",
        plan.chunk,
        if stream.borrows() {
            format!(", {} layer buffer(s) loaned from the cache", plan.stride)
        } else {
            format!(", {} x {:.0} MiB layer buffer(s)", plan.stride, mib(layer))
        },
        mib(scratch),
        mib(head),
        mib(buffers + scratch + head),
        mib(free),
    );
    if let Some(slab) = stream.slab.as_ref() {
        let layout = slab.layout();
        info!(
            "prefill ({when}): cache pool {:.0} MiB of {:.0} MiB carved in {} ranges{} of the {:.0} MiB slab",
            mib(layout.end()),
            mib(slab.bytes()),
            layout.count(),
            if layout.end() == slab.bytes() {
                " (exactly full)"
            } else {
                ""
            },
            mib(slab.bytes()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunk_line_is_the_two_points_that_were_measured() {
        assert_eq!(chunk_scratch(288) >> 20, 131);
        assert_eq!(chunk_scratch(1216) >> 20, 452);
        assert!(chunk_scratch(1216) > activation_bytes(1216));
        assert!(chunk_scratch(1216) < 1346 << 20);
    }

    #[test]
    fn plan_spends_memory_on_the_chunk_before_the_second_buffer() {
        let layer = 650 << 20;
        let gb = |g: usize| g << 30;
        let act = |c: usize| activation_bytes(c);

        assert_eq!(
            act(8192) - act(SUB_ROWS),
            (8192 - SUB_ROWS) * (24 << 10),
            "only the attention term may scale"
        );
        assert_eq!(plan(gb(12), layer, 8192, false).unwrap(), (2, 8192));
        assert_eq!(plan(2049 << 20, layer, 8192, true).unwrap(), (1, 4864));
        assert_eq!(
            plan_with_lend(2049 << 20, layer, 8192, true, 0, 1536 << 20).unwrap(),
            (2, 4864)
        );
        assert_eq!(
            plan_with_lend(gb(12), layer, 8192, true, 0, (1300 << 20) - 1)
                .unwrap()
                .0,
            1
        );
        assert_eq!(
            plan_with_lend(2049 << 20, layer, 8192, false, 0, 1536 << 20)
                .unwrap()
                .0,
            1,
            "without borrowing the pool is irrelevant: the buffers are real VRAM"
        );
        let lent = plan(2049 << 20, layer, 8192, true).unwrap().1;
        let owned = plan(2049 << 20, layer, 8192, false).unwrap().1;
        assert!(
            owned < lent,
            "{owned} owned rows should be fewer than {lent} lent"
        );
        assert!(chunk_scratch(lent) + CHUNK_HEAD <= 2049 << 20);

        assert_eq!(plan(2049 << 20, layer, 8192, false).unwrap(), (1, 2304));
        assert_eq!(plan(2400 << 20, layer, 8192, false).unwrap(), (1, 3328));
        assert_eq!(plan(3034 << 20, layer, 8192, false).unwrap(), (1, 5120));
        assert_eq!(plan(gb(12), layer, 512, false).unwrap().1, 512);
        assert_eq!(plan(gb(8), layer, 8192, false).unwrap(), (2, 8192));
        assert_eq!(
            plan_with_lend(2049 << 20, layer, 8192, true, 0, 1536 << 20).unwrap(),
            (2, 4864)
        );
        assert_eq!(plan(layer, layer, 1024, false), None);
    }

    #[test]
    fn the_kv_reserve_is_not_spendable() {
        let layer = 650 << 20;
        let free = 1300 << 20;
        let refused = plan_with_reserve(free, layer, 8192, false, 384 << 20);
        assert!(
            refused.is_none(),
            "with the KV reserve the stream cannot fit: {refused:?}"
        );
        assert!(plan_with_reserve(free, layer, 8192, false, 0).is_some());
        let shipping = plan_with_reserve(2498 << 20, layer, 8192, false, 384 << 20);
        assert!(
            shipping.is_some_and(|(_, chunk)| chunk >= 2048),
            "{shipping:?}"
        );
    }
}
