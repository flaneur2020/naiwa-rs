//! Pure expert-cache selection, ranking, and victim-choice functions.
//!
//! Marked `#[inline]` so cross-module calls from `decode`, `dma_miss`, and `cache_updates`
//! inline identically to single-file definitions.

use std::collections::HashMap;

use super::Slot;

/// The most-routed expert not in (or reserved for) a slot, ties to the lowest id; `None` when
/// every slot is taken.
#[inline]
pub(super) fn fill_candidate(slots: &[Slot], hit_counts: &[u64]) -> Option<usize> {
    if slots.iter().all(|s| s.expert.is_some()) {
        return None;
    }
    let mut taken = vec![false; hit_counts.len()];
    for e in slots.iter().filter_map(|s| s.expert) {
        taken[e] = true;
    }
    (0..hit_counts.len())
        .filter(|&e| !taken[e])
        .max_by_key(|&e| (hit_counts[e], std::cmp::Reverse(e)))
}

/// Indices of the `k` largest `logits`, largest first (ties: lowest index).
#[inline]
pub(super) fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

/// Every expert in `rankings` once, ordered by how many rankings list it, then by its best rank,
/// then by id.
#[inline]
pub(super) fn union_by_votes(rankings: impl IntoIterator<Item = Vec<usize>>) -> Vec<usize> {
    let mut seen: HashMap<usize, (usize, usize)> = HashMap::new(); // expert -> (votes, best rank)
    for ranking in rankings {
        for (rank, e) in ranking.into_iter().enumerate() {
            let entry = seen.entry(e).or_insert((0, rank));
            entry.0 += 1;
            entry.1 = entry.1.min(rank);
        }
    }
    let mut union: Vec<(usize, (usize, usize))> = seen.into_iter().collect();
    union.sort_by(|(a, (va, ra)), (b, (vb, rb))| vb.cmp(va).then(ra.cmp(rb)).then(a.cmp(b)));
    union.into_iter().map(|(e, _)| e).collect()
}

/// The misses to stream over PCIe (`--pcie-frac`): the distinct experts a step missed, ranked by
/// the total router weight routed to them, truncated to `cap`.
#[inline]
pub(crate) fn streamed_misses(misses: &[Vec<(u32, f32)>], frac: f32, cap: usize) -> Vec<u32> {
    if frac <= 0.0 || cap == 0 {
        return Vec::new();
    }
    let mut weight: HashMap<u32, f32> = HashMap::new();
    for (id, w) in misses.iter().flatten() {
        *weight.entry(*id).or_insert(0.0) += *w;
    }
    let take = ((frac * weight.len() as f32).ceil() as usize).min(cap);
    let mut ranked: Vec<(u32, f32)> = weight.into_iter().collect();
    ranked.sort_by(|(a, wa), (b, wb)| wb.total_cmp(wa).then(a.cmp(b)));
    ranked.truncate(take);
    ranked.into_iter().map(|(id, _)| id).collect()
}

/// The least-recently-used published resident that is not in `protect`, as `(expert, slot)`.
#[inline]
pub(super) fn lru_victim(slots: &[Slot], protect: &[usize]) -> Option<(usize, usize)> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(slot, s)| {
            let expert = s.expert?;
            (s.buf.is_some() && !protect.contains(&expert)).then_some((expert, slot))
        })
        .min_by_key(|(expert, slot)| (slots[*slot].last_use, *expert))
}
