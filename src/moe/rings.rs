//! Ring batching for prompt chunks: split a layer's routed experts into rounds that fit the slots the layer
//! can actually use, including slots lent by other layers of the same shape.
//!
//! The prompt path needs every expert a chunk routes to, which above ~26 rows is all 256 of them, but a layer
//! only owns ~25 cache slots. So the layer's expert set is processed in rounds: each round owns a set of
//! experts (its own slots plus the slots borrowed from same-shape layers), the experts are DMA'd into those
//! slots, and only the `(row, expert)` pairs of that round are computed.
//!
//! Two properties make this safe to swap in:
//!
//! - **it partitions, it does not repeat**: every routed `(row, expert)` pair lands in exactly one round, so the
//!   bytes moved and the FLOPs are the same as one big pass - unlike splitting a prompt into two chunks, which
//!   measured 2.2x slower because each chunk re-reads the whole expert set;
//! - **rounds are balanced by pairs, not by expert count**: experts differ in popularity by an order of
//!   magnitude, so slicing the expert list evenly would leave the hottest round several times larger than the
//!   coldest. The greedy below keeps the largest round within a small factor of the average.

/// Which slots a round runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoundSlots {
    /// The layer's own cache slots.
    Own,
    /// Slots lent by another layer of the same shape; its own slot indices are reused.
    Lent { lender: usize },
}

/// One round: the experts it owns and, per row, the routing entries it is responsible for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Round {
    pub(crate) slots: RoundSlots,
    pub(crate) experts: Vec<u32>,
    /// `(row, [(expert, weight)])`, only the pairs of `experts`. Rows with nothing here are omitted.
    pub(crate) rows: Vec<(usize, Vec<(u32, f32)>)>,
}

impl Round {
    /// Routing entries this round computes; the cost model the balancing uses.
    pub(crate) fn pairs(&self) -> usize {
        self.rows.iter().map(|(_, e)| e.len()).sum()
    }
}

/// A chunk's plan for one layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RingPlan {
    pub(crate) rounds: Vec<Round>,
}

impl RingPlan {
    pub(crate) fn pairs(&self) -> usize {
        self.rounds.iter().map(Round::pairs).sum()
    }

    /// Largest round by pairs, i.e. the one that decides the layer's critical path.
    pub(crate) fn largest_round(&self) -> usize {
        self.rounds.iter().map(Round::pairs).max().unwrap_or(0)
    }
}

/// Plan the rounds for one layer.
///
/// `entries` is the layer's routing: `(row, expert, weight)` for every routed pair. `own`, `lent` are the slot
/// counts available before a new round has to reuse slots - the layer's own and what same-shape layers can lend
/// (both already reduced by whatever the cache must keep; the planner only consumes what it is given).
///
/// Both zero means "no ring": the whole expert set goes in one round, which is what a layer with a whole-layer
/// buffer does today.
pub(crate) fn plan(
    entries: &[(usize, u32, f32)],
    own: usize,
    lent: usize,
    lender: Option<usize>,
) -> RingPlan {
    let capacity = own.saturating_add(lent);
    if entries.is_empty() || capacity == 0 {
        return RingPlan {
            rounds: vec![Round {
                slots: RoundSlots::Own,
                experts: entries.iter().map(|&(_, e, _)| e).collect::<std::collections::BTreeSet<_>>().into_iter().collect(),
                rows: group_rows(entries),
            }],
        };
    }

    // Popularity of each expert in this chunk, and the pairs it owns.
    let mut count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &(_, e, _) in entries {
        *count.entry(e).or_default() += 1;
    }
    // Greedy longest-processing-time bin packing over expert popularity: hot experts go to the emptiest round,
    // which keeps the rounds within a factor of each other instead of slicing the id order.
    let mut experts: Vec<(u32, usize)> = count.into_iter().collect();
    experts.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let total: usize = experts.iter().map(|(_, n)| *n).sum();
    let min_rounds = experts.len().div_ceil(capacity).max(1);
    let mut bins: Vec<(Vec<u32>, usize)> = vec![(Vec::new(), 0); min_rounds];
    for (expert, weight) in experts {
        // The emptiest bin that still has room for one more expert.
        let pick = bins
            .iter()
            .enumerate()
            .filter(|(_, (e, _))| e.len() < capacity)
            .min_by_key(|(_, (_, pairs))| *pairs)
            .map(|(i, _)| i);
        match pick {
            Some(i) => {
                bins[i].0.push(expert);
                bins[i].1 += weight;
            }
            None => bins.push((vec![expert], weight)),
        }
    }
    debug_assert!(bins.iter().map(|(_, p)| p).sum::<usize>() == total);

    // The first `own` slots' worth run on the layer's own slots, the rest on the lender's.
    let mut rounds = Vec::with_capacity(bins.len());
    let mut used_own = 0usize;
    for (experts, _) in bins {
        let slots = if used_own < own {
            used_own += experts.len();
            RoundSlots::Own
        } else {
            match lender {
                Some(l) => RoundSlots::Lent { lender: l },
                // No lender: keep the round on our own slots anyway; the caller reuses them.
                None => RoundSlots::Own,
            }
        };
        let set: std::collections::HashSet<u32> = experts.iter().copied().collect();
        let mine: Vec<(usize, u32, f32)> = entries
            .iter()
            .copied()
            .filter(|&(_, e, _)| set.contains(&e))
            .collect();
        rounds.push(Round {
            slots,
            experts,
            rows: group_rows(&mine),
        });
    }
    RingPlan { rounds }
}

/// `(row, expert, weight)` entries to `(row, [(expert, weight)])`, rows ascending, entries in input order.
fn group_rows(entries: &[(usize, u32, f32)]) -> Vec<(usize, Vec<(u32, f32)>)> {
    let mut rows: std::collections::BTreeMap<usize, Vec<(u32, f32)>> = std::collections::BTreeMap::new();
    for &(row, expert, weight) in entries {
        rows.entry(row).or_default().push((expert, weight));
    }
    rows.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8 experts, 4 rows x topk 2; `own = 2`, `lent = 2` -> capacity 4 -> exactly 2 rounds.
    fn sample() -> Vec<(usize, u32, f32)> {
        vec![
            (0, 0, 0.5),
            (0, 1, 0.5),
            (1, 1, 0.3),
            (1, 2, 0.7),
            (2, 2, 0.4),
            (2, 3, 0.6),
            (3, 3, 0.1),
            (3, 0, 0.9),
            (4, 7, 1.0),
            (4, 6, 0.0),
        ]
    }

    #[test]
    fn every_pair_lands_in_exactly_one_round() {
        let entries = sample();
        let plan = plan(&entries, 2, 2, Some(3));
        let pairs = plan.pairs();
        assert_eq!(pairs, entries.len(), "rounds partition the routing, no more, no less");

        let mut seen: Vec<(usize, u32)> = plan
            .rounds
            .iter()
            .flat_map(|r| r.rows.iter().flat_map(|(row, es)| es.iter().map(move |&(e, _)| (*row, e))))
            .collect();
        seen.sort_unstable();
        let mut want: Vec<(usize, u32)> = entries.iter().map(|&(r, e, _)| (r, e)).collect();
        want.sort_unstable();
        assert_eq!(seen, want);
    }

    #[test]
    fn the_weights_survive_the_split() {
        let entries = sample();
        let plan = plan(&entries, 2, 2, Some(3));
        let mut got: Vec<(usize, u32, f32)> = plan
            .rounds
            .iter()
            .flat_map(|r| r.rows.iter().flat_map(|(row, es)| es.iter().map(move |&(e, w)| (*row, e, w))))
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want = entries.clone();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, want);
    }

    #[test]
    fn rounds_respect_the_capacity_and_use_the_lender() {
        let entries = sample();
        let plan = plan(&entries, 2, 2, Some(3));
        for round in &plan.rounds {
            assert!(round.experts.len() <= 4, "a round cannot exceed own + lent slots");
        }
        // 6 distinct experts, capacity 4 -> 2 rounds; the second one runs on the lender's slots.
        assert_eq!(plan.rounds.len(), 2);
        assert_eq!(plan.rounds[0].slots, RoundSlots::Own);
        assert_eq!(plan.rounds[1].slots, RoundSlots::Lent { lender: 3 });
    }

    #[test]
    fn the_largest_round_stays_near_the_average() {
        // 40 rows x topk 2 over 12 experts, one of them three times as hot as the rest: capacity 2, so 6 rounds.
        // Greedy longest-processing-time keeps the largest within 2x of the average, which is what decides the
        // layer's critical path.
        let mut entries = Vec::new();
        for row in 0..40usize {
            let hot_hits = if row % 2 == 0 { 3 } else { 0 };
            let experts: Vec<u32> = if hot_hits > 0 {
                vec![0, 1 + (row % 11) as u32]
            } else {
                vec![1 + (row % 11) as u32, 1 + ((row + 5) % 11) as u32]
            };
            let weight = 1.0 / experts.len() as f32;
            for e in experts {
                entries.push((row, e, weight));
            }
        }
        let plan = plan(&entries, 1, 1, Some(7));
        assert_eq!(plan.rounds.len(), 6, "12 experts, capacity 2");
        let avg = plan.pairs() / plan.rounds.len();
        assert!(
            plan.largest_round() <= 2 * avg,
            "largest round {} against average {avg}",
            plan.largest_round()
        );
        for round in &plan.rounds {
            assert!(!round.experts.is_empty());
            assert!(round.experts.len() <= 2);
        }
    }

    #[test]
    fn no_capacity_means_one_round_with_the_whole_set() {
        let entries = sample();
        let plan = plan(&entries, 0, 0, None);
        assert_eq!(plan.rounds.len(), 1);
        assert_eq!(plan.rounds[0].experts.len(), 6);
        assert_eq!(plan.pairs(), entries.len());
    }

    #[test]
    fn an_empty_chunk_plans_nothing_to_do() {
        let plan = plan(&[], 2, 2, Some(1));
        assert_eq!(plan.rounds.len(), 1);
        assert!(plan.rounds[0].experts.is_empty());
        assert_eq!(plan.pairs(), 0);
    }
}
