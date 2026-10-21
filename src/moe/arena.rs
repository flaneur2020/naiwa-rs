//! The expert-VRAM arena: one slot pool, expert cache from the top down, prefill from the bottom up.
//!
//! One user, one prompt at a time, so the two consumers never need the same bytes *at the same moment*: the
//! cache wants them while decoding, the prefill stream wants them while a prompt is being read. A single pool
//! with two watermarks serves both, and the eviction order falls out of the geometry:
//!
//! ```text
//!  slot 0                          prefill_high        cache_low                        total
//!  |<------ the stream's buffers ----->|<---- free ---->|<------- the expert cache ------->|
//! ```
//!
//! - The cache allocates **downward** from the top, so its newest (hottest) resident is nearest `total` and its
//!   coldest is nearest `cache_low`.
//! - The prefill stream allocates **upward** from the bottom, in whole-layer runs (`borrow`).
//! - When the stream needs more than the free gap, the cache gives up slots from `cache_low` upward - i.e. its
//!   **coldest first**, which is exactly the order an LRU would have produced, without a second policy.
//! - `release` returns the stream's run; the cache refills from `cache_low` downward as it would after any
//!   eviction, so nothing has to be saved or restored.
//!
//! The arena owns no memory itself: it hands out slot indices and the list of slots the caller must evict, which
//! keeps the policy testable without a GPU.

/// Slot indices handed out by [`SlotArena`].
pub(crate) type SlotId = usize;

/// A borrowed run of slots, plus the cache slots the caller has to evict to make it fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Borrow {
    /// The slots the stream owns until [`SlotArena::release`].
    pub(crate) slots: std::ops::Range<SlotId>,
    /// Cache slots to invalidate, coldest first.
    pub(crate) evict: Vec<SlotId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArenaError {
    /// Asking for more slots than the pool has.
    TooLarge { want: usize, total: usize },
}

/// One pool of uniformly sized slots shared by the expert cache and the prefill stream.
#[derive(Debug, Clone)]
pub(crate) struct SlotArena {
    total: usize,
    /// Slots `[0, prefill_high)` are the stream's.
    prefill_high: usize,
    /// The lowest slot the expert cache currently holds: it occupies `[cache_low, total)` and fills from `total`
    /// downward, so `cache_low == total` means an empty cache.
    cache_low: usize,
}

impl SlotArena {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            total,
            prefill_high: 0,
            cache_low: total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Slots the stream currently owns.
    pub(crate) fn prefill_range(&self) -> std::ops::Range<SlotId> {
        0..self.prefill_high
    }

    /// Slots the expert cache currently holds.
    pub(crate) fn cache_range(&self) -> std::ops::Range<SlotId> {
        self.cache_low..self.total
    }

    /// Free slots between the two watermarks.
    pub(crate) fn gap(&self) -> usize {
        self.cache_low - self.prefill_high
    }

    /// Fill `want` slots for the cache, **from the top down**. Returns the slots it may now use, or `None` when
    /// the free gap is smaller than that (the cache simply holds less).
    pub(crate) fn cache_fill(&mut self, want: usize) -> Option<std::ops::Range<SlotId>> {
        let want = want.min(self.gap());
        let low = self.cache_low - want;
        self.cache_low = low;
        (want > 0).then_some(low..low + want)
    }

    /// Take `want` slots from the bottom for the stream. Whatever the cache holds inside the new run is listed
    /// in `evict`, **coldest (lowest index) first**, because the cache filled from the top.
    ///
    /// Borrowing only ever grows the run: a caller that needs less calls [`Self::release`] and borrows again.
    pub(crate) fn borrow(&mut self, want: usize) -> Result<Borrow, ArenaError> {
        if want > self.total {
            return Err(ArenaError::TooLarge {
                want,
                total: self.total,
            });
        }
        let high = self.prefill_high.max(want);
        let mut evict = Vec::new();
        if high > self.cache_low {
            // The stream eats the cache's cold end, and the cache's low bound moves up with it.
            evict.extend(self.cache_low..high);
            self.cache_low = high;
        }
        self.prefill_high = high;
        Ok(Borrow {
            slots: 0..high,
            evict,
        })
    }

    /// Give the stream's run back. Those slots become refillable by the cache, which grows down into them.
    pub(crate) fn release(&mut self) {
        self.prefill_high = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_fills_from_the_top_and_the_stream_from_the_bottom() {
        let mut arena = SlotArena::new(100);
        assert_eq!(arena.cache_range(), 100..100, "empty");
        assert_eq!(arena.cache_fill(100), Some(0..100), "the cache may take everything");
        assert_eq!(arena.cache_range(), 0..100);
        assert_eq!(arena.gap(), 0);

        let taken = arena.borrow(30).unwrap();
        assert_eq!(taken.slots, 0..30, "the stream always owns the bottom");
        assert_eq!(taken.evict, (0..30).collect::<Vec<_>>(), "the cache's coldest 30 slots");
        assert_eq!(arena.cache_range(), 30..100);
        assert_eq!(arena.gap(), 0);
    }

    #[test]
    fn borrowing_more_evicts_more_and_only_grows_the_run() {
        let mut arena = SlotArena::new(100);
        arena.cache_fill(100);
        assert_eq!(arena.borrow(30).unwrap().evict.len(), 30);
        let more = arena.borrow(60).unwrap();
        assert_eq!(more.slots, 0..60);
        assert_eq!(more.evict, (30..60).collect::<Vec<_>>(), "only the newly eaten slots");
        assert_eq!(arena.cache_range(), 60..100);
    }

    #[test]
    fn a_full_borrow_empties_the_cache_and_refill_happens_after_release() {
        let mut arena = SlotArena::new(100);
        arena.cache_fill(100);
        let all = arena.borrow(100).unwrap();
        assert_eq!(all.evict.len(), 100);
        assert_eq!(arena.cache_range(), 100..100);
        assert_eq!(arena.cache_fill(10), None, "nothing is free while the stream holds it");

        arena.release();
        assert_eq!(arena.prefill_range(), 0..0);
        assert_eq!(arena.gap(), 100);
        assert_eq!(arena.cache_fill(100), Some(0..100), "the cache refills from the released run");
    }

    #[test]
    fn a_borrow_that_fits_the_gap_evicts_nothing() {
        let mut arena = SlotArena::new(100);
        arena.cache_fill(40);
        assert_eq!(arena.cache_range(), 60..100);
        let taken = arena.borrow(60).unwrap();
        assert!(taken.evict.is_empty(), "60 slots fit exactly under the cache's low bound");
        assert_eq!(arena.cache_range(), 60..100);
        assert_eq!(arena.gap(), 0);
    }

    #[test]
    fn over_borrowing_is_refused_and_leaves_the_arena_untouched() {
        let mut arena = SlotArena::new(50);
        arena.cache_fill(50);
        assert_eq!(arena.borrow(51), Err(ArenaError::TooLarge { want: 51, total: 50 }));
        assert_eq!(arena.cache_range(), 0..50);
        assert_eq!(arena.prefill_range(), 0..0);
    }

    #[test]
    fn the_watermarks_never_cross() {
        let mut arena = SlotArena::new(64);
        arena.cache_fill(64);
        for want in [0, 1, 8, 63, 64, 0, 32, 64] {
            arena.borrow(want).unwrap();
            assert!(arena.prefill_high <= arena.cache_low);
            assert!(arena.cache_low <= arena.total);
        }
        arena.release();
        arena.cache_fill(64);
        assert_eq!(arena.cache_range(), 0..64);
    }
}
