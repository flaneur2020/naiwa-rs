//! `NAIWA_PRED_DBG`: how well the lookahead's predicted expert set matches the routing the layer
//! actually took.
//!
//! The prediction is the CPU router matmul run one layer ahead (`CachedLayer::predict`). It is the
//! only thing in the engine that knows a layer's experts before the top-k doorbell returns, so it
//! bounds two open questions at once: how much of the CPU expert job could be computed early (the
//! window), and how much of the lookahead's speculative H2D could not have helped (the traffic).

use std::sync::Mutex;

use super::{header_once, Probe, Window};

use std::sync::atomic::AtomicBool;

#[derive(Default)]
pub(crate) struct PredStats {
    /// Passes with both a prediction and an actual set.
    pub passes: u64,
    /// Sum over passes of |predicted ∩ actual|.
    pub overlap: u64,
    /// Sum over passes of |actual|.
    pub actual: u64,
    /// Passes whose predicted set equals the actual set (as sets).
    pub exact: u64,
    /// Predicted ids that were already resident when the prefetch ran: copies that could not help.
    pub wasted: u64,
    /// Sum over passes of |predicted|.
    pub predicted: u64,
}

static PRED_STATS: Mutex<PredStats> = Mutex::new(PredStats {
    passes: 0,
    overlap: 0,
    actual: 0,
    exact: 0,
    wasted: 0,
    predicted: 0,
});
static PROBE: Probe = Probe::new(|d| d.pred_dbg);
static WINDOW: Window = Window::new(48); // one pass over the model's layers
static HEADER: AtomicBool = AtomicBool::new(false);

/// One slot per layer: the prediction published by the previous layer's lookahead job, waiting for
/// this layer's routing to arrive so the two can be compared. Empty means "no prediction".
static PENDING: Mutex<Vec<Option<Vec<u32>>>> = Mutex::new(Vec::new());

#[inline]
pub(crate) fn enabled() -> bool {
    PROBE.enabled(false)
}

/// The lookahead predicted `ids` for `layer`. `already_resident` is how many of them the prefetch
/// found in the cache already (it copies nothing for those).
pub(crate) fn record_predicted(layer: usize, ids: &[usize], already_resident: usize) {
    if !enabled() {
        return;
    }
    let mut ids: Vec<u32> = ids.iter().map(|&e| e as u32).collect();
    // Dedupe before counting: a prediction lists top_k per row, so a verify window repeats experts
    // and `wasted` would otherwise be counted against a deduped denominator.
    ids.sort_unstable();
    ids.dedup();
    let mut pending = PENDING.lock().unwrap();
    if pending.len() <= layer {
        pending.resize(layer + 1, None);
    }
    pending[layer] = Some(ids);
    PRED_STATS.lock().unwrap().wasted += already_resident as u64;
}

/// The layer's real top-k arrived. Compare against the stored prediction, once per pass.
pub(crate) fn record_actual(layer: usize, ids_rows: &[Vec<u32>]) {
    if !enabled() {
        return;
    }
    let predicted = {
        let mut pending = PENDING.lock().unwrap();
        match pending.get_mut(layer).and_then(Option::take) {
            Some(v) => v,
            None => return,
        }
    };
    // The actual demand of the pass: the union over rows (a verify window has several rows).
    let mut actual = ids_rows.concat();
    actual.sort_unstable();
    actual.dedup();
    let mut pred = predicted;
    pred.sort_unstable();
    pred.dedup();
    let overlap = pred.iter().filter(|e| actual.binary_search(e).is_ok()).count();
    let exact = u64::from(overlap == pred.len() && overlap == actual.len());

    let due = {
        let mut s = PRED_STATS.lock().unwrap();
        s.passes += 1;
        s.overlap += overlap as u64;
        s.actual += actual.len() as u64;
        s.predicted += pred.len() as u64;
        s.exact += exact;
        WINDOW.tick() == WINDOW.size()
    };
    if due {
        let _ = super::dump_predict_stats();
    }
}

/// Take and reset the window, print the `PRED_DBG` line, and return the snapshot.
pub(crate) fn dump_predict_stats() -> PredStats {
    let out = std::mem::take(&mut *PRED_STATS.lock().unwrap());
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
    header_once(
        &HEADER,
        "PRED_DBG passes overlap% exact% wasted% predicted/actual",
    );
    eprintln!(
        "PRED_DBG {} {:.1} {:.1} {:.1} {:.2}",
        out.passes,
        pct(out.overlap, out.actual),
        pct(out.exact, out.passes),
        pct(out.wasted, out.predicted),
        out.predicted as f64 / out.actual.max(1) as f64,
    );
    PredStats { ..out }
}
