//! Fused CPU MoE decode over stacked GGUF expert weights: the routed experts of a few activation
//! rows (one decode token, or the rows of a speculative verify step) in two parallel passes.
//!
//! The generic path (`MoEExperts::forward` -> three `gather_forward`s) runs gate, up and down as
//! three separate routed gemvs, each quantizing its input and dispatching its own rayon tasks, with
//! the activation, the product and the router-weighted sum done between them as candle CPU ops on
//! bf16 tensors. At decode sizes (a few experts per layer) that structure, not the dot kernels,
//! dominates: a routed gemv costs ~45 us per call before any dot. Here
//!
//! 1. each activation row is quantized once for gate and up, and one dispatch computes
//!    `silu(gate . x) * (up . x)` for every distinct (expert, row) pair into an f32 buffer;
//! 2. one dispatch over (row, block of output rows) accumulates `w * (down . h)` for every slot of
//!    the row straight into the `[rows, hidden]` output, so the weighted sum costs nothing extra.
//!
//! Slots with weight 0 (the caller's padding to a common width) are skipped. Intermediates stay
//! f32, so results match the generic path to within its bf16 rounding, not bit for bit.

use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
use candle_core::Result;
use rayon::prelude::*;

/// The pool the fused decode runs on. Its calls are short, latency-bound dispatches that arrive
/// once per layer with GPU work in between, so the global pool's physical-core default leaves the
/// SMT siblings idle: 10 threads on a 6-core/12-thread host decode 8-14% faster per step than the
/// 6-thread default, and beat 8 and 11 as well (worklog W92). Defaults to the logical CPUs minus
/// two, one for the decode thread and one for the expert-cache lookahead;
/// `NAIWA_CPU_MOE_THREADS=N` overrides it and `0` keeps the global pool.
fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("NAIWA_CPU_MOE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            // Measured on a 6C/12T Zen2 (np8 verify, one job): 1 thread 5057 us, 4 -> 2066,
            // 6 -> 1816, **8 -> 1395**, 10 -> 1515. The default used to be `available - 2` (= 10
            // here), i.e. past the optimum: SMT siblings add little for these dots. Cap at 8.
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(0, |n| n.get().saturating_sub(2).min(8))
            });
        (n > 0)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("cpu-moe-{i}"))
                    .build()
                    .ok()
            })
            .flatten()
    })
    .as_ref()
}

/// Runs `f(0..n)` in parallel over the tasks.
///
/// An engine may install its own dispatcher with [`install_task_dispatcher`]: a pinned spin
/// pool that keeps its workers hot between a decode step's short, once-per-layer calls (the
/// rayon pool parks in between and pays a wake per call, worklog W92). While installed it
/// replaces the rayon pool here entirely; `NAIWA_CPU_MOE_DISPATCH=rayon` keeps the rayon
/// path for A/B runs.
static CUSTOM_DISPATCHER: std::sync::OnceLock<Option<Dispatcher>> = std::sync::OnceLock::new();

/// The signature of a task dispatcher: run `f(0..n)` on the engine's own pool.
pub type Dispatch = fn(n: usize, f: &(dyn Fn(usize) + Sync + Send));

struct Dispatcher {
    dispatch: Dispatch,
}

/// Install the engine's task dispatcher for the fused decode's two passes. Pass `None` to
/// fall back to the rayon pool.
pub fn install_task_dispatcher(dispatch: Option<Dispatch>) {
    let _ = CUSTOM_DISPATCHER.set(dispatch.map(|dispatch| Dispatcher { dispatch }));
}

fn par_tasks(n: usize, f: impl Fn(usize) + Sync + Send) {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*TRACE.get_or_init(|| std::env::var_os("NAIWA_DISPATCH_TRACE").is_some()) {
        return par_tasks_inner(n, f);
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    let first = AtomicU64::new(u64::MAX);
    let last = AtomicU64::new(0);
    let longest = AtomicU64::new(0);
    let work = AtomicU64::new(0);
    let origin = std::time::Instant::now();
    par_tasks_inner(n, |i| {
        let start = origin.elapsed().as_nanos() as u64;
        first.fetch_min(start, Ordering::Relaxed);
        f(i);
        let end = origin.elapsed().as_nanos() as u64;
        last.fetch_max(end, Ordering::Relaxed);
        longest.fetch_max(end - start, Ordering::Relaxed);
        work.fetch_add(end - start, Ordering::Relaxed);
    });
    let total = origin.elapsed().as_nanos() as u64;
    let last = last.load(Ordering::Relaxed);
    eprintln!("DISPATCH tasks={n} first_us={} span_us={} tail_us={} max_task_us={} sum_task_us={}",
        first.load(Ordering::Relaxed) / 1000, total / 1000,
        total.saturating_sub(last) / 1000, longest.load(Ordering::Relaxed) / 1000,
        work.load(Ordering::Relaxed) / 1000);
}

fn par_tasks_inner(n: usize, f: impl Fn(usize) + Sync + Send) {
    // Read once: this runs twice per fused call, once per layer.
    static KEEP_RAYON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let keep_rayon = *KEEP_RAYON
        .get_or_init(|| std::env::var_os("NAIWA_CPU_MOE_DISPATCH").map_or(false, |v| v == "rayon"));
    if !keep_rayon {
        if let Some(d) = CUSTOM_DISPATCHER.get().and_then(|d| d.as_ref()) {
            (d.dispatch)(n, &f);
            return;
        }
    }
    match pool() {
        Some(pool) => pool.install(|| (0..n).into_par_iter().for_each(&f)),
        None => (0..n).into_par_iter().for_each(&f),
    }
}

/// Output rows per rayon task: the same granularity as the routed gemv.
/// Output rows per task within one expert's run. A task touches `RUN_ROWS` consecutive rows of
/// one expert before hopping to the next slot's expert (460 KB away): W101 phases measured the
/// short 32-row runs (5.7 KB) at ~1/3 of strata's DRAM bandwidth, because a run that short
/// never amortises the DRAM row activation. Override for sweeps with NAIWA_RUN_ROWS.
fn run_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NAIWA_RUN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0 && n.is_multiple_of(8))
            // 64 measured better than 32 on the 12 GB test box: 78.8 ms vs 80.4 ms per pass with an
            // unchanged hit rate (84.3%), because a longer run amortises the DRAM row activation of a
            // 2 MB expert stack read across more rows. 8 was worse (81.9 ms).
            .unwrap_or(64)
    })
}

/// `NAIWA_FUSED_PHASES=1` (default on): per-call wall times of a fused decode's stages, so the
/// module-level A/B (scripts/strata-ab) can split our arm the way strata's pool splits itself
/// (`ms_multi_gu/q/down`). Read with [`last_phases`] after a call; a no-op struct when off.
///
/// Stages: pair/slot build (host, inside `fused_moe_decode`), act quant (inside `gate_up`),
/// pass 1's dispatch, h quant (inside `down_sum`), pass 2's dispatch.
#[derive(Clone, Default)]
pub struct Phases {
    pub on: bool,
    pub us_pairs: f64,
    pub us_act: f64,
    pub us_pass1: f64,
    pub us_hq: f64,
    pub us_pass2: f64,
}

thread_local! {
    static LAST: std::cell::RefCell<Phases> = const { std::cell::RefCell::new(Phases {
        on: false,
        us_pairs: 0.0,
        us_act: 0.0,
        us_pass1: 0.0,
        us_hq: 0.0,
        us_pass2: 0.0,
    }) };
}

/// The stages of the most recent `fused_moe_decode` on this thread (zeros when the env is
/// off, and until the first call).
pub fn last_phases() -> Phases {
    LAST.with(|p| p.borrow().clone())
}

fn phases_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NAIWA_FUSED_PHASES").map_or(true, |v| v != "0"))
}

impl Phases {
    /// Elapsed µs if the phase split is on; 0.0 keeps the whole struct quiet when off.
    fn elapsed_us(&self, from: std::time::Instant) -> f64 {
        if self.on {
            from.elapsed().as_secs_f64() * 1e6
        } else {
            0.0
        }
    }
}

/// Raw output pointer shared by tasks that write disjoint elements.
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// SAFETY: callers write each index from exactly one task.
    #[inline]
    unsafe fn set(&self, index: usize, value: f32) {
        *self.0.add(index) = value;
    }
}

/// The stacked blocks of `qt` as `&[T]`, checked against its `[E, n, k]` shape.
fn blocks_of<T: GgmlType>(data: &[u8], (e, n, k): (usize, usize, usize)) -> Result<&[T]> {
    let bytes = std::mem::size_of::<T>();
    if k % T::BLCK_SIZE != 0
        || data.len() != e * n * (k / T::BLCK_SIZE) * bytes
        || data.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
    {
        candle_core::bail!("expert stack bytes do not match its shape");
    }
    // SAFETY: `T` is a `repr(C)` ggml block; length and alignment checked above.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<T>(), data.len() / bytes) })
}

/// Pass 1: `h[p] = silu(gate[e] . x[t]) * (up[e] . x[t])` for each pair `p = (e, t)`.
fn gate_up<G: GgmlType>(
    gate: &[u8],
    up: &[u8],
    (e, inter, hidden): (usize, usize, usize),
    xs: &[f32],
    pairs: &[(u32, usize)],
    ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    if hidden % G::VecDotType::BLCK_SIZE != 0 {
        return Ok(None);
    }
    let (gate_raw, up_raw) = (gate, up);
    let gate = blocks_of::<G>(gate, (e, inter, hidden))?;
    let up = blocks_of::<G>(up, (e, inter, hidden))?;
    let row_blocks = hidden / G::BLCK_SIZE;
    let expert_blocks = inter * row_blocks;

    // Each row's activation, quantized once for both projections.
    let vd = hidden / G::VecDotType::BLCK_SIZE;
    let rows = xs.len() / hidden;
    let mut used = vec![false; rows];
    for &(_, t) in pairs {
        used[t] = true;
    }
    let t0 = std::time::Instant::now();
    let mut xq = vec![G::VecDotType::zeros(); rows * vd];
    for t in 0..rows {
        if used[t] {
            G::VecDotType::from_float(
                &xs[t * hidden..(t + 1) * hidden],
                &mut xq[t * vd..(t + 1) * vd],
            );
        }
    }
    ph.us_act = ph.elapsed_us(t0);

    let mut h = vec![0f32; pairs.len() * inter];
    let out = Out(h.as_mut_ptr());
    let rr = run_rows();
    let chunks = inter.div_ceil(rr);
    // One task owns a whole expert's run of pairs (pairs are sorted by expert). When that run is
    // two rows or longer, IQ2_S dots both in one pass over the weights: the grid lookups are
    // weight-only, so the second row is nearly free (M36 §12). `NAIWA_NO_X2=1` is the A/B arm and
    // `NAIWA_NO_EXPERT_GROUPING=1` restores per-pair tasks.
    let groups: Vec<(usize, usize)> = if std::env::var_os("NAIWA_NO_EXPERT_GROUPING").is_some() {
        (0..pairs.len()).map(|p| (p, p + 1)).collect()
    } else {
        let mut g = Vec::with_capacity(pairs.len());
        let mut start = 0;
        while start < pairs.len() {
            let expert = pairs[start].0;
            let mut end = start + 1;
            while end < pairs.len() && pairs[end].0 == expert {
                end += 1;
            }
            g.push((start, end));
            start = end;
        }
        g
    };
    #[cfg(target_arch = "x86_64")]
    let two_row = has_x2(G::DTYPE);
    par_tasks(groups.len() * chunks, |task| {
        let (gi, c) = (task / chunks, task % chunks);
        let (start, end) = groups[gi];
        let base = pairs[start].0 as usize * expert_blocks;
        let r0 = c * rr;
        for r in r0..(r0 + rr).min(inter) {
            let at = base + r * row_blocks;
            let g_row = &gate[at..at + row_blocks];
            let u_row = &up[at..at + row_blocks];
            #[cfg(target_arch = "x86_64")]
            if two_row && end - start >= 2 {
                let bs = std::mem::size_of::<G>();
                // SAFETY: `G` is `BlockIQ2S` here (checked above), repr(C) and unpadded.
                let (gb, ub) = unsafe {
                    (
                        block_bytes(&gate_raw[at * bs..(at + row_blocks) * bs]),
                        block_bytes(&up_raw[at * bs..(at + row_blocks) * bs]),
                    )
                };
                let mut p = start;
                while p + 1 < end {
                    let (t0, t1) = (pairs[p].1, pairs[p + 1].1);
                    let x0 = &xq[t0 * vd..(t0 + 1) * vd];
                    let x1 = &xq[t1 * vd..(t1 + 1) * vd];
                    // SAFETY: as above.
                    let pair =
                        unsafe { x2_row_pair(G::DTYPE, gb, ub, block_bytes(x0), block_bytes(x1)) };
                    match pair {
                        Some((g0, u0, g1, u1)) => {
                            for (pp, g, u) in [(p, g0, u0), (p + 1, g1, u1)] {
                                let silu = g / (1.0 + (-g).exp());
                                // SAFETY: (pp, r) is written by this task only.
                                unsafe { out.set(pp * inter + r, silu * u) };
                            }
                            p += 2;
                        }
                        None => {
                            let g = G::vec_dot(hidden, g_row, x0);
                            let u = G::vec_dot(hidden, u_row, x0);
                            let silu = g / (1.0 + (-g).exp());
                            // SAFETY: (p, r) is written by this task only.
                            unsafe { out.set(p * inter + r, silu * u) };
                            p += 1;
                        }
                    }
                }
                if p < end {
                    let x = &xq[pairs[p].1 * vd..(pairs[p].1 + 1) * vd];
                    let g = G::vec_dot(hidden, g_row, x);
                    let u = G::vec_dot(hidden, u_row, x);
                    let silu = g / (1.0 + (-g).exp());
                    // SAFETY: (p, r) is written by this task only.
                    unsafe { out.set(p * inter + r, silu * u) };
                }
                continue;
            }
            for p in start..end {
                let x = &xq[pairs[p].1 * vd..(pairs[p].1 + 1) * vd];
                let g = G::vec_dot(hidden, g_row, x);
                let u = G::vec_dot(hidden, u_row, x);
                let silu = g / (1.0 + (-g).exp());
                // SAFETY: (p, r) is written by this task only.
                unsafe { out.set(p * inter + r, silu * u) };
            }
        }
    });
    Ok(Some(h))
}

/// Pass 2: `out[t] = sum_j w[t, j] * (down[e] . h[pair(t, j)])`.
#[allow(clippy::too_many_arguments)]
fn down_sum<D: GgmlType>(
    down: &[u8],
    (e, hidden, inter): (usize, usize, usize),
    h: &[f32],
    pairs: &[(u32, usize)],
    slots: &[Vec<(usize, f32)>],
    ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    if inter % D::VecDotType::BLCK_SIZE != 0 {
        return Ok(None);
    }
    let down = blocks_of::<D>(down, (e, hidden, inter))?;
    let row_blocks = inter / D::BLCK_SIZE;
    let expert_blocks = hidden * row_blocks;
    let vd = inter / D::VecDotType::BLCK_SIZE;
    let t0 = std::time::Instant::now();
    let mut hq = vec![D::VecDotType::zeros(); pairs.len() * vd];
    for p in 0..pairs.len() {
        D::VecDotType::from_float(
            &h[p * inter..(p + 1) * inter],
            &mut hq[p * vd..(p + 1) * vd],
        );
    }
    ph.us_hq = ph.elapsed_us(t0);

    let rows = slots.len();
    let mut dst = vec![0f32; rows * hidden];
    let out = Out(dst.as_mut_ptr());
    let rr = run_rows();
    let chunks = hidden.div_ceil(rr);
    par_tasks(rows * chunks, |task| {
        let (t, c) = (task / chunks, task % chunks);
        let r0 = c * rr;
        let end = (r0 + rr).min(hidden);
        let n = end - r0;
        let mut acc: Box<[f32]> = vec![0.0; n].into_boxed_slice();
        for &(p, w) in &slots[t] {
            let e = pairs[p].0 as usize;
            let xq = &hq[p * vd..(p + 1) * vd];
            for (k, r) in (r0..end).enumerate() {
                let at = e * expert_blocks + r * row_blocks;
                acc[k] += w * D::vec_dot(inter, &down[at..at + row_blocks], xq);
            }
        }
        for (k, r) in (r0..end).enumerate() {
            // SAFETY: (t, r) is written by this task only.
            unsafe { out.set(t * hidden + r, acc[k]) };
        }
    });
    Ok(Some(dst))
}

// ---- the Q2_0 down rows (W102) ----------------------------------------------------------
//
// W101's module A/B put our pass 2 at 642 us/layer against strata's 219 with identical bytes
// and dot counts — the per-call `vec_dot` structure, not the scheduling. Each 180-byte row
// paid the call overhead, the serial per-block `sumf` FMA chain and the per-call horizontal
// sum (~253 ns/row); strata's `q2_rows_any` (pool.cpp:651 -> q2_avx2_rows.inl) unpacks a
// row's 2-bit codes once and FMAs whole blocks into a vector accumulator, the `-1` code
// offset as one scalar correction at row end: 53 ns/row measured in dot_micro (vs 122 for
// the per-call form, 2.3x). Same arithmetic, different summation order.
//
// The activation stays in ActQ form (int8 codes over 32-value chunks, an f32 scale — no f16
// rounding — and `hx = scale * sum(q)`, the exact per-chunk correction), x86-64/AVX2+FMA
// only; other arches and `NAIWA_NO_Q2ROWS=1` fall back to the generic path above.
#[cfg(target_arch = "x86_64")]
mod q2_rows {
    use std::arch::x86_64::*;

    use crate::gguf::q2_0::BLOCK_BYTES;

    /// One pair's quantized `h`: int8 codes per 32-value chunk, the chunk scale (f32) and the
    /// chunk's `scale * sum(q)` — the subtraction term that folds the `-1` code offset in.
    pub struct ActQ {
        pub q: Vec<i8>,
        pub scale: Vec<f32>,
        pub hx: Vec<f32>,
        pub nch: usize,
    }

    /// Quantize `x` the way `quantize_row_q8_0` does (amax/127, round, clamp), keeping the
    /// scale as f32 and recording `hx` per chunk.
    pub fn quant_act(x: &[f32], nch: usize) -> ActQ {
        let mut q = vec![0i8; nch * 32];
        let mut scale = vec![0f32; nch];
        let mut hx = vec![0f32; nch];
        for c in 0..nch {
            let xb = &x[c * 32..(c + 1) * 32];
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut sum = 0i32;
            for (j, &v) in xb.iter().enumerate() {
                let v = (v * id + if v >= 0.0 { 0.5 } else { -0.5 }).clamp(-127.0, 127.0) as i8;
                q[c * 32 + j] = v;
                sum += v as i32;
            }
            scale[c] = s;
            hx[c] = s * sum as f32;
        }
        ActQ { q, scale, hx, nch }
    }

    /// f16 -> f32 on F16C (every AVX2 CPU has it), as strata's `h2f` does.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn h2f(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    /// 16 code bytes -> 64 codes in value order (two 32-byte vectors); strata's `unpack64`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack64(codes: &[u8], lo: &mut __m256i, hi: &mut __m256i) {
        let b = _mm_loadu_si128(codes.as_ptr() as *const __m128i);
        let m3 = _mm_set1_epi8(3);
        let c0 = _mm_and_si128(b, m3);
        let c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
        let c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
        let c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
        let a0 = _mm_unpacklo_epi8(c0, c1);
        let a1 = _mm_unpacklo_epi8(c2, c3);
        let b0 = _mm_unpackhi_epi8(c0, c1);
        let b1 = _mm_unpackhi_epi8(c2, c3);
        *lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));
        *hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));
    }

    /// One Q2_0 weight row (as raw bytes) against one pair's ActQ: the `row_multi<1>` shape —
    /// vector accumulate per block, one horizontal sum and one scalar correction per row.
    ///
    /// `row` is `row_blocks` consecutive Q2_0 blocks; `a` is chunk-indexed (`2 * row_blocks`
    /// chunks of 32). Codes are 0..3 and the weight is `code - 1`, so
    /// `sum((code-1) * d * q * s) = d * sum(s * sum(code*q) - hx)` per chunk pair.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn row_dot(row: &[u8], row_blocks: usize, a: &ActQ, p: usize) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        let mut corr = 0f32;
        let q = a.q.as_ptr().add(p * a.nch * 32);
        let sc = &a.scale[p * a.nch..];
        let hx = &a.hx[p * a.nch..];
        for b in 0..row_blocks {
            let blk = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let d = h2f(u16::from_le_bytes([blk[0], blk[1]]));
            let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            unpack64(&blk[2..], &mut lo, &mut hi);
            let s0 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(lo, _mm256_loadu_si256(q.add(b * 64) as *const __m256i)),
                ones,
            );
            let s1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(hi, _mm256_loadu_si256(q.add(b * 64 + 32) as *const __m256i)),
                ones,
            );
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[2 * b]), _mm256_cvtepi32_ps(s0), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc);
            corr += d * (hx[2 * b] + hx[2 * b + 1]);
        }
        let h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr
    }

    pub const ROW_BYTES_FACTOR: usize = BLOCK_BYTES; // bytes per 64-value block
}

/// One Q2_0 block's element count, re-exported for the shape check below.
#[cfg(target_arch = "x86_64")]
use crate::gguf::q2_0::QK2_0 as Q2_0_QK;

/// Pass 2 for a Q2_0 down stack on the AVX2 row kernel. `None` when the shape does not fit
/// or the CPU lacks AVX2+FMA (`NAIWA_NO_Q2ROWS=1` forces the fallback for A/B runs).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn down_sum_q2_0(
    down: &[u8],
    (e, hidden, inter): (usize, usize, usize),
    h: &[f32],
    pairs: &[(u32, usize)],
    slots: &[Vec<(usize, f32)>],
    ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    if inter % Q2_0_QK != 0
        || down.len() != e * hidden * (inter / Q2_0_QK) * q2_rows::ROW_BYTES_FACTOR
        || !q2_rows_ok()
    {
        return Ok(None);
    }
    let row_blocks = inter / Q2_0_QK; // Q2_0 blocks per down row
    let row_bytes = row_blocks * q2_rows::ROW_BYTES_FACTOR;
    let expert_bytes = hidden * row_bytes;

    let t0 = std::time::Instant::now();
    let nch = inter / 32;
    let acts: Vec<q2_rows::ActQ> = (0..pairs.len())
        .map(|p| q2_rows::quant_act(&h[p * inter..(p + 1) * inter], nch))
        .collect();
    ph.us_hq = ph.elapsed_us(t0);

    let rows = slots.len();
    let mut dst = vec![0f32; rows * hidden];
    let out = Out(dst.as_mut_ptr());
    let rr = run_rows();
    let chunks = hidden.div_ceil(rr);
    par_tasks(rows * chunks, |task| {
        let (t, c) = (task / chunks, task % chunks);
        let r0 = c * rr;
        let end = (r0 + rr).min(hidden);
        let n = end - r0;
        let mut acc: Box<[f32]> = vec![0.0; n].into_boxed_slice();
        for &(p, w) in &slots[t] {
            let base = pairs[p].0 as usize * expert_bytes;
            let a = &acts[p];
            for (k, r) in (r0..end).enumerate() {
                let row = &down[base + r * row_bytes..base + (r + 1) * row_bytes];
                // SAFETY: pure function of the weights and this pair's activation.
                acc[k] += w * unsafe { q2_rows::row_dot(row, row_blocks, a, 0) };
            }
        }
        for (k, r) in (r0..end).enumerate() {
            // SAFETY: (t, r) is written by this task only.
            unsafe { out.set(t * hidden + r, acc[k]) };
        }
    });
    Ok(Some(dst))
}

#[cfg(target_arch = "x86_64")]
fn q2_rows_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::env::var_os("NAIWA_NO_Q2ROWS").is_none()
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn down_sum_q2_0(
    _down: &[u8],
    _dims: (usize, usize, usize),
    _h: &[f32],
    _pairs: &[(u32, usize)],
    _slots: &[Vec<(usize, f32)>],
    _ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    Ok(None)
}

// ---- the IQ4_NL down rows (W103) ----------------------------------------------------------
//
// L24's pass 2 ran the generic per-row `vec_dot_q8_0` (124 ns/row on the L24 stack, dot_micro E):
// per-call feature detect, length checks and two `bail!` paths per 360-byte row. The same
// arithmetic in a row kernel with ggml's 2-block unroll — two independent vector accumulators,
// one f16 scale decode per block — measures 95 ns/row (dot_micro H), and strata's
// `iq4nl256_down_rows` is the same shape. The activation is plain Q8_0 (f16 d + 32 codes per
// chunk, exactly what the generic path's `hq` holds). x86-64/AVX2+FMA only; other arches and
// `NAIWA_NO_I4NLROWS=1` fall back to the generic path.
#[cfg(target_arch = "x86_64")]
mod i4nl_rows {
    use std::arch::x86_64::*;

    use crate::gguf::iq4nl::KVALUES_IQ4NL;

    pub const QK: usize = 32; // values per IQ4_NL block
    pub const BLOCK_BYTES: usize = 18; // f16 d + 16 nibble bytes
    pub const Q8_CHUNK_BYTES: usize = 34; // f16 d + 32 codes, per Q8_0 chunk

    /// f16 -> f32 on F16C, as the Q2_0 row kernel's `h2f` does.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn h2f(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    /// One IQ4_NL weight row (`row_blocks` consecutive 18-byte blocks) against one Q8_0
    /// activation (`row_blocks` consecutive 34-byte chunks): ggml's 2-block unroll with two
    /// independent accumulators, one hsum per row.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn row_dot(row: &[u8], row_blocks: usize, hq: &[u8]) -> f32 {
        let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let m4b = _mm_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let w = row.as_ptr();
        let q = hq.as_ptr();
        let (mut acc1, mut acc2) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut b = 0;
        while b + 1 < row_blocks {
            for (bb, acc) in [(b, &mut acc1), (b + 1, &mut acc2)] {
                let blk = w.add(bb * BLOCK_BYTES);
                let qb = q.add(bb * Q8_CHUNK_BYTES);
                let bits = _mm_loadu_si128(blk.add(2) as *const __m128i);
                let lo = _mm_and_si128(bits, m4b);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(bits), m4b);
                let q4 = _mm256_set_m128i(_mm_shuffle_epi8(values, hi), _mm_shuffle_epi8(values, lo));
                let aq = _mm256_sign_epi8(q4, q4); // |w|: the unsigned operand of maddubs
                let q8 = _mm256_loadu_si256(qb.add(2) as *const __m256i);
                let p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)),
                    ones,
                );
                let d = h2f(u16::from_le_bytes([*blk, *blk.add(1)]))
                    * h2f(u16::from_le_bytes([*qb, *qb.add(1)]));
                *acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p), *acc);
            }
            b += 2;
        }
        let mut acc = _mm256_add_ps(acc1, acc2);
        if b < row_blocks {
            let blk = w.add(b * BLOCK_BYTES);
            let qb = q.add(b * Q8_CHUNK_BYTES);
            let bits = _mm_loadu_si128(blk.add(2) as *const __m128i);
            let lo = _mm_and_si128(bits, m4b);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bits), m4b);
            let q4 = _mm256_set_m128i(_mm_shuffle_epi8(values, hi), _mm_shuffle_epi8(values, lo));
            let aq = _mm256_sign_epi8(q4, q4);
            let q8 = _mm256_loadu_si256(qb.add(2) as *const __m256i);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
            let d = h2f(u16::from_le_bytes([*blk, *blk.add(1)]))
                * h2f(u16::from_le_bytes([*qb, *qb.add(1)]));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p), acc);
        }
        let hd = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let s = _mm_add_ps(hd, _mm_movehl_ps(hd, hd));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }
}

/// Pass 2 for an IQ4_NL down stack on the AVX2 row kernel. `None` when the shape does not fit
/// or the CPU lacks AVX2+FMA (`NAIWA_NO_I4NLROWS=1` forces the fallback for A/B runs).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn down_sum_i4nl(
    down: &[u8],
    (e, hidden, inter): (usize, usize, usize),
    h: &[f32],
    pairs: &[(u32, usize)],
    slots: &[Vec<(usize, f32)>],
    ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    if inter % i4nl_rows::QK != 0
        || down.len() != e * hidden * (inter / i4nl_rows::QK) * i4nl_rows::BLOCK_BYTES
        || !i4nl_rows_ok()
    {
        return Ok(None);
    }
    let row_blocks = inter / i4nl_rows::QK;
    let row_bytes = row_blocks * i4nl_rows::BLOCK_BYTES;
    let expert_bytes = hidden * row_bytes;

    let t0 = std::time::Instant::now();
    let nch = inter / i4nl_rows::QK;
    // Q8_0 activation per pair: f16 d + 32 codes per 32-value chunk, the generic path's `hq`
    // layout (quantize amax/127, round, clamp).
    let mut hq = vec![0u8; pairs.len() * nch * i4nl_rows::Q8_CHUNK_BYTES];
    for p in 0..pairs.len() {
        for c in 0..nch {
            let xb = &h[p * inter + c * 32..p * inter + (c + 1) * 32];
            let out = &mut hq[p * nch * i4nl_rows::Q8_CHUNK_BYTES + c * i4nl_rows::Q8_CHUNK_BYTES..];
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if s > 0.0 { 1.0 / s } else { 0.0 };
            out[..2].copy_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
            for (j, &v) in xb.iter().enumerate() {
                out[2 + j] = (v * id + if v >= 0.0 { 0.5 } else { -0.5 }).clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }
    ph.us_hq = ph.elapsed_us(t0);

    let rows = slots.len();
    let mut dst = vec![0f32; rows * hidden];
    let out = Out(dst.as_mut_ptr());
    let rr = run_rows();
    let chunks = hidden.div_ceil(rr);
    par_tasks(rows * chunks, |task| {
        let (t, c) = (task / chunks, task % chunks);
        let r0 = c * rr;
        let end = (r0 + rr).min(hidden);
        let n = end - r0;
        let mut acc: Box<[f32]> = vec![0.0; n].into_boxed_slice();
        for &(p, w) in &slots[t] {
            let base = pairs[p].0 as usize * expert_bytes;
            let q = &hq[p * nch * i4nl_rows::Q8_CHUNK_BYTES..(p + 1) * nch * i4nl_rows::Q8_CHUNK_BYTES];
            for (k, r) in (r0..end).enumerate() {
                let row = &down[base + r * row_bytes..base + (r + 1) * row_bytes];
                // SAFETY: pure function of the weights and this pair's activation.
                acc[k] += w * unsafe { i4nl_rows::row_dot(row, row_blocks, q) };
            }
        }
        for (k, r) in (r0..end).enumerate() {
            // SAFETY: (t, r) is written by this task only.
            unsafe { out.set(t * hidden + r, acc[k]) };
        }
    });
    Ok(Some(dst))
}

#[cfg(target_arch = "x86_64")]
fn i4nl_rows_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::env::var_os("NAIWA_NO_I4NLROWS").is_none()
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn down_sum_i4nl(
    _down: &[u8],
    _dims: (usize, usize, usize),
    _h: &[f32],
    _pairs: &[(u32, usize)],
    _slots: &[Vec<(usize, f32)>],
    _ph: &mut Phases,
) -> Result<Option<Vec<f32>>> {
    Ok(None)
}

/// Runs `$body` with `$T` bound to the block type of `$dtype`; `None` for types without a CPU
/// vec-dot here.
macro_rules! with_block {
    ($dtype:expr, $T:ident, $body:expr) => {{
        use super::iq2s::BlockIQ2S;
        use super::iq3s::BlockIQ3S;
        use super::iq3xxs::BlockIQ3XXS;
        use super::iq4nl::BlockIQ4NL;
        use super::iq4xs::BlockIQ4XS;
        use super::q2_0::BlockQ2_0;
        use candle_core::quantized::k_quants::*;
        match $dtype {
            GgmlDType::Q4_0 => {
                type $T = BlockQ4_0;
                $body
            }
            GgmlDType::Q4_1 => {
                type $T = BlockQ4_1;
                $body
            }
            GgmlDType::Q5_0 => {
                type $T = BlockQ5_0;
                $body
            }
            GgmlDType::Q5_1 => {
                type $T = BlockQ5_1;
                $body
            }
            GgmlDType::Q8_0 => {
                type $T = BlockQ8_0;
                $body
            }
            GgmlDType::Q2K => {
                type $T = BlockQ2K;
                $body
            }
            GgmlDType::Q3K => {
                type $T = BlockQ3K;
                $body
            }
            GgmlDType::Q4K => {
                type $T = BlockQ4K;
                $body
            }
            GgmlDType::Q5K => {
                type $T = BlockQ5K;
                $body
            }
            GgmlDType::Q6K => {
                type $T = BlockQ6K;
                $body
            }
            GgmlDType::IQ4XS => {
                type $T = BlockIQ4XS;
                $body
            }
            GgmlDType::IQ4NL => {
                type $T = BlockIQ4NL;
                $body
            }
            GgmlDType::Q2_0 => {
                type $T = BlockQ2_0;
                $body
            }
            GgmlDType::IQ3_XXS => {
                type $T = BlockIQ3XXS;
                $body
            }
            GgmlDType::IQ2_S => {
                type $T = BlockIQ2S;
                $body
            }
            GgmlDType::IQ3_S => {
                type $T = BlockIQ3S;
                $body
            }
            _ => Ok(None),
        }
    }};
}


/// Two activations dotted against one expert's weight row in a single pass over the weights
/// (IQ2_S only: its grid lookups are weight-only, so the second row is nearly free). `None` means
/// the caller should fall back to two single dots - `NAIWA_NO_X2=1` is the A/B arm.
#[cfg(target_arch = "x86_64")]
fn x2_row_pair(
    dtype: GgmlDType,
    g_row: &[u8],
    u_row: &[u8],
    q0: &[u8],
    q1: &[u8],
) -> Option<(f32, f32, f32, f32)> {
    if std::env::var_os("NAIWA_NO_X2").is_some() {
        return None;
    }
    let dot = match dtype {
        GgmlDType::IQ2_S => crate::gguf::iq2s::vec_dot_q8_k_x2,
        GgmlDType::IQ3_XXS => crate::gguf::iq3xxs::vec_dot_q8_k_x2,
        GgmlDType::IQ3_S => crate::gguf::iq3s::vec_dot_q8_k_x2,
        _ => return None,
    };
    let (g0, g1) = dot(g_row, q0, q1).ok()?;
    let (u0, u1) = dot(u_row, q0, q1).ok()?;
    Some((g0, u0, g1, u1))
}

/// The three i-quant gate/up stacks that have a two-row kernel.
#[cfg(target_arch = "x86_64")]
fn has_x2(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::IQ2_S | GgmlDType::IQ3_XXS | GgmlDType::IQ3_S
    )
}

/// The bytes of a `&[T]` of repr(C), unpadded block types.
///
/// SAFETY: every type this is used with is `#[repr(C)]` with no padding (the same assumption the
/// block types' own `as_bytes` helpers make).
#[cfg(target_arch = "x86_64")]
unsafe fn block_bytes<T>(s: &[T]) -> &[u8] {
    std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
}

/// SwiGLU MoE decode on the CPU over the stacked expert weights.
///
/// `gate`, `up`: `[E, inter, hidden]`; `down`: `[E, hidden, inter]`, all on the CPU. `xs` is
/// `rows x hidden` f32; `ids` and `weights` are row-major `[rows, topk]`. Returns `rows x hidden`
/// f32, or `None` when a dtype or shape is not served here (the caller uses the generic path).
pub fn fused_moe_decode(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    xs: &[f32],
    ids: &[u32],
    weights: &[f32],
    topk: usize,
) -> Result<Option<Vec<f32>>> {
    let (Ok((e, inter, hidden)), Ok(ud), Ok((de, dn, dk))) = (
        gate.shape().dims3(),
        up.shape().dims3(),
        down.shape().dims3(),
    ) else {
        return Ok(None);
    };
    if ud != (e, inter, hidden) || (de, dn, dk) != (e, hidden, inter) || gate.dtype() != up.dtype()
    {
        return Ok(None);
    }
    if topk == 0
        || ids.len() != weights.len()
        || !ids.len().is_multiple_of(topk)
        || xs.len() != ids.len() / topk * hidden
    {
        return Ok(None);
    }
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= e) {
        candle_core::bail!("expert id {bad} out of range for {e} experts");
    }
    let rows = ids.len() / topk;

    // Distinct (expert, row) pairs, in expert order so pass 1 walks the stack forward; and each
    // row's slots as (pair, weight), dropping weight-0 padding.
    let mut pairs: Vec<(u32, usize)> = Vec::with_capacity(ids.len());
    for t in 0..rows {
        for j in 0..topk {
            if weights[t * topk + j] != 0.0 {
                pairs.push((ids[t * topk + j], t));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let mut slots: Vec<Vec<(usize, f32)>> = vec![Vec::with_capacity(topk); rows];
    for t in 0..rows {
        for j in 0..topk {
            let w = weights[t * topk + j];
            if w != 0.0 {
                let p = pairs
                    .binary_search(&(ids[t * topk + j], t))
                    .expect("pair present");
                slots[t].push((p, w));
            }
        }
    }
    if pairs.is_empty() {
        return Ok(Some(vec![0f32; rows * hidden]));
    }

    let (g, u, d) = (gate.data()?, up.data()?, down.data()?);
    // Stage timing, read via `last_phases()` (NAIWA_FUSED_PHASES=0 disables): the pair/slot
    // build above, act quant + pass 1 (inside gate_up), h quant + pass 2 (inside down_sum).
    let t = std::time::Instant::now();
    let mut ph = Phases { on: phases_on(), ..Default::default() };
    ph.us_pairs = if ph.on { t.elapsed().as_secs_f64() * 1e6 } else { 0.0 };
    let h = with_block!(
        gate.dtype(),
        G,
        gate_up::<G>(&g, &u, (e, inter, hidden), xs, &pairs, &mut ph)
    )?;
    let Some(h) = h else {
        return Ok(None);
    };
    ph.us_pass1 = ph.elapsed_us(t);
    let t1 = std::time::Instant::now();
    // The Q2_0 and IQ4_NL down stacks take their AVX2 row kernels (W102/W103: the per-call
    // vec_dot shape pays call overhead and serial chains per row, dot_micro); anything else
    // (or the NAIWA_NO_Q2ROWS/NAIWA_NO_I4NLROWS fallbacks) goes through the generic path.
    let down_row_kernel: Option<&str> = match down.dtype() {
        GgmlDType::Q2_0 if q2_rows_ok() => Some("q2"),
        GgmlDType::IQ4NL if i4nl_rows_ok() => Some("i4nl"),
        _ => None,
    };
    let out: Option<Vec<f32>> = match down_row_kernel {
        Some("q2") => {
            down_sum_q2_0(&d, (e, hidden, inter), &h, &pairs, &slots, &mut ph)?
                .or_else(|| {
                    // shape fallback (or NAIWA_NO_Q2ROWS for A/B): the generic per-call path
                    with_block!(
                        down.dtype(),
                        D,
                        down_sum::<D>(&d, (e, hidden, inter), &h, &pairs, &slots, &mut ph)
                    )
                    .ok()
                    .flatten()
                })
        }
        Some("i4nl") => {
            down_sum_i4nl(&d, (e, hidden, inter), &h, &pairs, &slots, &mut ph)?
                .or_else(|| {
                    with_block!(
                        down.dtype(),
                        D,
                        down_sum::<D>(&d, (e, hidden, inter), &h, &pairs, &slots, &mut ph)
                    )
                    .ok()
                    .flatten()
                })
        }
        _ => {
            with_block!(
                down.dtype(),
                D,
                down_sum::<D>(&d, (e, hidden, inter), &h, &pairs, &slots, &mut ph)
            )?
        }
    };
    ph.us_pass2 = ph.elapsed_us(t1);
    // `NAIWA_FUSED_PHASE_DBG=1` (M37 step 1): one line per call, per quant type, so the host
    // cost can be attributed to a kernel family instead of guessed at.
    if std::env::var_os("NAIWA_FUSED_PHASE_DBG").is_some() {
        eprintln!(
            "PHASE_DBG gate={:?} down={:?} rows={rows} experts={} pairs={} us_pairs={:.0} us_act={:.0} us_gateup={:.0} us_hq={:.0} us_down={:.0}",
            gate.dtype(),
            down.dtype(),
            pairs.iter().map(|p| p.0).collect::<std::collections::BTreeSet<_>>().len(),
            pairs.len(),
            ph.us_pairs,
            ph.us_act,
            ph.us_pass1 - ph.us_act,
            ph.us_hq,
            ph.us_pass2 - ph.us_hq,
        );
    }
    LAST.with(|l| *l.borrow_mut() = ph.clone());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::fused_moe_decode;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor, D};
    use std::sync::Arc;

    /// Fused output against the generic CPU path it replaces: three routed gemvs
    /// (`qtensor_indexed_moe_forward`) with silu, the product and the weighted sum in between, all
    /// in f32. Both quantize the activations the same way, so they differ only in summation order.
    /// Covers several rows, an expert shared by two rows, and weight-0 padding.
    #[test]
    fn fused_matches_generic_cpu_path() {
        let (e, inter, hidden, topk, rows) = (6usize, 64usize, 256usize, 3usize, 3usize);
        let mk = |n: usize, k: usize, seed: f32| {
            let v: Vec<f32> = (0..e * n * k)
                .map(|i| ((i as f32) * seed).sin() * 0.3)
                .collect();
            Tensor::from_vec(v, (e, n, k), &Device::Cpu).unwrap()
        };
        for (gt, dt) in [
            (GgmlDType::Q4K, GgmlDType::Q8_0),
            (GgmlDType::Q8_0, GgmlDType::Q4_0),
            (GgmlDType::Q2K, GgmlDType::Q5_0),
        ] {
            let gq = Arc::new(QTensor::quantize(&mk(inter, hidden, 0.37), gt).unwrap());
            let uq = Arc::new(QTensor::quantize(&mk(inter, hidden, 0.21), gt).unwrap());
            let dq = Arc::new(QTensor::quantize(&mk(hidden, inter, 0.13), dt).unwrap());
            let xs: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i as f32) * 0.011).cos())
                .collect();
            let ids = vec![1u32, 4, 2, 4, 0, 1, 5, 5, 3];
            let ws = vec![0.5f32, 0.3, 0.2, 0.6, 0.4, 0.0, 0.7, 0.0, 0.3];
            let got = fused_moe_decode(&gq, &uq, &dq, &xs, &ids, &ws, topk)
                .unwrap()
                .unwrap();

            let fwd = super::super::cpu::qtensor_indexed_moe_forward;
            let x = Tensor::from_vec(xs.clone(), (rows, 1, hidden), &Device::Cpu).unwrap();
            let idt = Tensor::from_vec(ids.clone(), (rows, topk), &Device::Cpu).unwrap();
            let g = fwd(&gq, &x, &idt).unwrap();
            let u = fwd(&uq, &x, &idt).unwrap();
            let h = ((&g / (g.neg().unwrap().exp().unwrap() + 1.0).unwrap()).unwrap() * u).unwrap();
            let y = fwd(&dq, &h, &idt).unwrap(); // [rows, topk, hidden]
            let w = Tensor::from_vec(ws.clone(), (rows, topk, 1), &Device::Cpu).unwrap();
            let want = y
                .broadcast_mul(&w)
                .unwrap()
                .sum(D::Minus2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                err <= 1e-3 * scale,
                "{gt:?}/{dt:?}: max err {err} vs scale {scale}"
            );
        }
    }

    /// The Q2_0 down row kernel (`down_sum_q2_0`, W102) against the generic per-call path over
    /// the same tensors: candle cannot build Q2_0 storage for the generic chain's QTensor ops
    /// (its `unreachable` in quantized/mod.rs), so the reference here is the fused path itself
    /// with the row kernel disabled. The two differ only in summation order (the correction is
    /// exact in the integers), so the fused test's 1e-3 relative tolerance applies.
    #[test]
    fn q2_0_row_kernel_matches_per_call_path() {
        if !cfg!(target_arch = "x86_64")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let (e, inter, hidden, topk, rows) = (6usize, 128usize, 256usize, 3usize, 3usize);
        let mk = |n: usize, k: usize, seed: f32| {
            let v: Vec<f32> = (0..e * n * k)
                .map(|i| ((i as f32) * seed).sin() * 0.3)
                .collect();
            Tensor::from_vec(v, (e, n, k), &Device::Cpu).unwrap()
        };
        let gq = Arc::new(
            QTensor::quantize(&mk(inter, hidden, 0.37), GgmlDType::Q4K).unwrap(),
        );
        let uq = Arc::new(
            QTensor::quantize(&mk(inter, hidden, 0.21), GgmlDType::Q4K).unwrap(),
        );
        // candle cannot allocate Q2_0 storage (its from_data is a stub); build the same
        // bytes QTensor::quantize would through the fork's own quantizer and builder.
        let dwr = mk(hidden, inter, 0.13)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut dq_bytes = vec![0u8; dwr.len() / 64 * 18];
        crate::gguf::q2_0::quantize_q2_0(&dwr, &mut dq_bytes).unwrap();
        let dq = Arc::new(
            QTensor::new(
                crate::gguf::q2_0::cpu_storage(&dq_bytes).unwrap(),
                (e, hidden, inter),
            )
            .unwrap(),
        );
        let xs: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.011).cos())
            .collect();
        let ids = vec![1u32, 4, 2, 4, 0, 1, 5, 5, 3];
        let ws = vec![0.5f32, 0.3, 0.2, 0.6, 0.4, 0.0, 0.7, 0.0, 0.3];
        let kernel = fused_moe_decode(&gq, &uq, &dq, &xs, &ids, &ws, topk)
            .unwrap()
            .unwrap();
        // SAFETY: the test thread is the only reader of NAIWA_NO_Q2ROWS; the OnceLock it
        // seeds is raced into exactly one of the two states, both valid.
        unsafe { std::env::set_var("NAIWA_NO_Q2ROWS", "1") };
        let fallback = fused_moe_decode(&gq, &uq, &dq, &xs, &ids, &ws, topk)
            .unwrap()
            .unwrap();
        unsafe { std::env::remove_var("NAIWA_NO_Q2ROWS") };
        let scale = fallback.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = kernel
            .iter()
            .zip(&fallback)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 1e-3 * scale,
            "Q2_0 row kernel: max err {err} vs scale {scale}"
        );
    }

    /// The IQ4_NL down row kernel (`down_sum_i4nl`, W103) against the generic per-call path
    /// over the same tensors; the same shape as the Q2_0 test above. IQ4NL storage builds
    /// through the fork's own quantizer and builder (candle's `QTensor::quantize` does not
    /// know the type).
    #[test]
    fn i4nl_row_kernel_matches_per_call_path() {
        if !cfg!(target_arch = "x86_64")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let (e, inter, hidden, topk, rows) = (6usize, 128usize, 256usize, 3usize, 3usize);
        let mk = |n: usize, k: usize, seed: f32| {
            let v: Vec<f32> = (0..e * n * k)
                .map(|i| ((i as f32) * seed).sin() * 0.3)
                .collect();
            Tensor::from_vec(v, (e, n, k), &Device::Cpu).unwrap()
        };
        let gq = Arc::new(
            QTensor::quantize(&mk(inter, hidden, 0.37), GgmlDType::Q4K).unwrap(),
        );
        let uq = Arc::new(
            QTensor::quantize(&mk(inter, hidden, 0.21), GgmlDType::Q4K).unwrap(),
        );
        let dwr = mk(hidden, inter, 0.13)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut dq_bytes = vec![0u8; dwr.len() / 32 * 18];
        crate::gguf::iq4nl::quantize_iq4_nl(&dwr, &mut dq_bytes).unwrap();
        let dq = Arc::new(
            QTensor::new(
                crate::gguf::iq4nl::cpu_storage(&dq_bytes).unwrap(),
                (e, hidden, inter),
            )
            .unwrap(),
        );
        let xs: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.011).cos())
            .collect();
        let ids = vec![1u32, 4, 2, 4, 0, 1, 5, 5, 3];
        let ws = vec![0.5f32, 0.3, 0.2, 0.6, 0.4, 0.0, 0.7, 0.0, 0.3];
        let kernel = fused_moe_decode(&gq, &uq, &dq, &xs, &ids, &ws, topk)
            .unwrap()
            .unwrap();
        // SAFETY: the test thread is the only reader of NAIWA_NO_I4NLROWS; the OnceLock it
        // seeds is raced into exactly one of the two states, both valid.
        unsafe { std::env::set_var("NAIWA_NO_I4NLROWS", "1") };
        let fallback = fused_moe_decode(&gq, &uq, &dq, &xs, &ids, &ws, topk)
            .unwrap()
            .unwrap();
        unsafe { std::env::remove_var("NAIWA_NO_I4NLROWS") };
        let scale = fallback.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = kernel
            .iter()
            .zip(&fallback)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 1e-3 * scale,
            "IQ4_NL row kernel: max err {err} vs scale {scale}"
        );
    }

    /// Decode-shaped timing, fused against the generic chain: `MISSES` random experts per row out of
    /// a 128-expert pool (so the weights come from DRAM), IQ2_S gate/up (640x2560) and IQ4_NL down
    /// (2560x640) as in most of this model's layers, called from a thread outside the rayon pool.
    #[test]
    #[ignore = "throughput microbenchmark"]
    fn bench_fused_vs_generic() {
        use crate::gguf::{iq2s::BlockIQ2S, iq4nl::BlockIQ4NL};
        use candle_core::quantized::GgmlType;
        let misses: usize = std::env::var("MISSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let calls: usize = std::env::var("CALLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        // Idle time between calls, as between the layers of a decode step (rayon's workers sleep).
        let gap = std::time::Duration::from_micros(
            std::env::var("GAP_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        );
        let (e, inter, hidden) = (128usize, 640usize, 2560usize);
        let raw = |n: usize, k: usize, bs: usize, blk: usize, seed: u32| {
            let mut v = vec![0u8; e * n * (k / blk) * bs];
            let mut s = seed;
            for b in v.iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
            v
        };
        let q = |dt: GgmlDType, n, k, bs, blk, seed| {
            let bytes = raw(n, k, bs, blk, seed);
            let storage = match dt {
                GgmlDType::IQ2_S => crate::gguf::iq2s::cpu_storage(&bytes),
                _ => crate::gguf::iq4nl::cpu_storage(&bytes),
            }
            .unwrap();
            Arc::new(QTensor::new(storage, vec![e, n, k]).unwrap())
        };
        let gsz = std::mem::size_of::<BlockIQ2S>();
        let dsz = std::mem::size_of::<BlockIQ4NL>();
        let gq = q(
            GgmlDType::IQ2_S,
            inter,
            hidden,
            gsz,
            BlockIQ2S::BLCK_SIZE,
            1,
        );
        let uq = q(
            GgmlDType::IQ2_S,
            inter,
            hidden,
            gsz,
            BlockIQ2S::BLCK_SIZE,
            2,
        );
        let dq = q(
            GgmlDType::IQ4NL,
            hidden,
            inter,
            dsz,
            BlockIQ4NL::BLCK_SIZE,
            3,
        );
        for rows in [1usize, 3, 5] {
            let topk = misses;
            let xs: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i as f32) * 0.011).cos())
                .collect();
            let ws = vec![0.3f32; rows * topk];
            let (gq, uq, dq, xs2, ws2) =
                (gq.clone(), uq.clone(), dq.clone(), xs.clone(), ws.clone());
            let r = std::thread::spawn(move || {
                let mut s = 9u32;
                let mut pick = || {
                    (0..rows * topk)
                        .map(|_| {
                            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                            (s >> 8) % e as u32
                        })
                        .collect::<Vec<u32>>()
                };
                let mut busy = std::time::Duration::ZERO;
                for _ in 0..calls {
                    let ids = pick();
                    if !gap.is_zero() {
                        std::thread::sleep(gap);
                    }
                    let t = std::time::Instant::now();
                    let _ = fused_moe_decode(&gq, &uq, &dq, &xs2, &ids, &ws2, topk)
                        .unwrap()
                        .unwrap();
                    busy += t.elapsed();
                }
                let fused = busy.as_secs_f64() / calls as f64;
                let fwd = super::super::cpu::qtensor_indexed_moe_forward;
                let x = Tensor::from_vec(xs2.clone(), (rows, 1, hidden), &Device::Cpu)
                    .unwrap()
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap();
                let w = Tensor::from_vec(ws2.clone(), (rows, topk, 1), &Device::Cpu).unwrap();
                let t = std::time::Instant::now();
                for _ in 0..calls {
                    let ids = Tensor::from_vec(pick(), (rows, topk), &Device::Cpu).unwrap();
                    let g = fwd(&gq, &x, &ids).unwrap();
                    let u = fwd(&uq, &x, &ids).unwrap();
                    let h = (candle_nn::ops::silu(&g).unwrap() * u).unwrap();
                    let y = fwd(&dq, &h, &ids).unwrap();
                    let _ = y
                        .to_dtype(candle_core::DType::F32)
                        .unwrap()
                        .broadcast_mul(&w)
                        .unwrap()
                        .sum(D::Minus2)
                        .unwrap()
                        .to_dtype(candle_core::DType::BF16)
                        .unwrap();
                }
                (fused, t.elapsed().as_secs_f64() / calls as f64)
            })
            .join()
            .unwrap();
            println!(
                "rows {rows} x {misses} misses: generic {:7.1} us/layer, fused {:7.1} us/layer ({:.2}x)",
                r.1 * 1e6, r.0 * 1e6, r.1 / r.0
            );
        }
    }
}
