# M21 — the continued turn: why every non-first turn runs its suffix on the CPU, and what it costs

Status: **closed as negative** (2026-10-24): three mechanisms refuted (§8, §9, §10), one law established, and a
cache-sizing defect handed off (§10). The kernel direction is M22.
Parent: [m20-expert-cache-ceiling.md](m20-expert-cache-ceiling.md) — §27, §30, §31.
One mechanism this milestone: **the small-suffix prefill path** — a continued turn's new rows go to the GPU instead
of the CPU expert path, by choosing the threshold from the upload arithmetic rather than leaving it above every
suffix we actually serve. No kernel is written in M21; if the arithmetic says a kernel is needed, that is M22.

---

## 1. Why this milestone exists

M20's side-by-side with strata (3 rounds, KV aligned, §30/§31) leaves one gap that dwarfs the others in the chat
workload, and it is not the one the previous four milestones were about:

| class | metric | strata | naiwa | delta |
|---|---|---:|---:|---:|
| doc | TTFT >= 2 p50 | 672 ms | 1 193 ms | **+78%** |
| doc | TTFT >= 2 p90 | 762 ms | 1 714 ms | **+125%** |
| short | TTFT >= 2 p50 | 655 ms | 1 117 ms | **+71%** |
| short | TTFT >= 2 p90 | 804 ms | 2 201 ms | **+174%** |
| code | TTFT >= 2 p50 | 1 201 ms | 1 276 ms | +6% |

Decode is +10..23% (§30) and turn-1 prefill is +58..65% once per conversation (§27); **the continued turn is +71..
78% at the median and up to +174% at p90, on every turn**. Fitting turn >= 2 TTFT against the number of new rows
(`prompt * (1 - reuse)`, 54 points per engine, g19) says the shape of it:

| | fixed | marginal |
|---|---:|---:|
| strata | 623 ms | 3.09 ms/row |
| naiwa | ~0 ms | **21-28 ms/row** |

**Our slope is 7-8x theirs and our intercept is better than theirs.** We are not paying for a checkpoint or a
drafter replay; we are paying 21 ms for each new row.

## 2. The cause, as far as it is known

`--prefill-stream-min-tokens` (default 1024, our config 64) sends a chunk **below** the threshold to the CPU expert
path. New-row counts of continued turns in g19:

| new rows | requests |
|---|---:|
| < 32 | 5 / 54 |
| **< 64** | **45 / 54** |
| < 128 | 54 / 54 |
| < 1024 | 54 / 54 |

So **54/54 continued turns took the CPU path** — every non-first turn of every conversation. Strata runs the same
rows on the GPU.

The arithmetic that decides a threshold, for an `r`-row chunk where the prompt routes `topk = 10` experts per row:

| | CPU path | GPU path (streamed) |
|---|---|---|
| cost | `r x 21 ms` (measured) | one upload of every routed expert: `min(r x topk, experts/layer) x 48 layers x 1.88 MB` bytes, at 13.4 GB/s (M17 calibration) |
| 41 rows | 860 ms | 410 experts x 48... see note: per layer `r x topk` is 410 across the whole model only at `r x topk <= 256` | **~57 ms of DMA plus kernel time** |

The break-even is therefore not a guess: it is `r x 21 ms` against `rows->experts x 1.88 MB / 13.4 GB/s`, and at
`r = 41` the streamed path is more than 10x cheaper before any kernel efficiency is assumed. The threshold in the
shipped config is above every suffix we serve, which is the defect.

## 3. What M21 must establish

| # | question | measurement |
|---|---|---|
| T0 | does lowering `--prefill-stream-min-tokens` below the typical suffix make continued turns faster? | sweep `{64, 32, 16, 8}`, doc probe, turn >= 2 TTFT p50/p90 |
| T1 | at what suffix length does streaming stop paying, i.e. where is the real break-even? | same sweep, grouped by new rows (<16, 16-32, 32-64, 64-128) |
| T2 | does the streamed path need more VRAM at small chunks (`plan()`'s stride, the ring) and does it survive? | `log_plan`, OOM arms, `NAIWA_DECODE_RESERVE_MIB` |
| T3 | does it cost anything on turn 1 (where the chunk is large and already streamed) or on TPOT? | turn-1 TTFT, TPOT per class |

## 4. Gate

- **doc TTFT >= 2 p50 <= 700 ms, p90 <= 800 ms** (strata: 672 / 762; we are 1 193 / 1 714).
- turn-1 TTFT unchanged within noise (3 760 ms on doc).
- TPOT per class not worse than §30's medians (code 35.7 / doc 44.4 / long 44.1 / short 50.5).
- chat harness 2 rounds, 0 failures; oracle arm untouched (`frac = 0`, `--no-lookahead`).
- Per the repo rule: median of >= 3 runs, no retried rounds.

## 5. Plan

1. **Sweep the flag** (`64 -> 32 -> 16 -> 8`), doc probe, 3 runs each, one server at a time. Record TTFT turn 1 /
   2 / 3, TPOT, and `new rows`, so T1 falls out of the same data.
2. **Instrument the decline reason.** `prefill_stream::forward` already has a `dbg_note` for "declined, no host
   stacks"; add one for "below threshold" with the row count, so a run states which path every chunk took instead of
   inferring it from the timing.
3. **Check the memory plan at the winning threshold**: `reserve()` returning stride 0 silently falls back to the CPU
   (`Ok(None)`), which would look like "the flag did nothing". `log_plan` + one arm at `1.85 GB` cache.
4. **Then decide.** If the sweep reaches the gate, ship the threshold (and the reason in the arg's doc comment). If
   the DMA arithmetic does not hold up - i.e. small chunks are dominated by something other than the expert path -
   the next milestone is M22 and the kernel (grouped GEMM/MMQ reading the byte ring), which is what would also
   remove the 650 MiB prefill workspace.

## 6. Non-goals

- No change to the cache, the profile seeding, or promotion.
- No change to `MTP` or the prefix-reuse machinery: the suffix is what it is, this is about where its experts run.
- No new CUDA code. If the sweep shows the streamed path is itself the problem (e.g. per-chunk setup ~ms), that is a
  finding, not something to patch here.

## 7. Open questions

- The measured 21 ms/row is the *CPU* path at 24-101 rows. Is it rayon throughput (experts on CPU cores), or the
  per-layer host round-trip that M20 §30 priced at ~125 us x 48? The sweep answers it indirectly: if the streamed
  path removes ~all of the slope, it was the expert compute; if a ~600 ms residual stays, it was the round-trip.
- Does the streamed path already work at 8 rows, or does its per-chunk setup (ring reserve, per-layer fences) put a
  floor under it? T1 and T3 answer this.
- `DEFAULT_MIN_TOKENS = 1024` exists for a reason (a streamed chunk uploads every routed expert whatever its length).
  That reasoning was correct for *long* chunks and is wrong for the suffixes we serve; the sweep should confirm the
  crossover empirically rather than by argument.

## 8. First measurements: the hypothesis is refuted, and the real model is the chunk count (2026-10-24)

### T0: the threshold is not the lever

`--prefill-stream-min-tokens` swept over `{64, 32, 16, 8, 1}`, doc probe, 3 runs each, one server per arm. Turn-2
TTFT with 41-73 new rows:

| min_tokens | turn-2 TTFT (51 new rows) |
|---:|---:|
| 64 (shipped) | 1 178 / 1 194 / 1 233 ms |
| 32 | 1 204 / 1 242 / 1 253 |
| 16 | 1 203 / 1 242 / 2 509 |
| 8 | 1 226 / 1 233 / ... |
| 1 | 1 253 / ... |

**Identical within noise.** So the CPU/GPU choice for these chunks changes nothing, and §2's arithmetic - which
predicted a 10x win - is wrong somewhere. It is wrong in its premise about what the chunk *is*.

### What the chunk actually is

`NAIWA_STREAM_DBG=1` now prints the shape every MoE forward is handed (`src/moe/cached_layer.rs`). For a continued
turn with 51 new rows, layer 0 sees:

```
seq 32      <- the suffix is chopped into 32-row chunks
seq 19
seq 5 x N   <- the decode steps (MTP window)
```

32 is the **paged-attention block size**, i.e. the suffix is prefilled to each block boundary because a prefix
checkpoint is written there. A 73-row suffix therefore becomes `32 + 32 + 9` = **three chunks**.

### The model that fits everything

Chunk cost ≈ **620 ms + 1.5 ms/row**, so a continued turn costs `chunks x 620 ms`:

| new rows | chunks | predicted | measured |
|---|---:|---:|---:|
| 41-52 | 2 (32+9..20) | 1.24 s | 768-1 265 ms |
| 73 | 3 (32+32+9) | 1.86 s | 2 282-2 300 ms |
| 105 | 4 | 2.48 s | 2 323-2 537 ms |
| 1 725 (turn 1) | 1 x 8192-cap | 3.85 s | 3 715-4 053 ms |

**The p90 is not jitter: it is one extra chunk.** doc TTFT >= 2 p90 1 714 ms against p50 1 193 ms is 3 chunks against 2.

And it is not reuse-specific: a **fresh 52-token prompt costs 1 283 ms**, the same as a 51-row suffix (1 204 ms).
The reused prefix is free; what we pay for is prefilling the new rows, in 32-row pieces.

Where the 620 ms comes from, per chunk: `32 rows x topk 10 x 48 layers = 15 360` (row, expert) pairs on the CPU path
at ~40 us each. Rows in a 32-row chunk barely share experts (320 picks among 256), so nothing amortises - while the
1 725-row turn-1 chunk hits every expert ~66 times and runs at 2.35 ms/row. **The cost is the absence of amortisation,
and the 32-row chunking is what forbids it.**

### What M21 now is

Not the threshold (`--prefill-stream-min-tokens` stays at 64; the sweep is recorded above so nobody re-runs it). The
mechanism is **the chunk split of a small suffix**:

- if the suffix is small (say <= 128 rows), prefill it as **one chunk**, and write the checkpoint at its end instead
  of at every block boundary inside it;
- expected: 51 rows `1.20 s -> ~0.70 s`, 73 rows `2.29 s -> ~0.75 s`, and the p90 collapses onto the p50 because
  the extra chunks are gone;
- gate unchanged: doc TTFT >= 2 p50 <= 700 ms and p90 <= 800 ms, turn-1 and TPOT untouched.

Open: whether the checkpoint at the end of a merged chunk still satisfies the recurrent-prefix contract
(`MISTRALRS_ALIGN_CHECKPOINTS` exists precisely because the checkpoint must land on a block boundary) - so the first
step of the mechanism is to check whether one block-aligned checkpoint at the *end* of a merged chunk is enough, or
whether the intermediate ones are load-bearing for the *next* turn. If they are, the mechanism becomes "checkpoint at
the end, and if the next turn's reuse then starts earlier, re-verify": a different trade, measured the same way.

## 9. Second measurement: merging the suffix is worse, and it says why (2026-10-24)

`MISTRALRS_MERGE_SUFFIX_BELOW=N` (fork, `pipeline/prompt_chunks.rs`): when a prompt's suffix is `N` rows or fewer,
`build_prompt_chunk_plan` drops `block_align`, so the suffix is prefilled as one chunk instead of being cut at every
paged-attention block boundary. Doc probe, three runs per arm, one server at a time.

| arm | turn-2 TTFT (51 new rows, 2 aligned chunks) | turn-3 | turn-4 |
|---|---|---|---|
| off (block-aligned) | 1 194 / 1 227 / 2 525 ms | 1 090 / 1 169 / 2 277 | 628 / 688 / 1 135 |
| merge <= 64 (1 chunk) | **2 504 / 2 505 / 4 828** | 2 276 / 2 675 / 3 791 | 439 / 3 639 / 4 330 |
| merge <= 128 | **2 087 / 2 102 / 4 753** | 2 090 / 3 750 / 5 744 | 350 / 753 / 3 536 |

**Merging costs 2-4x.** A 51-row suffix as one chunk takes 2.5-4.8 s where two 32-row chunks take 1.2 s, so the block
boundary is not merely convenient for checkpoints - **the prefill path is fast only when a chunk ends on a block
boundary**. The 32-row cut is load-bearing, and the "one bigger chunk amortises the fixed cost" argument (which the
§8 chunk model invites) is wrong: the fixed cost is not per-chunk overhead that a bigger chunk amortises, it is
proportional to the *distinct experts the chunk routes*, and an unaligned chunk takes a slower path on top.

(One `off` arm OOM'd at startup in the first sweep - leftover rows of a previous server still resident, since
`memory.used` was 0 only after waiting - and was re-run clean; the numbers above are the clean arm.)

### What this leaves

Both mechanisms M21 set out to try are now refuted by measurement:

| mechanism | prediction | result |
|---|---|---|
| lower `--prefill-stream-min-tokens` (CPU -> GPU for small chunks) | 10x | **identical** at 64/32/16/8/1 (§8) |
| merge a small suffix into one chunk | chunks x 620 ms -> 1 x 620 | **2-4x worse** (§9) |

And the two numbers that bound it: a 32-row aligned chunk costs ~600 ms whichever path computes it (CPU at
`min_tokens=64`, streamed at `min_tokens=16`), while a 1 725-row chunk costs 2.35 ms/row. The difference is
**amortisation of the distinct-expert set**: 32 rows route ~35% of a layer's 256 experts and can pay for that set
once; 1 725 rows route all of them and pay for the same set once as well, so cost per row collapses.

So the remaining lever for a continued turn is not the chunking and not the path choice; it is **the cost of one
pass over the distinct-expert set of a small chunk** - the CPU expert kernel for a 32-row batch, or the streamed
union of ~90 experts per layer. That is a kernel milestone (M22), and M21 closes with two negative results and one
model:

```
continued-turn TTFT ≈ (number of 32-row aligned chunks) x 600 ms
                     = (new_rows / 32 + 1) x 600 ms
```

which predicts strata's shape as well: their 623 ms fixed + 3 ms/row means they pay it **once per turn, not per
32 rows** - they chunk the suffix by something other than a 32-row block, and their per-row cost is 100x smaller.
That is the thing M22 has to reproduce, and it is a kernel fact, not a scheduling one.

## 10. Third mechanism refuted, and a defect the cache sweep walked into (2026-10-24)

### `NAIWA_PREFILL_THRESHOLD` (the decode-split cut, default 32)

If the constant is 32 because a chunk of a few rows that goes the whole-chunk way costs ~600 ms, raising it so a
whole small suffix takes the decode split should be the same win for a continued turn. Swept `{32, 64, 128, 256}`,
doc probe, 3 runs each:

| threshold | turn-2 TTFT (51 new rows) | turn-3 |
|---|---|---|
| 32 (shipped) | 1 233 / 1 239 / 1 254 ms | 1 074 / 1 110 / 2 351 |
| 64 | 1 160 / 1 167 / 1 185 | 1 065 / 1 078 / 2 335 |
| 128 | 1 207 / 1 225 / 1 232 | 1 126 / 2 069 / 2 937 |
| 256 | 1 259 / 1 273 / 1 273 | 1 125 / 2 123 / 1 150 |

**Identical.** So the continued turn does not care whether its suffix is cut at 32 rows, merged, computed on the CPU,
or streamed: **all four knobs land on ~1.2 s.** Three mechanisms, one answer, so the cost is not in any of them.

The number that explains it is M17's: *a prefill chunk reads ~23 GiB of expert data*, because even 32 rows route
nearly all 256 experts of all 48 layers. Nothing about the chunk's shape changes the size of that pass; chunking,
merging and path selection only change how many times it is paid - and the merge arm paid it in a slow way, the
others pay it the same way.

### The cache sweep found a defect instead

While testing whether residency matters (it would not, per the above), two cache sizes were run:

| cache | turn-1 TTFT | turn-2 | turn-3 |
|---|---:|---:|---:|
| 1.4 GB | **2 858 / 2 953 / 2 989** | 1 397 / 2 408 / 2 417 | 1 241 / 1 242 / 2 135 |
| 2.3 GB (shipped) | 3 597 / 3 769 / 3 773 | 1 233 / 1 239 / 1 254 | 1 074 / 1 110 / 2 351 |
| **3.0 GB** | **154 008 / 154 859** | 991 / 1 147 | 6 077 / 8 977 |

Two things fall out:

1. **A smaller cache makes turn-1 prefill faster** (2 858 ms at 1.4 GB against 3 773 at 2.3 GB, -24%): the prefill
   stream's buffers and the cache compete, and at 1.4 GB the plan gives the stream more room.
2. **At 3.0 GB turn-1 prefill degrades by 41x** (154 s for 1 725 rows, i.e. 89 ms/row - the whole chunk falling
   back to the CPU path with nothing resident). This is a **defect, not a trade**: M19/M20 read the 2.6-3.0 GB arms
   as OOM; here is a 3.0 GB boot that runs, and its prefill is unusable. Whatever the plan decides when the cache is
   large, it must refuse to launch the stream rather than silently lose 150 s - that is §4.4's "failure shrinks,
   never silently" applied to the prefill plan.

M21 therefore closes with one model and two negative results, and hands off one defect:

```
continued-turn TTFT ≈ 1.2 s  (independent of chunking, path, and decode-split threshold)
                    = one pass over the chunk's distinct-expert set
                      (M17: ~23 GiB per chunk, because 32 rows already route ~all 256 experts x 48 layers)
```

Making that pass cheap is a **kernel** question (a grouped GEMM that reads only the resident subset on the GPU and the
remainder on the CPU, without materialising the full set) - M22 - and strata's 3 ms/row is that kernel, not a
scheduling trick: their fixed 623 ms + 3 ms/row means one set pass per *turn*, not per chunk, and 100x cheaper per
row.

## 11. The 3.0 GB defect, named: the plan measures free VRAM before the rings exist

The 3.0 GB arm's log says it in two lines:

```
prefill (warmup): chunk 7424, 1 x 650 MiB layer buffer(s) + 504 MiB activations + 512 MiB headroom
                  = 1666 MiB of 1730 MiB free                      <- the plan, at warmup: fits
WARN prefill stream: off (one 650 MiB layer buffer + 504 MiB activations + 512 MiB headroom
                  does not fit in 1346 MiB free)                   <- the first chunk, 384 MiB later
```

`PrefillStreams::plan()` runs at warmup and decides `stride 1, chunk 7424` from 1 730 MiB free. Between that
measurement and the first chunk, the staging rings are built (~215 MiB across the seven quant shapes) and the KV
grows; `reserve()` re-measures, finds 1 346 MiB, and `plan()` returns `None` - so `Plan { stride: 0, chunk: 0 }` is
published and **the stream is off for the whole session**. Every prefill chunk then takes the CPU path: **154 s for a
1 725-row prompt, 89 ms/row, 41x slower** - the code's own warning says 33x.

The plan is not wrong to refuse: at 1 346 MiB free, one 650 MiB layer buffer plus the 512 MiB headroom and the
160 MiB activation floor cannot coexist. Three ways out, smallest first:

| | what | cost | what it buys |
|---|---|---|---|
| **(a1)** | take the measurement **once**, after everything that allocates has allocated (rings, doorbells, KV), so the number is not stale; and when the stream does not fit, **clamp `--expert-cache-gb` down until it does**, logging the clamp, instead of launching a cache that starves the stream | hours | 3.0 GB becomes "cache clamped to ~2.4 GB, stream intact, turn-1 3.7 s" instead of "3.0 GB, 154 s". Also makes the 1.4 GB arm's -24% turn-1 explicable in one line rather than a coincidence |
| **(a2)** | make the stream's unit **sub-layer**: ~90 experts (170 MiB) instead of a full 650 MiB layer stack, so it fits beside almost any cache | days, and it is the grouped-GEMM kernel | a 3.0 GB cache *with* a fast prefill - M20's original objective, which the old ledger called impossible |
| (a3) | default `NAIWA_PREFILL_STRICT` on | minutes | a loud refusal instead of a 41x silent regression - correct by §4.4, but it refuses to serve rather than serving well |

(a1) is the one to do first: it is small, it removes a 41x silent regression from the shipping configuration space,
and it is the precondition for measuring (a2) fairly.

## 12. (a1) shipped: the plan now reserves the KV's growth, and the cache ceiling moved to 2.6 GB (2026-10-24)

Three changes, all small:

1. **The plan reserves the KV's growth.** `plan_with_reserve(free, layer, ceiling, borrowing, kv_reserve)`: the KV
   pool grows to the first prompt's length *after* warmup measured it. 384 MiB is what this card lost between warmup
   and the first chunk at `--pa-context-len 4096` (measured, §11), and the term is the `NAIWA_KV_RESERVE_MIB` knob.
   A unit test pins it: at 1 730 MiB free the plan refuses with the reserve and accepts without it, and the shipping
   2.3 GB configuration (2 498 MiB free) still takes the full 8 192 chunk.
2. **The refusal is loud and gives the number.** It bisects the free VRAM for the smallest cache reduction at which
   a chunk fits and says: *"drop --expert-cache-gb by at least 0.62 GB"*.
3. **It stays refused.** `PrefillStream::refused` is sticky: a stream that cannot fit no longer publishes
   `stride 0` and quietly serves at 41x. `NAIWA_PREFILL_ALLOW_CPU=1` is the opt-in for the slow path (measured: 152
   s, exactly the old behaviour), and without it the request fails with the remedy in the message.

### The cache ceiling moves: 2.3 -> 2.6 GB

| `--expert-cache-gb` | plan | behavior |
|---|---|---|
| 2.3 (was shipping) | chunk 8 192, `1684 MiB of 2498 MiB free` | unchanged: 0 failures, turn-1 3 637-3 805 ms |
| **2.6** | chunk 8 192, `1684 MiB of 2178 MiB free` | **works**: turn-1 3 679-3 779 ms, TPOT turn-2 41.8 ms - +300 MiB of cache for free |
| 2.8 | - | CUDA OOM **at startup** (the cache itself does not fit; an earlier 2.8 boot got through and served 154 s, so this size was always luck) |
| 3.0 | warns, then every request fails | *"prefill stream refused ... drop --expert-cache-gb by at least 0.62 GB"* |

So M20's premise - "2 355 MiB is our ceiling and 2.6-3.0 GB OOMs" - was wrong in an instructive way: 2.6 GB never
OOM'd, it lost the *plan* and then served at 41x, which the earlier milestones read as a failed arm. The ceiling the
ledger should have reported is **2.6 GB with a working prefill**, and the 384 MiB that decided it was the KV's own
growth, invisible to a plan that measured once at warmup.

Gate status: the prefill plan defect of §10 is fixed; the *continued turn* has not moved (that is M22's kernel), and
the 2.6 GB arm is being run through the chat harness for the correctness gate before it is considered shippable.
