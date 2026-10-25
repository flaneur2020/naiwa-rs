# M21 — the continued turn: why every non-first turn runs its suffix on the CPU, and what it costs

Status: **open** (planned 2026-10-24).
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
