# M16 — chat with MTP: prefix reuse under speculation

Status: **in progress** (2026-10-21). One mechanism this milestone: **a speculative decode that does not
cost cross-turn prefix reuse**, so the per-layer host round trip can be amortised over a multi-row step.
Parent: [m10-chat.md](m10-chat.md).

## Why this, now

M10 left one measured term standing: chat spends **~36 ms/token of host↔device round trip**
(`dry 1.2 ms/layer × 48`, `busy 0.000` — the GPU is idle at every layer entry) on top of a 19 ms GPU chain.
Two things were closed on the way: the CPU miss path is *not* on that path (`--pcie-frac 1.0` removes it with
no TPOT change), and the resident weights are already at the file's cost (4,300 MiB, 1,299 MiB of it BF16 in
the file). So the only lever left is **rows per step**: one round trip per layer per *step*, not per token.

MTP is that lever, and the measurements say it works *when acceptance is high* — but it also sets
`SpeculativePrefixReplay::Full`, and M10 measured prefix reuse at **0.00% with MTP vs 33.33% without** on the
same conversation. On `doc`/`long` that costs 20–30 s of TTFT per turn, which is a bigger term than the decode
gain. Hence this milestone.

## What is established (measured, M10.3 items 3b and 6)

| fact | number |
|---|---|
| TPOT at 25–30% acceptance (`--mtp-n-predict 4`) | 59–62 ms, no better than no-MTP 55 ms |
| TPOT at 41–80% acceptance (same arm, later turns) | 48 → **35.8 ms**, under Strata's 38 |
| expert hit rate with 5 rows | 78–81% (vs 51% at one row) |
| host cost per layer | `dry = 1.00 + ~0.25 × rows` ms (1 row 1.21, 5 rows 2.27) |
| `--pcie-frac 1.0` with MTP | worse: 2 668 MiB staged per window vs 848 → PCIe saturation |

So MTP's gain is `48 × dry / accept_len`, i.e. it is fighting a host term that grows with rows.

## Root cause of the 0% reuse (traced)

`MISTRALRS_PREFIX_DBG=1` (fork `bea0e450`) shows the constant kills **capture as well as restore**, which
was not visible from the call site:

```
replay_for = Full
  → clamp_speculative_prefix_cache_hit(cached, 32, Full) = 0
  → recurrent_checkpoint_boundary(...) = None
  → build_prompt_chunk_plan(..., replay, ...) receives no boundary → the prompt stays one chunk
  → snapshot_paged_recurrent_prefix is never called (not one `snapshot` line)
  → every restore prints `no_replay_boundary` and stages a recurrent reset
```

| arm | snapshots | restore | prefix hit rate |
|---|---|---|---|
| no MTP | seq 2 @32, seq 3 @96 | `no_replay_boundary cached=0` | **33.33%** |
| MTP, `Full` | none | `cached=0/32/96` | 0.00% |
| MTP, `MISTRALRS_MTP_PREFIX_REPLAY=none` | seq 2 @32, seq 3 @96 | `cached=0` | **33.33%** |

The override (`full|none|suffix:<n>`, default `Full`) is the fork's `builtin_mtp_prefix_replay()`;
`qwen4_exp` is the wrapper this model uses, not `qwen3_5`.

**Why `Full` is at least partly over-conservative.** The documented hazard is that a block hash covers target
tokens only, while a block written during a speculative step can hold draft rows in its tail that later become
interior. But capture is prompt-only (`is_prompt && !scheduler_visible_prompt_step`) and the boundary is
strictly inside the prompt, where every row is a committed target token; the drafts live in the drafter's own
paged cache (`drafter_prefill_chunk`, placeholder row `0`). The hazard as stated applies to a capture path this
build does not take.

## Two determinism sources (found, must be controlled before any gate means anything)

| source | effect | control |
|---|---|---|
| CPU vs GPU expert path | which runs depends on the **lookahead's timing**, and the two reduce in different orders → the same greedy prompt differs run to run (90 vs 102 tokens), even with `NAIWA_CPU_MOE_THREADS=1` | `--no-lookahead --pcie-frac 1.0` |
| prompt length | boundary is `align_down(total_len - 1, 32)`, so a **≤32-token prompt has no boundary at all** | item 1 below |

The second is also the standing `short` TTFT puzzle (M10 open question): those turns cannot checkpoint, so
their cross-turn reuse is 0 for a reason unrelated to MTP.

## Work items

1. **A prompt that ends on a block boundary must be checkpointable.** `total_len - 1` excludes the end of the
   prompt, so a one-block prompt (a lot of `short` chat turns) has no boundary. Gate: a 40-token prompt shows
   `snapshot ... tokens=32` (already true) *and* a 32-token prompt shows `tokens=32` (today: nothing), with the
   same output as without reuse.
2. **Deterministic equivalence gate for `=none`.** `--no-lookahead --pcie-frac 1.0`, prompt ≥ 70 tokens sent
   three times, arms `full` vs `none`. Gate: a `snapshot` line exists, and the arms are byte-identical.
3. **Promote MTP into the chat arm** only after 1–2: `harness/chat/engines/naiwa.json` gains
   `--mtp --mmproj … --mtp-n-predict 4` plus the replay override. Gate: full 3-round corpus, `doc`/`long`
   TTFT≥2 within the M10 baseline (3.67 s / 6.13 s turn 1) *and* TPOT ≤ 38 ms on at least one class.
4. **Flatten the row-proportional host term** (`0.25 ms/row/layer`), which is what makes the amortisation
   pay at all: `dry` must stop scaling with rows. This is M10 item 3b's device-side dispatch, re-aimed at np5.

Non-goals here: KV int8, host-mapped embedding, the prefill blob-ring. Those are memory/`e2e` terms, tracked
in M10.

## Results

### item 1 — aligned prompts checkpoint (done)

`recurrent_checkpoint_boundary` now takes `align_down(total_len, 32)` and accepts `boundary <= total_len`.
Oracle re-verified after the change, both variants byte-identical to `harness/m1/reference.oracle.txt`
(`frac 0 --no-lookahead`, with and without `--prefill-stream`); the oracle runs `--prefix-cache-n 0`, so this
path is inactive there by construction.

### item 2 — the `=none` equivalence gate (done, passing)

`--no-lookahead --pcie-frac 1.0` (deterministic), MTP on, 96-token prompt sent three times:

| arm | `PREFIX_DBG` | three responses |
|---|---|---|
| `none` | `snapshot: seq 1 tokens=64 blocks=2`, restore never reports `no_replay_boundary` | `09cc90a793cd` ×3 |
| `full` | no snapshot, `no_replay_boundary cached=64` twice | `09cc90a793cd` ×3 |

So reuse demonstrably happens with `none`, does not with `full` (the must-fail control), and the outputs are
byte-identical either way.

### item 3 — MTP in the chat arm (measured, gate partly met)

`harness/chat/engines/naiwa.json` now carries `--mtp --mtp-n-predict 4` and
`env: {MISTRALRS_MTP_PREFIX_REPLAY: none}`; `run.py` learned an `env` passthrough and prints it in the launch
line so a run reproduces from its own log. 3 rounds, `short`+`doc`+`long`, naiwa only (medians):

| class | TTFT t1 | TTFT ≥2 p50 | TPOT p50 | reuse p50 | e2e/conv | M10 baseline (no MTP) |
|---|---:|---:|---:|---:|---:|---|
| doc | 9 528 ms | 4 396 ms | **43 ms** | **0.96** | **50.7 s** | 6 060 / 3 670 / 55 / 0.83–0.94 / 52.1 |
| long | 9 084 ms | 3 560 ms | **44 ms** | **0.89** | **46.6 s** | 6 130 / 3 800 / 55 / 0.83–0.94 / 53.5 |
| short | 3 050 ms | 4 443 ms | **52 ms** | **0.76** | **60.5 s** | — / 4 570 / 55 / — / 63.6 |

**What the milestone was for:** reuse is back (0.76–0.96 vs **0.00% with MTP** before) and TPOT drops
55–56 → 43/44/52 ms, i.e. −22%; end-to-end is better on all three classes.

**What it did not get:** the gate asked for TPOT ≤ 38 ms on one class (got 43) and `doc` TTFT≥2 within the
baseline (got 4 396 vs 3 670 ms). Two separate causes, both now measured:

| new cost | number | where |
|---|---|---|
| turn-1 prefill with MTP | 6.06 → **9.53 s** (284 → 181 tok/s) on `doc` | the drafter prompt path: the chunk plan still picks 8192 (log confirms), so it is not chunking; candidate is the speculative prompt chunk forcing the unpacked attention policy |
| row-proportional host term | `dry = 1.00 + 0.25 × rows` ms/layer | item 4: at 5 rows the round trip is only half amortised, and it is what keeps TPOT at 43 instead of ~20 |

## Turn-1 prefill: decomposed (2026-10-21)

A 1,725-token `doc` turn 1, one request, `--prefill-stream --prefill-stream-min-tokens 64`, MTP on, timed through
`PREFIX_DBG` (`prefill:` per chunk, `prefill_step:` per step, `drafter_prefill:` inside the drafter):

| piece | time | rate | note |
|---|---:|---:|---|
| target forward, 1,696 rows | 3.50 s | 485 tok/s | same as the no-MTP arm (3.48 s) — unaffected by MTP |
| **drafter prefill, 1,696 rows** | **2.73–4.01 s** | 424–620 tok/s | `mtp_forward`; embed 6 ms, hidden/positions µs |
| tail chunk, 29 rows | 2.32–2.63 s | ~12 tok/s | below `--prefill-stream-min-tokens`, so it is a CPU prefill; the **no-MTP arm pays this too** |
| total | 9.0–10.4 s | 166–190 tok/s | vs the no-MTP 6.1 s (3.48 + 2.3) |

Three separate causes, and the accounting matters because they are paid differently:

1. **The drafter prefill is a CPU MoE, once per conversation.** The MTP head's experts are always loaded to the
   host (`moe/provider.rs`: `if ctx.site.is_mtp` returns `CpuExperts` before `attach_prefill` runs, so it never
   gets a prefill stream either — `prefill_stream.rs::attach` also excludes `is_mtp`). For a prompt chunk it
   falls through to `MoEExperts::forward` → `FastExpertsWeights::forward_gather`, which on the CPU pays one
   weight-block read **per (token, expert) pair**: 1 696 × 8 × ~2.6 MB ≈ **35 GB** at the ~9–10 GB/s the
   one-row kernel gets. That is the whole 3–4 s. A grouped CPU kernel (one GEMM per expert, ~26 rows each)
   would read the same 1.35 GB once: **≈ 0.2 s**.
2. **It is paid once, not per turn.** With reuse enabled, a repeated request restored 1 696 tokens and its
   drafter prefill covered only the 29 new rows (`mtp_forward` 50–65 ms). So this is a cold-conversation cost:
   −4 s on `e2e/conv` for `doc`/`long`, not on TTFT≥2.
3. **The 29-row tail is a pre-existing defect, not MTP's.** `align_down(total, 32)` splits 1 725 as
   1 696 + 29, and a 29-row chunk touches ~all 128 experts of every layer, so it is expensive either way
   (CPU 2.63 s at `min-tokens 64`; stream 2.32 s at `min-tokens 8` — 128 experts × 1.88 MB × 48 layers ≈
   11.5 GB of PCIe). It exists in the no-MTP arm too and is the bulk of the M10 "TTFT≥2 3.67 s" number.

## Gates

- Oracle unchanged: `frac 0 + --no-lookahead` byte-identical to `harness/m1/reference.oracle.txt`, with and
  without `--prefill-stream`. MTP is *not* in the oracle path (it is a chat-only arm), so item 3 additionally
  needs its own equivalence gate (item 2).
- One mechanism per run; medians of ≥ 3 turns; failed rounds are recorded, not retried.
- Status of the three gates, 2026-10-21: oracle ✅ (both variants); reuse equivalence ✅ (item 2 table);
  chat corpus ⚠️ partly (TPOT −22% and e2e better on all classes, but TPOT > 38 ms and `doc` TTFT≥2 above
  baseline — see the two costs above). Item 4 is what closes it.

## Next

1. **Grouped CPU MoE for prefill chunks** (the decomposed cost above): `FastExpertsWeights::forward_gather`
   reads each expert's weights once per (token, expert). Grouping the rows per expert makes the MTP head's
   1 696-row prefill read 1.35 GB instead of 35 GB — 4 s → ~0.2 s, and the same kernel serves every layer that
   keeps its prefill on the CPU (uncached layers, the stream's decline path). Gate: `mtp_forward` ≤ 300 ms for
   1 696 rows, identical logits on the oracle path, 3-round `doc`/`long` e2e −4 s.
2. **Item 4, the row term**: `dry` must stop scaling with rows, or MTP's amortisation stays capped at
   `1.0/(1.0+0.25·R)`. This is the M10 3b device-dispatch work, now with a measured payoff curve.
3. Then re-run the 3-round corpus against strata for the class-level gate.
