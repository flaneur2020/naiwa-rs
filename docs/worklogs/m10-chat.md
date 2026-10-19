# M10 — multi-round chat: naiwa-rs vs Strata

Status: **proposed**. Nothing here is implemented.
Parent: [m9c.md](m9c.md). This is the roadmap's **M10** (`docs/roadmap.md` §M10).
(`worklogs/m10.md` is the older *MoE naming and ownership* milestone from the pre-rescope numbering; it is
unrelated and stays.)

Scope: the **measurement**, not the alignment. Build `harness/chat/`, run one honest side-by-side baseline of
multi-round chat on the same card with the same model artifact, and record it. Tuning is M10.3 and must not start
before the baseline exists.

## Why rounds, not another microbenchmark

Every number in `design.md` §5 is a single request at `--depth 4`: MoE-dominated, ~no attention, no prefix reuse.
A chat session is the opposite. Turn *n*'s prompt is turns 1..*n*'s transcript, so

- (a) prompt processing grows every turn,
- (b) almost all of that prompt is a prefix the engine already holds,
- (c) per-turn TTFT is decided by prefix reuse and prefill batching, not by decode throughput.

**The measured unit is a conversation, not a request.** The per-turn table is the primary output and every
aggregate is derived from it. That is also the only shape in which a parity claim means anything: Strata quotes
single-shot "writes answers" / "reads your prompt", and long chats are a separate table in its own docs.

## Feasibility on this box (checked, not assumed)

| fact | evidence |
|---|---|
| Strata runs on the same card | `~/code/strata` supports RTX 20–50 (≥12 GB); `engine/strata` built 2026-10-13; `.venv` present |
| same model artifact | both load `~/models/qwen3-coder/IQ1_M/Qwen3.8-…-0000{1,2}.gguf`; strata additionally needs `~/code/Strata-data/packs/coder-iq1_m` + `data/expert-profile-coder.bin` → record the pack hash |
| a single client view of both | both are OpenAI-compatible, and both report `prompt_tokens_details.cached_tokens` (strata maps its engine `cache_n`/`reused`) → **prefix reuse is observable without engine hooks** |
| thinking can be pinned off on both | `reasoning_effort` / `enable_thinking` are request fields in both servers |
| engines cannot co-reside | 12 GB VRAM / 60 GB RAM; strata's own `bench/bench_vs_llama.py` already works this way: sequential, PID-only stop, order alternated per round |

## Protocol

### Corpus (`harness/chat/corpus.jsonl`, committed)

One line = one conversation:

```json
{"id":"short-01","class":"short","system":"…","turns":[{"user":"…","max_tokens":96}, …]}
```

- The driver replays turns in order and **appends the assistant's streamed answer verbatim** to the message list,
  so every turn is a strict prefix extension of the previous prompt. That is what makes reuse measurable: a turn
  that is not a prefix extension is not testing the cache.
- Sampling is fixed for both engines: `temperature 0`, `top_p 1`, `seed 0`, `enable_thinking false` /
  `reasoning_effort "none"`, per-turn `max_tokens` fixed in the corpus, no extra `stop`.
- Classes (one conversation each to start; more only where a class proves noisy):

| class | shape | what it isolates |
|---|---|---|
| `short` | 8 turns × ≤96 tok, no system prompt | per-turn TTFT with reuse; decode |
| `doc` | one 2.2k-token pasted document, then 6 questions | prefill + prefix reuse dominate |
| `code` | 4 turns of code continuation, 160 tok/turn | long outputs; MTP acceptance |
| `long` | one 2k turn, then 2 turns × 400 tok | M10.4: KV growth near the context limit |

### Runs

- **Round** = start one engine cold, wait for `/v1/models`, run one *discarded* warmup conversation, run every
  conversation in the corpus, stop by PID. Two engines × ≥3 rounds, **order alternated** each round.
  Restarting per round is deliberate: a warm process carries the previous round's prefix cache into this one and
  hides the first turn.
- One engine at a time, never co-resident. VRAM / RSS sampled at 1 Hz for the peak columns.
- Concurrency **1** for this protocol. Strata serves one sequence at a time behind a FIFO, so any other number is
  not like-for-like; a `--concurrency 1,2,4` sweep is naiwa-only and is reported apart (roadmap M10.3).

### Per-turn metrics (from the SSE stream, not from engine logs)

| metric | definition |
|---|---|
| TTFT | request write → first content delta |
| ITL p50/p90 | delta→delta within a turn, first delta excluded |
| decode tok/s | generated tokens / (last delta − first delta) |
| end-to-end | request write → last delta / `[DONE]` |
| reuse | `cached_tokens / prompt_tokens` from `usage`, with `prompt_tokens` itself |
| growth | TTFT(turn n) / TTFT(turn 1) — the curve is the deliverable, not one number |
| failure | non-200, timeout, `finish_reason=length`, or a cross-engine prompt-token mismatch |

Turn 1 has nothing to reuse, so the **headline is turns ≥2**: TTFT p50 and ITL p50 over turns 2..N per class,
plus decode tok/s. Turn 1 is reported separately as the cold-prefill cost.

### Comparability record (written into every result file)

engine + commit, GGUF hashes, strata pack/expert-profile hash, card + driver, KV dtype (strata `int8` vs naiwa
BF16 paged — recorded, not buried), context length, expert-cache size, `frac`, spec/MTP depth and acceptance,
sampling, prefix-cache settings (`--prefix-cache-n` vs strata's conversation checkpoint), concurrency, corpus hash.

### Two arms, because speculation is not symmetrical

Primary arm = each engine's **shipped** config
(strata `strata-coder-iq1_m.json`: `--spec 4 --spec-min-p 0.5 --kv int8 --max-context 32768`;
naiwa: `--cpu-moe --expert-cache-gb 4 --pcie-frac 0.2 --mtp --mtp-n-predict 4 --pa-context-len 32768`).
Second arm = speculation off on both (`--spec 0` / no `--mtp`), which separates decode speed from draft
acceptance. Report both; never pair one engine's spec-on number with the other's spec-off number.

## Judging

1. **M10.1 — the harness is reproducible.** Same corpus + config → same schema and turn count; `prompt_tokens`
   per turn agree across engines within **2%** (a larger gap means a different tokenizer or template, i.e. a
   different prompt: the run fails rather than being averaged); ≥3 rounds; medians, never means; every failure
   recorded and attributed to a turn.
2. **M10.2 — an honest baseline.** One table, per class × engine × arm, no tuning in this step.
3. **M10.3 — align** (its own milestone): naiwa within 10% of strata on TTFT p50 / ITL p50 / tok/s at concurrency
   1 and 15% at 4, revised *before* tuning if M10.2 shows the band is unrealistic.
4. **M10.4 — long context** is the `long` and `doc` classes here; it replaces the 4-token extrapolation, so there
   is no separate long-context measurement.

## Gates

- Harness: same corpus + config → same schema; ≥3 rounds; engine order alternated; medians; peaks sampled.
- Engine untouched: no `src/` change in M10.1/M10.2; `frac 0` + `--no-lookahead` stays byte-exact against
  `harness/m1/reference.oracle.txt`.
- Every parity claim carries the comparability record. A structural difference (KV dtype, spec depth,
  one-sequence FIFO) belongs in the table, not in a footnote.

## Work items

- `harness/chat/corpus.jsonl` + `harness/chat/README.md` (protocol, exact commands, what may not change between
  rounds).
- `harness/chat/run.py`: stdlib only (`urllib`, `json`, `time`; optional `psutil` for RSS), one SSE parser for
  both engines, one JSON per turn, one summary per run directory.
- `harness/chat/engines/{naiwa,strata}.json`: launch config in the shape strata's own bench uses
  (`exe`, `args`, `cwd`, `port`, `log`), so either side is driven without editing the driver.
- `harness/chat/summarize.py`: per class × engine × arm → TTFT p50/p90 (turn 1 vs ≥2), ITL p50/p90, tok/s, reuse,
  peaks; emits the markdown table that goes into this file.
- The M10.2 table is appended **here**; no second results document.

## Open questions

- Strata is one sequence behind a FIFO. If its `--batch-groups`/slot path is enabled in the config under test it
  is not (it is not in `strata-coder-iq1_m.json`), concurrency 4 is comparable; otherwise label the concurrency arm
  naiwa-only.
- Prefix reuse is **not the same mechanism** on the two sides: naiwa has a 16-entry paged prefix cache at 32-token
  block granularity; strata holds one conversation checkpoint (`strata_prefix`, `SLOT_PREFIX_MIN` 512, KV
  save/restore, int8). Equal `reused` ratios therefore do not imply equal TTFT — the per-turn table has to show
  it, and M10.3 must not "fix" a mechanism difference by matching a number.
- `long` requires `--pa-context-len` = strata's `--max-context` = 32768. If 32768 does not fit beside a 4 GB
  expert cache on this card, the class shrinks (8k) and that is recorded as a scope limit, not tuned around.
- Strata needs its own expert-cache size pinned (`--expert-cache auto` picks by free VRAM): pin it to the same
  budget as naiwa's `--expert-cache-gb 4` or record the difference.
- If a turn hits `max_tokens` on one engine and not the other, the next prompt diverges and the conversation stops
  being comparable: the driver records `finish_reason` per turn and the summariser refuses to pair such a
  conversation.

## First command (sketch)

```bash
python3 harness/chat/run.py --engines naiwa,strata --rounds 3 \
    --corpus harness/chat/corpus.jsonl --arm shipped \
    --out ~/bench-rel/m10/2026-10-20
python3 harness/chat/summarize.py --out ~/bench-rel/m10/2026-10-20
```
