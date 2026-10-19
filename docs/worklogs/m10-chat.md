# M10 — multi-round chat: naiwa-rs vs Strata

Status: **M10.1 done and M10.2 measured (2026-10-19)**. M10.3 (align) not started.
Parent: [m9c.md](m9c.md). This is the roadmap's **M10** (`docs/roadmap.md` §M10).
(`worklogs/m10.md` is the older *MoE naming and ownership* milestone from the pre-rescope numbering; it is
unrelated and stays.)

Scope: the **measurement**, not the alignment. Harness: [`harness/chat/`](../../harness/chat/README.md).

## Why rounds, not another microbenchmark

Every number in `design.md` §5 is a single request at `--depth 4`: MoE-dominated, ~no attention, no prefix reuse.
A chat session is the opposite. Turn *n*'s prompt is turns 1..*n*'s transcript, so

- (a) prompt processing grows every turn,
- (b) almost all of that prompt is a prefix the engine already holds,
- (c) per-turn TTFT is decided by prefix reuse and prefill batching, not by decode throughput.

**The measured unit is a conversation, not a request.** That is also the only shape in which a parity claim
means anything: Strata quotes single-shot "writes answers" / "reads your prompt" numbers, and long chats are a
separate table in its own docs.

## Feasibility on this box (checked, then used)

| fact | evidence |
|---|---|
| Strata runs on the same card | `~/code/strata`, RTX 20–50 support; `engine/strata` + `.venv` present |
| same model artifact | both load `~/models/qwen3-coder/IQ1_M/…-0000{1,2}.gguf`; strata additionally needs `~/code/Strata-data/packs/coder-iq1_m` |
| one client view of both | both are OpenAI-compatible and both report `prompt_tokens_details.cached_tokens` (strata maps its `cache_n`/`reused`) |
| thinking pinned off on both | `enable_thinking: false` (naiwa) / `reasoning_effort: "none"` (strata) |
| engines cannot co-reside | 12 GB VRAM / 60 GB RAM; sequential, PID-only stop, order alternated per round |

## Protocol

### Corpus (`harness/chat/corpus.jsonl`, committed, hash in `run.json`)

One line = one conversation: `{"id","class","system","turns":[{"user","max_tokens"}]}`. `{{DOC}}` on line 1's
`_constants` expands the 1.7k-token document used by `doc` and `long`.

| class | shape | isolates |
|---|---|---|
| `short` | 8 turns × 96 tok | per-turn TTFT with reuse; decode |
| `doc` | 1.7k-token document, then 6 questions | prefill + prefix reuse dominate |
| `code` | 4 turns × 160 tok | long outputs; draft acceptance |
| `long` | 1.7k turn, then 2 × 400 tok | KV growth near the context limit (M10.4) |

**The transcript grows with each engine's own streamed answer**, exactly as a chat client does.
A fixed transcript of *authored* assistant turns cannot measure reuse: mistralrs snapshots the
recurrent prefix at the end of a **prompt**, so a request whose shared prefix ends in text the engine never
generated never matches. Measured both ways: authored transcripts → zero reuse on naiwa for seven turns
(`--prefix-cache-n 1` and `2`); live answers → the same engine still reports zero, and strata reports 0.95–0.99
(see findings 1 and 3). Turn 1 is identical on both engines, which is where prompt-token equality is checked
(it was: `short` turn 1 = 40 tokens on both, `doc` turn 1 = 1723 on both).

### Runs

- **Round** = cold engine process, wait for `/v1/models`, one discarded 2-turn warmup conversation, every
  conversation in the corpus, stop by PID. Order alternated per round.
- `short`+`code`: **3 rounds**, both engines. `doc`+`long`: **1 round**, both engines — naiwa's `doc` cost
  1,136 s per conversation, so three rounds would have been an hour of pure re-prefill.
- Concurrency **1**: strata serves one sequence behind a FIFO, so any other number is not like-for-like.
- VRAM/RSS sampled at 1 Hz.

### Per-turn metrics (from the SSE stream, not engine logs)

TTFT (request write → first content delta); **TPOT** (decode window / gaps — a speculative decoder delivers
several tokens per step, so an ITL *median* reads 0); decode tok/s; end-to-end; `prompt_tokens`,
`completion_tokens`, `cached_tokens`, `reuse = cached/prompt`. Turn 1 has nothing to reuse, so the headline is
turns ≥2, and the per-turn curve is the deliverable.

### Pinned configuration (both sides, recorded per run)

| | naiwa-rs | strata |
|---|---|---|
| model | IQ1_M native, both shards | same shards via `packs/coder-iq1_m` |
| context | `--pa-context-len 4096` | `--max-context 4096` |
| KV | BF16 paged, 32-token blocks | `--kv int8` |
| expert cache | `--expert-cache-gb 1.5` | `--expert-cache 800` blobs (~1.5 GB) |
| speculation | `--mtp --mtp-n-predict 4` | `--spec 4 --spec-min-p 0.5` |
| `pcie_frac` | 0.2 | its own startup probe picked 0.37 |
| prefix reuse | `--prefix-cache-n 1` | one resident sequence + conversation checkpoint |
| peak VRAM / RSS | 11.55 GiB / 56.0 GiB | 7.04 GiB / 26.5 GiB |

Strata's `--expert-cache auto` fills the card and is not a comparison, so it is pinned to naiwa's budget.
**Speculation cannot be paired off**: strata refuses to start a native (IQ) pack without `--spec T, T >= 2`
("it needs `--native SHARD1, --spec T (T >= 2)`"), so the paired arm has speculation on both sides.
`--prefix-cache-n` on this model is a **recurrent state snapshot of ~111 MB**, not a few KV blocks: the default
16 needs 1.9 GB and does not fit beside a 1.5 GB cache at 4096 context. Pinned to 1.

## M10.2 baseline — 2026-10-19

Both engines, `spec` arm, concurrency 1, thinking off, medians. Raw runs:
`~/bench-rel/m10/2026-10-19` (3 rounds) and `~/bench-rel/m10/2026-10-19-long` (1 round);
92 turns, 0 failures.

```bash
python3 harness/chat/run.py --engines naiwa,strata --rounds 3 --classes short,code \
    --out ~/bench-rel/m10/2026-10-19
python3 harness/chat/run.py --engines naiwa,strata --rounds 1 --classes doc,long \
    --out ~/bench-rel/m10/2026-10-19-long
```

| class | engine | TTFT t1 | TTFT ≥2 p50 | p90 | TPOT p50 | decode tok/s | reuse p50 | e2e/conv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | naiwa | 3,095 ms | **40,630 ms** | 66,333 ms | 59 ms | 17.0 | **–** | 333.2 s |
| `short` | strata | 1,030 ms | **614 ms** | 809 ms | 38 ms | 26.1 | **0.95** | 29.1 s |
| `code` | naiwa | 3,041 ms | 31,950 ms | 36,524 ms | 40 ms | 25.0 | – | 106.6 s |
| `code` | strata | 985 ms | 1,150 ms | 1,287 ms | 27 ms | 36.5 | 0.67 | 21.2 s |
| `doc` | naiwa | 136,060 ms | 164,196 ms | 175,027 ms | 51 ms | 19.5 | – | 1,136.4 s |
| `doc` | strata | 2,321 ms | 620 ms | 710 ms | 39 ms | 25.9 | 0.99 | 22.1 s |
| `long` | naiwa | 135,588 ms | 182,142 ms | 192,893 ms | 52 ms | 19.3 | – | 550.5 s |
| `long` | strata | 2,251 ms | 777 ms | 827 ms | 38 ms | 26.6 | 0.99 | 34.7 s |

Per-turn TTFT, `short` (identical prompt at turn 1, then each engine's own transcript):

| turn | naiwa | strata | naiwa reuse | strata reuse |
|---:|---:|---:|---:|---:|
| 1 | 3,095 ms | 1,030 ms | – | 0.00 |
| 2 | 12,929 ms | 811 ms | – | 0.80 |
| 3 | 23,794 ms | 609 ms | – | 0.92 |
| 4 | 35,936 ms | 632 ms | – | 0.94 |
| 5 | 45,064 ms | 590 ms | – | 0.96 |
| 6 | 55,204 ms | 660 ms | – | 0.95 |
| 7 | 66,333 ms | 572 ms | – | 0.96 |
| 8 | 76,894 ms | 616 ms | – | 0.97 |

### Reading

1. **The gap is TTFT, not decode.** Turns ≥2: no class is closer than **28×** and `doc` is **265×**.
   End-to-end per conversation: 5–51×. Decode is 1.4–1.6× (TPOT 59 vs 38 ms on `short`, 40 vs 27 on `code`),
   i.e. outside the roadmap's ±10% band but the same order of magnitude.
2. **naiwa reports no prefix reuse at all** — `cached_tokens` is absent from every turn ≥2 of every class, so
   TTFT grows linearly with the transcript (3.1 s → 76.9 s over eight turns) while strata's is flat at
   ~0.6–1.2 s because it reuses 95–99%. Two probes separate "the flag is off" from "the mechanism misses":
   with `--prefix-cache-n 1` and again with `2`, a request that repeats a 216-token prompt and changes only the
   tail reports `cached_tokens 192` (16.9 s → 2.1 s); the same engine over a generated transcript reports none.
   So the recurrent checkpoints exist and are keyed by prompt boundaries that a growing chat transcript does
   not hit — a correctness-of-expectation gap, not a missing flag. Root cause not investigated (M10.1/M10.2 are
   measurement-only).
3. **Prefill is the whole TTFT.** Serve-mode prefill on naiwa is **13 tok/s** (201 tokens = 14.8 s, 743 = 56 s,
   1,827 = 139 s, measured directly with `max_tokens 1`), identical with and without `--prefill-stream`, which
   is why `doc`/`long` turn 1 costs 136 s against strata's 2.3 s (~750 tok/s). Two reasons, both measured:
   - `--prefill-stream` cannot engage with its own defaults: `--prefill-stream-min-tokens` defaults to
     **1024** while the scheduler's `--max-prefill-chunk-tokens` defaults to **512**, so no chunk ever
     qualifies. This is a configuration contradiction, not a missing feature.
   - Forcing it (`--prefill-stream-min-tokens 256 --max-prefill-chunk-tokens 512`) makes it borrow the cache
     ("borrowed 1863 MiB of expert-cache slots") and then OOM at a 2 GB cache on this card. So even reachable,
     it does not currently run at the benchmark's cache budget.
4. **Speculation buys almost nothing in chat on naiwa.** TPOT 59 ms (17 tok/s) is what `design.md` documents
   for *no* MTP (15.8 tok/s), while the np8 synthetic round reaches 52 tok/s. Draft acceptance on real chat/code
   text is the difference; it is an M10.3 question, not a harness one.
5. **Decode-side parity is reachable in principle**: both engines are within 1.5× per token at the same cache
   budget and context. Nothing in the decode path is 10× off; the prefill path is.

### Revised M10.3 agenda (replaces "close the gap on TTFT/ITL/tok/s")

Order by leverage, not by taste:

1. **Prefix reuse across turns** (fixes 28–265× of TTFT): find why the recurrent checkpoints do not match a
   generated transcript, and make a chat turn reuse its own history.
2. **A prefill path that runs** (fixes the 136 s first turn): make `--prefill-stream`'s defaults consistent with
   the scheduler's chunk size, and make it fit at a 1.5 GB cache budget; otherwise raise the CPU prefill rate.
3. **Draft acceptance on chat text** (fixes the 1.4–1.6× decode gap): MTP n=4 on this workload behaves like
   no speculation.
4. Only then re-check the roadmap's ±10% band. Do not tune the MoE/decode path for this workload before 1–3;
   the np8 measurements say it is not where the chat gap is.

## Judging

1. **M10.1 — reproducible harness.** ✅ Same corpus + config → same schema and turn count; 92/92 turns written;
   turn-1 `prompt_tokens` equal across engines (40 = 40, 1723 = 1723); ≥3 rounds for the head-line classes;
   medians; every failure recorded (none).
2. **M10.2 — honest baseline.** ✅ Table above, no tuning, `spec` arm paired, engine order alternated.
3. **M10.3 — align** (open): re-scoped by the baseline — see the agenda above.
4. **M10.4 — long context.** `doc` and `long` are measured (1,723–2,491 prompt tokens, strata at 0.99 reuse,
   naiwa re-prefilling all of it). Deeper contexts stay out of scope until the context/KV budget question in
   "Open questions" is settled.

## Gates (executed)

- Harness: 3 rounds for `short`/`code`, 1 for `doc`/`long` (labelled), medians, order alternated, 0 failed turns.
- Engine untouched: no `src/` change; `frac 0` + `--no-lookahead` oracle is unaffected and was last verified at
  `21cd122`.
- Every number above carries the comparability record in `run.json` and is reproduced by
  `harness/chat/summarize.py`.

## Work items (done)

- `harness/chat/corpus.jsonl`, `run.py` (stdlib-only driver: launch by PID, readiness, warmup, SSE timestamps,
  VRAM/RSS samplers, per-round config copy for strata), `summarize.py`, `README.md`.
- `harness/chat/engines/{naiwa,naiwa-nospec,strata,strata-engine}.json` — launch configs with a `record` block;
  `naiwa-nospec` exists but was not run (strata cannot pair it).
- This table, generated by `summarize.py`, not by hand.

## Open questions

- Why does mistralrs's recurrent prefix checkpoint never match a generated transcript? (`cached_tokens` absent
  for 21/21 `short` turns ≥2 at `--prefix-cache-n` 1 and 2, while a repeated prompt prefix caches fine.) Answer
  it before deciding whether M10.3.1 is a bug fix or a design change.
- `--prefill-stream`'s two contradictory defaults: which one moves, and can the stream run inside a 1.5–2 GB
  cache on a 12 GB card at all?
- Strata's startup probe picks `pcie_frac 0.37`; naiwa's `--pcie-frac 0.2` is a hand-tuned constant. Whether
  that matters for chat (it did not for np8) is unmeasured here.
- Host RSS: naiwa 56.0 GiB vs strata 26.5 GiB. Pinned expert stacks + host PLE gather is the likely cause; not
  investigated, and it is the difference between "runs on 64 GB" and "runs on 32 GB".
- Concurrency >1 is unmeasured: strata is a one-sequence FIFO. Any concurrency claim needs strata's slot/batch
  path configured first.
