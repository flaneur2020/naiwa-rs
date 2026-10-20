# harness/chat — multi-round chat benchmark

One conversation, replayed turn by turn against an OpenAI-compatible server, with every streamed
token timestamped. Built for M10 ([`docs/worklogs/m10-chat.md`](../../docs/worklogs/m10-chat.md)):
does naiwa-rs serve a *chat session* at Strata's performance?

```bash
python3 harness/chat/run.py --engines naiwa,strata --rounds 3 \
    --classes short,code --out ~/bench-rel/m10/$(date +%F)
python3 harness/chat/summarize.py --out ~/bench-rel/m10/$(date +%F)
```

`run.py` prints a line per turn; `summarize.py` prints the markdown tables that go into the
milestone. Results belong in the milestone document, not in this directory.

## Protocol

| rule | why |
|---|---|
| **one engine at a time, cold process per round** | 12 GB VRAM / 60 GB RAM: they cannot co-reside, and a warm process would carry the previous round's prefix cache into the next one |
| **engine order alternates each round** | thermal/clock drift averages out over ≥3 rounds |
| **one discarded warmup conversation per round** | absorbs model load and leaves cache state that no measured conversation shares |
| **startup is retried, turns are not** | a load-time CUDA OOM on this 12 GB card is intermittent (the PLE host-gather fallback fragments the allocator); a failed *turn* is data and is recorded, never retried |
| **concurrency 1** | Strata serves one sequence at a time behind a FIFO; any other number is not like-for-like |
| **the transcript grows with the engine's own answer** | that is what a chat client does, and it is what exercises prefix/KV reuse. A fixed transcript of *authored* assistant turns cannot: mistralrs snapshots the recurrent prefix at the end of a **prompt**, so a request whose shared prefix ends in text the engine never generated never matches, and naiwa measured zero reuse for seven turns that way |
| **turn 1 is identical on both engines** | it is the only turn whose prompt is comparable byte for byte; the driver records `prompt_tokens` so later turns are compared by reuse ratio and TPOT, not by identical text |
| **thinking pinned off** | `enable_thinking: false` (naiwa) / `reasoning_effort: "none"` (strata): a reasoning block would change output length, not engine speed |
| **budget-matched expert cache** | Strata's `auto` fills the card, which is not a comparison. `engines/strata-engine.json` pins 800 blobs ≈ 1.5 GB, matching `--expert-cache-gb 1.5` |
| **`--max-num-batched-tokens 32`** | mistralrs captures a recurrent prefix checkpoint only when a prompt *step* ends on a 32-token block boundary; at the default 4096 a typical prompt never gets one and every turn re-prefills (141.9 s → 5.6 s for a 1.9k-token turn) |
| **two arms** | `--mtp` sets `SpeculativePrefixReplay::Full`, i.e. no cross-turn reuse at all — a cost, not a bug. `engines/naiwa.json` is the shipped config (MTP on), `engines/naiwa-nospec.json` is the one to ship for chat. Strata cannot run without speculation (native IQ pack needs `--spec T, T >= 2`), so it has no nospec counterpart |
| **context pinned to 4096 on both** | enough for the corpus (max prompt ≈ 2.5k); see the VRAM note below |
| **no `src/` change while measuring** | M10.1/M10.2 are measurement-only; the `frac 0` + `--no-lookahead` oracle stays the correctness gate |

Metrics per turn, all from the SSE stream: TTFT (request write → first content delta), TPOT (decode
window / gaps — speculative decoders deliver several tokens per step, so an ITL *median* would read
0), decode tok/s, end-to-end, `prompt_tokens`, `completion_tokens`, `cached_tokens` (=
`prompt_tokens_details.cached_tokens`, both engines report it), and `reuse = cached/prompt`.
Turn 1 has nothing to reuse, so the headline is turns ≥2.

## Files

```
corpus.jsonl                one conversation per line; `_constants` on line 1 supplies {{DOC}}
run.py                      driver: launch, readiness, warmup, replay, timestamps, peaks
summarize.py                run directory → markdown tables
engines/naiwa.json          naiwa shipped arm: MTP 4, 1.5 GB cache, frac 0.2, prefix 1
engines/naiwa-nospec.json   naiwa without speculation, 2 GB cache (naiwa-only arm; the one that reuses)
engines/strata.json         strata shipped arm: --spec 4 (required for a native IQ pack)
engines/strata-engine.json  the engine config the driver copies per round (port/log/context)
logs/<engine>-r<N>.log      server stdout for that round
turns.jsonl, rounds.jsonl, run.json
```

Engine configs are plain JSON: `exe`, `args` (`{port}`, `{out}`, `{round}`, `{config}` are
substituted), `cwd`, `port`, `body` (merged into every request), `record` (the comparability
fields printed by the summariser). `write_config` adds a per-round copy of a config file that an
engine takes by path (Strata).

## Corpora

| class | shape | isolates |
|---|---|---|
| `short` | 8 turns × 96 tok | per-turn TTFT with reuse, decode |
| `doc` | 2.1k-token document, then 6 questions | prefill + reuse dominate |
| `code` | 4 turns × 160 tok | long outputs, draft acceptance |
| `long` | 2k turn, then 2 × 400 tok | KV growth near the context limit |

One conversation per class to start; add lines with new `id`s only where a class proves noisy.

## VRAM notes (measured on this box)

- `--prefix-cache-n N` on this model is a **recurrent state snapshot** of ~111 MB each, not a few
  blocks of KV: 16 snapshots (the default) need 1.9 GB and do not fit beside a 1.5–2 GB expert
  cache at 4096 context. The harness pins 1.
- `--prefill-stream` defaults are unreachable in serve mode: `--prefill-stream-min-tokens` defaults
  to 1024 while `--max-prefill-chunk-tokens` defaults to 512, so no chunk can ever qualify. Forcing
  it (`--prefill-stream-min-tokens 256 --max-prefill-chunk-tokens 512`) makes it engage and then OOM
  at a 2 GB cache on this card. Both are open findings, not harness bugs; the shipped arms run
  without it.
- Strata additionally runs its own PCIe probe at startup and derives `pcie_frac` from it (0.37
  here), which naiwa cannot match; it is recorded in the round log.

## Reusing this harness

- New engine: add `engines/<name>.json`; no code change if it speaks `/v1/chat/completions`.
- New class: append a line to `corpus.jsonl`; `{{DOC}}` expands from the `_constants` line.
- A failed turn is data: it is written with `error`, the round aborts, and the summariser prints it.
  Do not retry silently — a retry changes the transcript.
