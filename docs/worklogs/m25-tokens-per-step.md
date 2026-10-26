# M25 - tokens per step (prompt-lookup suffix drafting)

**Status:** open, **nothing built**. Opened by M24's closing arithmetic.
**Parent:** M24 (§7 the attribution: the decode is PCIe-bound at 394 MB per step, 71% of the link), M22 §15/§16
(acceptance is the `code` class weakness; strata carries `--suffix-draft 3`, we carry nothing).
**One mechanism:** a **prompt-lookup suffix drafter** - propose the tokens that followed the current suffix the last
time it appeared in the prompt (or in the conversation so far), verify them with the existing MTP verify pass, and
consequently produce more tokens per step. No new memory, no new kernel, no model change.

## 1. The law this milestone is aimed at

M24 §8, measured, one relation with two inputs:

```
TPOT  =  bytes_per_step / (12.8 GB/s x tokens_per_step)
```

| input | ours, measured | where it comes from |
|---|---:|---|
| bytes per step | **394 MB** | 4.4 uploaded slots per layer of the 20 touched (22% miss) x 1.88 MB x 48 |
| **tokens per step** | **1.13** | step 43.6 ms against TPOT 38.5 ms; a step verifies 2 rows (`pairs 960` = 48 x 2 x 10) |

Our step spends 31 ms of its 43.6 ms on expert transfer. **Those bytes are already paid for 2 rows of verification**,
so every extra accepted token in that step is close to free: at 1.13 tokens/step a token costs 38.5 ms, at 2.0 it
costs 21.8 ms, at 3.0 it costs 14.5 ms - with the same transfers.

Strata's own numbers show the shape of it: **110.8 MB per token against our 348.7** (§7), and its cache hit rate
(45.9-59.4%) is no better than ours. The difference is that it amortises the same fetch over more output, partly with
`--spec 4` and partly with **`--suffix-draft 3`**, a prompt-lookup drafter whose own docs quote a **0.94-0.99 hit rate**
on continuations.

## 2. Why suffix drafting and not more MTP depth

| option | verdict from existing evidence |
|---|---|
| deeper MTP (`--mtp-n-predict 4+`) | M22 §10 chose auto depth; acceptance falls faster than depth rises on this class (the gpt-oss head is trained for 3), so the marginal draft is mostly rejected |
| larger verify (more rows per step) | rows are already 2; more rows means more bytes per step for the same transfer budget - it moves both terms of the law against us |
| **prompt-lookup suffix** | **costs no memory and no kernel; the draft is a memcpy of history.** It works exactly where MTP does not - code, where the continuation is copied from earlier in the same context (a function repeated, a schema, a test harness) |

This is also why the milestone is worth its own gate: its win is **workload-shaped**. On `doc`/`long` prose the suffix
chain will rarely hit and the mechanism must degrade to exactly today's behaviour, not to something slower.

## 3. Mechanism, precisely

1. **Index**: keep the last `S` tokens of the live sequence (prompt + generated) in a device-resident ring, plus a
   rolling hash of the last `n`-gram (`n` = 4, the usual choice) -> positions in that ring. Built incrementally as
   tokens are appended; no re-scan per step.
2. **Draft**: at each step, look up the current suffix and emit the up-to-`K` tokens that followed it (`K` = 3 to start,
   matching strata's default). Empty match -> fall back to the MTP head exactly as today.
3. **Verify**: unchanged - the drafts join the existing single verify pass (2 rows today, so 1 MTP draft + up to 3
   suffix drafts need the row budget reconsidered; that is the main open design question, see §5).
4. **Greedy only at first**: with `temperature 0` the verify is exact and the suffix draft is pure win; sampling needs
   the acceptance-ratio correction and is explicitly out of scope.

## 4. Gates

| gate | requirement |
|---|---|
| numeric oracle | `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt`; the suffix drafter is disabled there (like `--mtp` today), and a **must-fail control** proves it: with the drafter on and a deliberately wrong suffix match, the output must differ - i.e. verification really is checking |
| **the metric** | **tokens per step on `code` rises from 1.13 to >= 1.6**, and MB/token falls from 348.7 to <= 250. This is the gate; TPOT is recorded, never a gate |
| workload shape | `doc`/`long`/`short` tokens/step must not **fall** below today's value (the mechanism must be a no-op when the chain misses) |
| performance | medians of >= 3 rounds via `harness/chat/run.py --engines naiwa,strata --rounds 3` |
| liveness | 3 rounds x 4 classes, 0 failures; `cargo test` 53 passed / 4 ignored |
| memory | **zero** additional device memory beyond the ring (a few MB), which is the whole point over a bigger cache |

## 5. Open design question this milestone must answer first

The verify pass runs **2 rows** (1 MTP draft + 1 base). Three suffix drafts plus the MTP draft plus the base is 5 rows,
and 5 rows means 5 x the expert traffic per layer if the routed sets differ - which **moves the numerator of §1's law
against the denominator**. The first experiment is therefore arithmetic, not code: with the expert-profile from
`data/expert-profile-coder.bin`, compute the expected `distinct` count for 2, 3, 4 and 5 rows on the `code` corpus,
and pick the row budget whose `tokens/step / bytes/step` ratio is best. If no row budget beats today's, **M25 closes
negative before a line of drafting code is written.**

## 6. Not in scope

- The numerator (fewer bytes per miss): prefetching `gate_up` ahead of `down`, pinning, promote tuning. Queued behind
  this milestone because it touches the 2.2 GB ceiling from M20.
- Sampled decoding acceptance corrections.
- Anything about the kernels (M22.4 closed the repack at 1.08x).

## 7. §5's question, answered with measurements: an extra row costs ~200 distinct experts, and only pays if the row is a suffix row

The arithmetic was supposed to come before the code, so here it is. Same probe (`code`, three 40-token turns,
`NAIWA_STEP_BUDGET=1`, cache 2.2 GB), two row budgets:

| rows | how | `pairs` | `distinct` per step | step | TPOT | **tokens/step** | tokens per 1 000 distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| **2** | `--mtp-n-predict 1` | 960 | 722-770 | 43.6-53.7 ms | 38.1-38.8 ms | **1.13** | **1.52** |
| **4** | `--mtp-n-predict 3` | 1920 | 1 120-1 198 | 73.5-86.5 ms | 36.6-38.9 ms | **1.90** | **1.65** |

Three things fall out:

1. **Doubling the rows does not move TPOT** (38.5 -> 38.0 ms), which is exactly why M22's auto depth landed where it
   did. `distinct` grows **1.54x** (745 -> 1 159) while tokens/step grows **1.68x**, so the efficiency gain is **8%** -
   real, but far too small to see through the +-8% run-to-run noise.
2. **The marginal row costs ~200 distinct experts** ((1 159-745)/2), i.e. about **106 MB of upload** at 394 MB per 745
   distinct. A marginal *MTP* row returns ~0.38 accepted tokens (1.90-1.13 over two rows), so its price is
   ~280 MB/step per token.
3. **A suffix row is a different animal.** Its draft is copied from history, and strata's own documentation claims a
   **0.94-0.99 hit rate** on continuations; at even 0.6 accepted tokens per row it would return ~1.6x the tokens of an
   MTP row for the same ~106 MB. That, and only that, is the case for building it.

### Pre-registered estimate, and the number that kills it

Five rows (1 base + 1 MTP + 3 suffix) extrapolate to `distinct` ~1 350 and a step of ~79 ms; at 4.4 tokens/step that
is **TPOT ~18 ms (-53%)**, and the same arithmetic at 2.5 tokens/step gives 31 ms (-20%).

| what the first measurement must show | else |
|---|---|
| suffix drafts accepted at **>= 0.5 per row** on `code` | the added rows buy less than an MTP row: **M25 closes negative**, the mechanism is dead as designed, and the numerator (fewer bytes per miss) becomes the only remaining lever |
| `doc`/`long`/`short` tokens/step not falling | the chain is firing on prose and burning rows for nothing, so it must be gated by a prefix check (length and match run) |

The estimate is pre-registered here so that the eventual result cannot be read backwards, and the falsifier is a single
number: **accepted suffix tokens per drafted row on the `code` class.**
