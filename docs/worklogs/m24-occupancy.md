# M24 - decode-step occupancy

**Status:** open, **nothing built yet**. This milestone exists because M22 §22's nsys comparison changed the diagnosis:
our decode keeps the GPU busy **52%** of the wall clock where strata keeps it busy **88%**, and our per-kernel launch
gap is **15.1 ms/token** against their 7.5.
**Parent:** M22 (§22 the nsys comparison; §19-§21 the shared-expert hunt it closed), M9.1 (which deleted per-layer
graphs as neutral - see §4's falsification, that history is the main risk here).
**One mechanism:** capture **the whole decode step** as a single CUDA graph and replay it, instead of M2/M4/M9.1's
*per-layer* segments.

## 1. The measurement this starts from

Same probe on both engines (`code`, three 40-token turns, `MTP_N=1` for us, `--spec 4` for strata), nsys
`--trace=cuda --sample=none` (and `--cuda-graph-trace=node` for strata, which already replays a graph per step):

| per token | naiwa | strata |
|---|---:|---:|
| kernel launches | 1219 | 2912 |
| **GPU busy / wall** | **52%** | **88%** |
| GPU time | 16.2 ms | 52.5 ms |
| **launch gap** | **15.1 ms** | **7.5 ms** |
| inside the GPU time: expert-delivery wait | 7.12 ms (44%) | 18.6 + 10.9 ms (56%) |
| inside the GPU time: real compute | **~9 ms** | ~23 ms |

Two readings matter and they pull in opposite directions:

- **Our compute is 5x cheaper per token than strata's** (9 ms against 23), so nothing in the kernels needs work. The
  M22 §18 "shared expert = 17.5 ms" was the doorbell spinning while that stage's ops were in the queue.
- **We idle the GPU for 15.1 ms of every 38 ms token.** That is the whole remaining gap, and it is not arithmetic.

## 2. What the 15.1 ms could be (three candidates, none yet measured)

| # | candidate | what it predicts | how it is refuted |
|---|---|---|---|
| a | CUDA launch API cost: 1219 launches/token at ~12.4 us each | a whole-step graph removes nearly all of it; busy fraction jumps, launch gap collapses | if the gap survives a graph, (a) is dead |
| b | host work between launches (staging fills, `plan`, mmap borrow bookkeeping) blocking the queue | a graph changes nothing - the host still has to do that work before the replay is enqueued | `NAIWA_STEP_BUDGET` already prices the host critical path at 8.1 ms/step; if the gap is 15.1 ms/token the two cannot both be right, so one of them is measuring the wrong thing |
| c | the GPU genuinely has nothing to run because layer L+1's experts are not uploaded yet | the gap sits immediately before the expert kernels, not uniformly | visible in the per-kernel timeline once NVTX marks the host phases |

Note that (a) and (b) need different fixes and (c) needs a third one (pipelining, which would be M25's mechanism).
**M24 only commits to (a)**; §3 measures which one it is before building anything.

## 3. Plan

1. **Attribute the gap, cheaply** (measurement, not a mechanism): wrap the host phases that sit between submissions -
   `plan`, staging fill, borrow bookkeeping, sampling - in NVTX ranges (`nvtxDomainRangePushEx`), which nsys shows
   against the kernel timeline for free as a normal user. Output: the 15.1 ms split into (a) API launches, (b) host
   phases, (c) waiting for uploads. Gate: the split must sum to the measured gap within 1 ms, three runs.
2. **Then, and only if (a) dominates: capture the whole step.** A step is a fixed sequence for a fixed row count, so
   capture per (batch shape, speculation depth) into a graph cache keyed by shape, replay per step. Bounded cache (say
   8 entries), eviction on shape change, and a "replay only when the shape and the routed expert set are identical"
   rule - the routed set is *not* fixed, so the graph must read it from a device buffer the host updates, which is the
   part M2/M4 did not attempt.
3. **If (b) dominates**: stop, close M24 as negative, and hand the host phases to a follow-up that shrinks them.
4. **If (c) dominates**: close M24 as negative too and write the pipelining milestone instead - the work then belongs
   in the expert upload path, not here.

## 4. Gates

| gate | requirement |
|---|---|
| numeric oracle | `frac 0` + `--no-lookahead` **byte-identical** to `harness/m1/reference.oracle.txt` - and, because a graph that silently replays stale inputs would still look fast, **a must-fail control**: replay with a deliberately stale routed-expert buffer must change the output |
| structural | the nsys launch gap per token falls below 8 ms (strata's number) and the busy fraction exceeds 75%; if it does not, the milestone is negative regardless of TPOT |
| performance | medians of ≥ 3 rounds, `harness/chat/run.py --engines naiwa,strata --rounds 3`; TPOT per class recorded, **never** a gate |
| liveness | 3 rounds x 4 classes, 0 failures, `cargo test` 53 passed / 4 ignored |
| regression | `--pcie-frac 0.4`, `--expert-cache-gb 2.2`, `--mtp` auto depth + greedy drafting, promote-on-stage: all unchanged |

## 5. The falsification this milestone must survive

M2/M4 built per-layer graphs, M9.1 measured them **neutral** and deleted them ("No per-layer graphs and no attention
capture - both were measured neutral and are gone"). M24 must explain why a whole-step graph is different, and the
only honest answer is: *it may not be*. Per-layer capture still leaves ~48 graph launches per step plus all the host
work in between; a whole-step capture removes the sequencing entirely. If the attribution in §3 shows the gap is host
work rather than API calls, that history predicts the answer and M24 closes as negative for the price of the
attribution experiment alone.

## 6. Not in scope

- Expert-delivery pipelining (layer L+1's H2D behind layer L's math) - that is candidate (c) and would be its own
  milestone with its own oracle gate.
- Anything in the kernels: §22 measured our compute at ~9 ms/token against strata's ~23, and M22.4 already closed the
  repack with an isolated A/B (1.08x, not worth five subsystems).
- TTFT (M23 stands at its law) and the decode NaN after chunk (tracked separately).

## 7. Attribution (§3 step 1): it is (c) - the decode is PCIe-bound, and it never needed NVTX

The attribution turned out to need no instrumentation at all: nsys already captures the **CUDA runtime API trace** and
the **copy-engine activity**, and those two answer §2's question directly. Same probe (`code`, three 40-token turns,
`MTP_N=1`), 3 decode windows, 120 tokens:

| per token | naiwa | strata |
|---|---:|---:|
| GPU kernel time | 21.4 ms (52% of wall) | 102.5 ms (87%) |
| host CUDA API (serial, calling thread) | 33.9 ms | 20.2 ms |
| **HtoD copies** | **348.7 MB in 27.1 ms** | **110.8 MB in 8.5 ms** |
| small HtoD (`< 1 KB`) | 233 copies, 0.43 ms | - |
| DtoD | 43.0 MB | 0.1 MB |

The 348.7 MB is **509 copies of 685 KB per token** - 685 KB is exactly one `640 x 2560` IQ projection
(640 x 10 x 98 B = 627 KB plus padding), so this is **3 projections x 170 expert slots per token**.

Our own step instrument gives the denominators (`NAIWA_STEP_BUDGET=1`, same probe, TPOT 38.1 / 38.8 / 38.6 ms):

```
STEP_BUDGET layers 48 total 43.6 ms expert_dev 23.4 (48 layers) | plan 36.4 gpu 2.4 staging_wait 2.9 cpu_join 0.3
             | pairs 960 distinct 739 (23% grouping headroom) | waits>200us preload 0/48 staging 0/48 cpu 0/48
```

- `pairs 960` = 48 layers x **2 rows x topk 10**, so a step verifies **2 rows**.
- step 43.6 ms against TPOT 38.5 ms means **1.13 tokens per step** - the acceptance on this class is the worst we have.
- 348.7 MB/token x 1.13 = **394 MB per step**, i.e. **209 expert slots**, i.e. **4.4 slots per layer of the 20 touched
  (22% miss)** - consistent with the 26-slot-per-layer cache and with `distinct 739` of 960 pairs.

### The arithmetic that decides it

394 MB per step at the calibrated **12.8 GB/s** = **31 ms of PCIe inside a 43.6 ms step: 71% of the link, and the
step cannot be shorter than the transfer.** Candidate (a) - launch API - is `gpu 2.4 ms` per step, i.e. **6%**; a graph
can win at most that. Candidate (b) - host work between submissions - is `staging_wait 2.9 + cpu_join 0.3`, another
7%. Candidate (c) is the step's dominant term by a factor of five.

Strata's side of the same table, for scale: 110.8 MB/token in 8.5 ms, **3.1x less traffic per token**, because their
speculation accepts more tokens per step (the same bytes are amortised over more output) and their misses per layer are
comparable. Their own decode is not transfer-bound at all.

### Verdict: M24 closes **negative**, by its own rule

§3 said: *"if (c) dominates: close M24 as negative and write the pipelining milestone instead - the work then belongs
in the expert upload path, not here."* (c) dominates, so the whole-step CUDA graph is **not** the lever: the ceiling it
could reach is 6% of a step, while 71% of the step is a transfer that no launch mechanism can compress.

Recorded for the record, in case anyone revisits: the §22 "launch gap 15.1 ms/token" label was **wrong**. It was wall
time not covered by *kernels*, and most of it is the **copy engine** doing expert uploads - which is invisible in a
kernel-only view. That mislabel is what made a graph look like a 40% opportunity.

## 8. What the next milestone must move

Two quantities, both measured, multiply to the cost:

| quantity | ours | note |
|---|---:|---|
| **MB of expert traffic per step** | **394 MB** | 4.4 slots/layer of 20 touched (22% miss) x 1.88 MB |
| **tokens produced per step** | **1.13** | 2 rows verified; the acceptance is the `code` class weakness from M22 §15 |

`TPOT = bytes_per_step / (12.8 GB/s x tokens_per_step)`. To halve TPOT, halve the numerator or double the
denominator - there is no third way, and both branches are delivery-side:

- **Numerator**: fewer bytes per miss (fetch what is needed earlier, e.g. `gate_up` before `down`, so the matvec can
  start while the rest lands), or fewer misses (capacity is capped at 2.2 GB by M20, so this is a policy question:
  pinning and promote).
- **Denominator**: more tokens per step. Strata gets there with `--suffix-draft 3` (prompt lookup, its doc quotes
  0.94-0.99 hit on continuations); we have no suffix drafter at all, only the MTP head.

Both are worth their own milestone and neither is a kernel change. M25 is opened for the denominator (cheapest, no
memory ceiling involved); the numerator stays queued.
