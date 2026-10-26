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
