# naiwa-rs v2 — roadmap

Standalone implementation order. Architectural detail lives in
[`architecture-v2.md`](architecture-v2.md); this file is the plan of record for
*what to build next and how each step is judged*. Recorded 2026-10-16.

Progress scratch per milestone lives in [`worklogs/`](worklogs/) (`worklogs/m0.md`, …).

---

## Judging principles

Every step must be **feasible on its own** and **independently verifiable**:

1. **Numeric oracle** — a frozen eager reference: fixed prompt set + greedy decode.
   Bitwise identity is realizable at two levels we can actually use:
   - **kernel level** — diff two launches on the same input (M1.1: `0.000000`);
   - **engine level with `--pcie-frac 0`** — measured 2026-10-16: **8/8 runs identical**
     across two dispatcher configs (spin pool and single-thread rayon).
   It breaks down only with the **PCIe path on** (`--pcie-frac 0.55`: 5 runs gave 2
   continuations, 3× "The capital of Japan…" / 2× "The Eiffel Tower…"). So use
   `--pcie-frac 0` for M1/M2's engine-level oracle; **M3/M4 must first characterize why the
   PCIe path is nondeterministic** (float order from extra `atomicAdd` terms, or a
   staging-ring read race — unknown). See [`worklogs/m1.md`](worklogs/m1.md).
2. **Structural criteria** — launch count / capture count / allocation count / dry
   bucket. These do not depend on performance numbers, so they can be judged inside
   the ~8% run-to-run noise. Performance is recorded, never used as a gate.
3. **One new mechanism per milestone** — so a failure has a single cause.

## Dependency chain

```
M0 ── M1 ── M2 ── M3 ── M4 ── M5        M6 (needs Q2 + F1 + F5)
                    └────── M5 measurement
```

`M5`'s implementation only needs `M2`, but its *measurement* needs `M4` (the host
must already be off the critical path to make the comparison fair).

---

## Milestones

### M0 — zero-argument doorbell primitives (harness, ~1 day) — ✅ DONE 2026-10-16

- **Does**: prove the two properties everything else rests on —
  ① host→GPU visibility of the zero-argument `wait_ge_ptr` / `wait_copy_ptr`
  (target and slot offset live in mapped memory, not in kernel arguments), on the
  target card's fence semantics; ② the same kernel captured into one CUDA graph
  and replayed 200 times while the host rewrites the mapped values between replays.
- **Depends on**: nothing. **Hard blocker for everything.**
- **Judged by**: ① 1000/1000 iterations visible and correct; ② 200/200 replays correct;
  ③ the kernel demonstrably *waited* (`waited == iters/replays`) — the structural check that
  catches argument baking.
- **Artifact**: source `harness/m0/`; binaries + logs `~/bench-rel/naiwa-v2/m0/`.
- **Result**: all pass on RTX 4070 (Ada). Full evidence: [`worklogs/m0.md`](worklogs/m0.md).
  Two findings worth carrying forward: on Ada neither fence is load-bearing for this
  pattern (kept for RDNA portability), and **correct values are not evidence of no baking**
  — only the structural `waited` check caught the baked-target control.

### M1 — static address layout + descriptor-driven kernel (no graph) — ✅ DONE 2026-10-16

> See [`worklogs/m1.md`](worklogs/m1.md). Engine wiring landed in M2 step A: `serve --pcie-frac 0` is **byte-identical to the frozen oracle** with 48 captures engaged, and the ring set is frozen across the decode. Recon: The F4 target is
> `indexed_moe_fused_decode` (the GPU path), **not** the CPU `fused_moe_decode` the plan
> named. `MoeDecodeCapture` (fixed-grid, fixed buffers) is verified **bit-identical** to the
> host-geometry path under eager (fork `729d2c64`); the v2 engine is the imported `naiwa-rs`
> baseline with the 7-ring warmup + a frozen-ring-set assertion (`--pcie-frac > 0`: 3 rings,
> all created after load, 0 created during decode). Remaining: route the engine's resident-hit

- **Does**: ① `load → 7-ring warmup` — create the 7 quant shapes' staging rings and
  all per-layer banks once, before any capture; ② all `FlagBuf` / descriptor /
  staging buffers allocated at load, addresses fixed for the process lifetime;
  ③ `fused_moe_decode` becomes descriptor-driven ("read count → grid-stride over a
  descriptor array", F4).
- **Depends on**: M0 + F4.
- **Judged by**: ① no new allocation and no address change across a whole decode
  (assertion); ② **under eager with `--pcie-frac 0`**, old host-computed launch geometry vs
  descriptor-driven is token-identical — bitwise at kernel level (done, M1.1) **and** at
  engine level (the oracle is bit-reproducible at `frac 0`); ③ no graph / PCIe machinery
  involved.
- **Why independently verifiable**: it does not touch capture at all, so the largest
  kernel rewrite is tested on the existing eager path. A failure is a kernel regression,
  not an architecture failure.

### M2 — per-layer MoE graph skeleton (resident path only) — ✅ DONE 2026-10-16

> See [`worklogs/m2.md`](worklogs/m2.md). The MoE segment captures into one CUDA graph and
> replays with live staging data. Engine integration (steps A and B) routes the resident-hit
> path through it: warmup creates 48 captures, each decode step replays one graph per bank, and
> `serve --pcie-frac 0` is **byte-identical to the frozen oracle**. 150 captures cost 11.6 ms.
> The registry is per-`Bank` (`BankCapture`) rather than a typed `MoEGraphKey` map, which is the
> same thing at this granularity.

- **Does**: the MoE registry (`MoEGraphKey`, architecture-v2 §6.2 — **not** the
  whole-step key), the F2 emitter, per-layer capture and launch.
- **Depends on**: M1.
- **Judged by**: ① fully-resident config: greedy identical to eager — **byte-identical**
  (engine, `--pcie-frac 0`, vs the frozen oracle); ② the **MoE segment** goes from ~5 raw
  launches to **1 `cuGraphLaunch`** (380 replays measured; *not* the whole layer — attention is
  still eager under Q1=(c), so the whole-step 60→1 belongs to Q1=(b)); ③ the one-time cost of
  150 captures is measured — **11.6 ms**; ④ pad rows: the pad is still valid-slot + weight 0,
  `real_rows` deferred.
- **Note**: this is scaffolding, not the target behavior. Misses enter the graph in
  M3/M4 and never fall back to eager (architecture-v2 §6.2 invariant).

### M3 — CPU result return path into the graph (misses enter the graph) — ✅ DONE 2026-10-16

> Done via the device-epoch protocol: `epoch` (device) + one `ready` word per sync point + a
> single-buffered payload, with publication strictly FIFO per point. Each single-bank layer's graph
> carries a `wait_add_cpu` node, so the CPU experts' contribution is accumulated by the graph.
> **Gate: `--pcie-frac 0` + `NAIWA_NO_LOOKAHEAD=1` → 5/5 byte-identical to the frozen oracle.**
> With the lookahead on the *split* varies per step (pre-existing), so those checks compare the
> variant set, not a single reference. See [`worklogs/m3.md`](worklogs/m3.md).

- **Does**: replace `deferred`'s `cudaLaunchHostFunc` with `wait_copy_ptr`.
- **Depends on**: M2.
- **Judged by**: ① with misses present, greedy == eager; ② **misses are handled inside
  the graph** (an assertion/counter proves no eager path is taken); ③ zero sync points
  on the token thread (STEP_DBG dry bucket drops); ④ failure injection — a worker error
  still bumps seq, so the GPU never spins forever (`Drop` publishes).

### M4 — PCIe path into the graph (all three paths)

- **Does**: the 55% leg — a real H2D issued outside the graph by the token thread
  (workers make no CUDA calls), with the graph spinning for DMA completion.
- **Depends on**: M3.
- **Judged by**: ① hit + PCIe + CPU three-way sum == eager; ② **no recapture for any
  miss combination** (capture count is independent of miss/hit counts); ③ `frac` is only
  a bandwidth knob (X1/Q5).

### M5 — lookahead on a dedicated spin-then-park thread + verify the dispatch assumption

- **Does**: lookahead/prediction on 1–2 dedicated **spin-then-park** threads
  (architecture-v2 §6.3); the expert batch stays on rayon.
- **Depends on**: **implementation** needs only M2; **measurement** needs M4.
- **Judged by**: ① hit rate does not drop below the standing level (does not fall through
  the W105 ~78% knee); ② STEP_DBG shows expert dispatch is not on the critical path;
  ③ no A/B switches (architecture-v2 §6.4) — configuration recorded by commit.
- **Note**: if ② comes out backwards, evaluate it with a temporary code change; do not
  add a permanent fork.

### M6 — PLE (per Q2) + whole-step graph

- **Depends on**: the Q2 decision + F1 + F5.
- **Judged by**: whole step in **1 launch**; greedy == eager; the capture path has no
  host-dependent control flow, no temporary allocation, no H2D.

---

## Independent-verification matrix

| Step | New mechanism | Independent criterion (not performance) |
|---|---|---|
| M0 | mapped zero-arg spin primitives | visibility 1000/1000; replay 200/200, kernel demonstrably waited |
| M1 | static layout + descriptor kernel | eager token-identical; no new allocation / addresses fixed |
| M2 | per-layer capture | launches 60→1; token-identical to eager |
| M3 | CPU return path in graph | misses handled in-graph (counter); dry bucket drops |
| M4 | PCIe path in graph | three-way sum == eager; capture count independent of misses |
| M5 | dedicated lookahead thread | hit rate holds the knee; dispatch not on critical path |
| M6 | PLE + whole-step graph | 1 launch; token-identical to eager |

---

## Open

- **Q2 (PLE strategy)** — the only prerequisite of M6. Decide within M0–M2 by reading
  `ple.rs`'s n-gram hash / prefetch path: can the per-layer conv history be reproduced
  on the host? (~half a day.) See architecture-v2 §3.3.4.
- **PCIe-path nondeterminism — RESOLVED (benign).** Characterized 2026-10-16: it is float
  summation order, not a data race (a 64-slot ring does not stabilize it; `NAIWA_NO_LOOKAHEAD=1`
  gives 10/10 identical). The advisory lookahead's timing changes the resident/miss split, hence
  the `atomicAdd` term set, so near-ties flip. **M3/M4 are numerically safe to build on it**; but
  bitwise engine-level verification must use `--pcie-frac 0`, and M3/M4 checks are distributional
  otherwise. See [`worklogs/m1.md`](worklogs/m1.md).
