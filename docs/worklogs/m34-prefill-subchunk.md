# M34 — the prompt path prefetches a fraction of a layer, at a configurable depth

**Status:** proposed; design only, nothing built (2026-11-05). This file is the design and the plan; sections are
added as the steps land, the way M33 §1-§20 were.
**Parent:** M33 §20.13 (the prompt's wall clock is `chunks x 2.2 s`, and the per-layer decomposition that says why),
M18 (a second buffer buys 16-19% of overlap), M23 (the prompt law: 1.15 s per chunk + 1.67 ms per token), and
strata's `stream_all` ring read out in M33 §20.12.
**One mechanism:** the unit of the prompt path's prefetch becomes a **fraction of a layer's expert set** — 1, 1/2, 1/4
or 1/8 of the layer, a knob — with a **knob for how many sub-chunks may be in flight ahead of the consumer**, instead
of one whole layer per buffer with a depth of exactly one.

---

## 1. The measurement this starts from

M33 §20.13 attributed a 2 896-token prompt's 8 005 ms first token completely, by aggregating `NAIWA_STREAM_DBG` by
step:

| step | rows | layers | sum of `gap` | `copy` | `compute` |
|---|---:|---:|---:|---:|---:|
| 0 | 1 792 | 48 | 2 961 ms | 154 ms | 153 ms |
| 1 | 768 | 48 | 2 428 ms | 3.7 ms | 2.7 ms |
| 2 | 352 | 48 | 2 175 ms | 4.0 ms | 2.9 ms |

Two facts matter here:

1. the per-layer wall time is **62 / 51 / 45 ms for 1 792 / 768 / 352 rows** — it does not scale with the row count,
   so it is dominated by the layer's expert stream (≈0.5 GiB at 11.1 GB/s = 45 ms), not by the tokens;
2. the `copy` the log accounts for is 162 ms for the whole request, i.e. **the streaming cost is not showing up as
   measured copy time at all** — it shows up as *gap*, the time between one layer's entry and the next.

Point 2 is the reason this milestone exists. Today one layer's expert set is uploaded as a single unit and the layer
waits for the whole unit, so what the log can report as "copy" is only the part that is not covered by anything else.
The rest of the 45 ms is the copy engine working while the GPU has nothing to do for the layer, or the GPU working
while the copy engine is idle — the two alternate at layer granularity instead of interleaving.

For scale, the same comparison against strata on one cold 3 433-token prompt (M33 §20.13):

| | per chunk | effective rate of the expert stream |
|---|---:|---:|
| naiwa, 2 whole-layer buffers | 2.05 s | 11.1 GB/s |
| strata, 199-slot ring | 1.87 s | 13.1 GB/s |

So on top of the *count* of chunks (which the loan fixed, §20.13) there is a ~15% per-chunk difference, and its shape
is "a stalled pipeline against a continuous one".

## 2. What the code does today

| piece | where | what it does |
|---|---|---|
| buffer set | `PrefillStream::bufs: [Mutex<LayerBuf>; 2]` | two whole-layer buffers, ping-ponged by `layer % stride` |
| buffer size | `layer_bytes()` = the largest registered layer | one layer's three projections at once, ~650 MiB here |
| upload | `load(layer, stride)` → `copy.upload_async(&released, &parts, &ready)` | **one** async copy per projection of the whole layer, then **one** event (`buf.ready`) |
| wait | `forward(layer, …)` | waits on that single event, then runs the full-set MoE |
| depth | `Plan.stride` ∈ {1, 2} | 2 = double buffer: upload layer L+1 while layer L computes |
| VRAM | `plan_with_lend` prices `stride * layer_bytes` | so the buffers compete directly with the chunk |
| borrow | `borrow_buffers` (M33) | the buffers are ranges of the cache's slab when the loan is on |

Two consequences follow directly from that table:

- **the wait is all-or-nothing per layer.** The MoE for layer L cannot start until the *last* expert of L has landed.
  With a 650 MiB unit at 11 GB/s that is a ~45 ms window in which the GPU has nothing from this layer to do;
- **the depth is one layer.** `stride` cannot express "three sub-chunks ahead", because there is only one unit size to
  be ahead by.

## 3. Why splitting the unit and adding depth should help

The mechanism is a **pipeline granularity** one, not a bandwidth one — the bytes are identical either way.

- **Start earlier.** With the unit at 1/g of a layer, the first MoE work for layer L can start after `layer_bytes/g`
  has arrived instead of after all of it. At g=4 that is ~11 ms instead of ~45 ms of dead time at every layer
  boundary, and there are 48 boundaries per chunk.
- **Keep the copy engine fed.** Once the consumer is behind by a sub-chunk, the queue always has `depth` outstanding
  requests, so the copy engine sees a continuous stream across layer boundaries instead of a burst pattern with a
  stall in the middle. This is the effect strata gets from streaming all layers through one ring: their own note is
  that the copy engine "keeps working through the attention halves instead of waiting for each layer's routing".
- **Depth without a bigger budget.** Sub-chunking decouples the unit from the depth: today depth 2 costs 2 whole
  layers; at g=4, depth 4 costs one whole layer's worth of bytes and buys four units in flight. The whole point of the
  knobs is to find the `(g, depth)` pair that maximises overlap **at the VRAM the current plan already spends**.

What this cannot buy: the copy is 45 ms a layer and the compute is ~9 ms, so even perfect overlap removes only
`min(copy, compute)` per layer — about 0.43 s of a 2.2 s chunk. The rest of the hoped-for 0.18 s/chunk is the rate
difference (a continuous queue against a bursty one), which is a hypothesis this milestone has to measure, not assume.
**S0 below prices it before anything is built.**

## 4. The design

### 4.1 Knobs

Two new flags, because the repo's rule is that real knobs are flags and environment variables are probes:

```
--prefill-split <1|2|4|8>     experts per upload = num_experts / split   (default 1: today's behaviour)
--prefill-depth <N>           uploads allowed in flight ahead of the consumer (default 2: today's stride)
```

`split = 1, depth = 2` must be **exactly today's behaviour**, and it is what the oracle and the default gates run on.

The split is along the expert axis, not the byte axis, because that is the axis the consumer can use: a layer's expert
stacks are `[num_experts, …]` and the MoE gathers by expert id, so "the first half of the experts" is a slice the
kernels already understand. It also keeps every sub-chunk's three projections aligned the same way the whole-layer
upload is today, so the layout in `SlotMemory` / the borrowed ranges is unchanged in shape.

### 4.2 Buffers and the ring

`bufs: [LayerBuf; 2]` becomes a ring of `depth` sub-chunk buffers, `layer_bytes / split` bytes each:

| | today | M34 |
|---|---|---|
| buffers | 2 x 650 MiB | `depth` x `650 MiB / split` |
| in flight | 1 layer ahead | `depth` sub-chunks ahead |
| consumer waits on | one event per layer | one event per sub-chunk |
| VRAM at the current plan | 2 layers' worth | `depth/split` layers' worth |

The plan (`plan_with_lend`) prices the ring as `depth * ceil(layer_bytes / split)` instead of `stride * layer_bytes`,
so at `(4, 4)` the prompt path costs *half* of today and the freed bytes can go to the chunk — which is the direction
§20.13 says pays (chunk count is the wall clock). The planner keeps the same guard rails: half-by-half degradation,
and refusal rather than silently falling back to the CPU.

Question the design has to answer before S3: whether the ring should be **per layer** (each layer gets its own
`depth` sub-chunks, released when the layer ends — simple, no cross-layer indexing) or **model-wide** (strata's
choice; one ring shared by all layers, which is what makes a deep ring affordable but needs an id -> slot map). The
plan is to start per-layer, because it reuses `borrow_buffers`' layout and every layer's expert set already has a
home, and to treat the model-wide ring as the follow-on if the per-layer ring leaves the copy engine idle at the
layer boundary anyway.

### 4.3 Consumption: the MoE over a sub-range

A sub-chunk holds experts `[a, a + layer/split)`. Two ways to consume it, in order of risk:

- **(a) Slice the stacks and mask the ids.** `stack.narrow(0, a, n)` is a contiguous view, so `MoEExperts` can be built
  over the sub-range with no kernel change; each token's `topk_ids` are remapped into the local range, and the entries
  that fall outside it get weight 0 for this launch. The layer's output is the sum of the sub-launches' outputs.
  Cost: one extra combine per sub-chunk ([tokens, hidden] accumulation) and `split` masked launches instead of one.
- **(b) One launch, sub-chunk readiness inside it.** Not possible without a kernel change: ggml's grouped GEMM takes
  the full id set and reads whatever the views point at, so the data for *all* the ids has to be there. This is why
  (a) is the plan.

Correctness gate for (a): the sum of the masked sub-launches must equal the whole-layer launch **exactly** when
`split` is applied to a prompt that already fits, because the arithmetic per token is unchanged (the same experts, the
same order inside each expert) — the only new rounding is the combine's accumulation order. Measured at the layer
boundary with a relative-error probe, and gated as: `|split - whole| <= 1e-5` relative, plus the text-prefix rule from
M33 §20.8 for anything that touches the shipped path.

### 4.4 What this deliberately does not do

It does not adopt strata's 199-slot ring yet, even though that is where the design ends up. The ring's advantages over
this milestone are (i) it spans *several layers*, so the copy engine never sees a layer boundary at all, and (ii) its
schedule is routing-independent, so the copy engine never waits for a router. Both of those need an id -> slot map and
a schedule that streams experts that the current layer may not use, which is a much larger change to the staging layer
than a unit size and a depth. M34 is the same idea one axis at a time: **finer unit and deeper queue, still per layer,
still routing-driven.** If S4's numbers show the layer boundary is still where the copy engine idles, that is the
argument for the ring, and it will be recorded as such rather than assumed.

## 5. Plan

| # | step | what it settles | how it is checked |
|---|---|---|---|
| **S0** | **price the overlap before building it**: run the same prompt with `NAIWA_STREAM_SYNC=1` (blocking uploads, every overlap removed) against the default, on one 2 896-token prompt | the ceiling on what any overlap work can win, per chunk and per layer, from the existing code | `TTFT` and the per-layer `gap`/`copy` split, both arms, same session |
| S1 | per-sub-chunk completion events **inside one whole-layer buffer** — split the upload into `split` copies with `split` events, keep the buffer layout and the 2-buffer ring unchanged | whether finer events alone move the per-layer gap; no VRAM or plan change, so it is a pure overlap A/B | `gap` per layer at split 1/2/4, `copy` event count, TTFT |
| S2 | the MoE over a sub-range (4.3a) behind `--prefill-split`, default 1 | correctness and the cost of the masked launches and the extra combine | layer output vs the whole-layer launch (relative error), oracle byte-identical at split 1, `cargo test` |
| S3 | the ring of `depth` sub-chunk buffers + the planner pricing (4.2) + `--prefill-depth` | the actual mechanism, and whether the plan trades the freed bytes for a bigger chunk | planner logs (chunk, depth, split), VRAM peak (`NAIWA_PREFILL_PEAK=1`) vs the estimate |
| S4 | the matrix `split ∈ {1,2,4,8} x depth ∈ {2,3,4,6}` on two long prompts and one short | the default, and whether the layer boundary is still a stall (the ring's justification) | TTFT, per-layer `gap`/`copy`, nsys kernel total, chat `short` TPOT/TTFT |
| S5 | defaults, docs, and the `short`-corpus chat round to show the default pair does not cost the decode path | — | 3 rounds, 0 failures |

## 6. Gates

The standing ones, unchanged: `frac 0` + `--no-lookahead` byte-identical to `harness/m1/reference.oracle.txt` at the
**default** `(split 1, depth 2)`; `cargo test` 56 passed; the chat harness on `short` for 3 rounds with 0 failures.
Added by this milestone:

- **the split path is never the oracle's path** — a change in the combine's accumulation order is an arithmetic change,
  which by M33 §20.8's rule cannot be byte-exact, so the oracle keeps running `split 1` and the split path is gated on
  the relative-error probe plus a text prefix;
- **the planner must refuse, not degrade silently**: if `(split, depth)` does not fit, the plan halves the depth and
  then the split, and logs which;
- **no regression on the loan**: `NAIWA_PREFILL_BORROW` stays on by default (§20.13), so the ring has to work as
  ranges of the cache slab, not only as its own allocation.

## 7. Risks and open questions

1. **The extra combine.** `split` launches per layer each accumulate `[tokens, hidden]`. At 2 560 tokens that is
   ~13 MB a launch and ~0.1 ms each, i.e. ~0.4 ms a layer — small against 45 ms, but it is per layer and it is on the
   critical path, so S2 measures it rather than estimating it.
2. **Masked launches may not be free.** `split` launches iterate the token rows `split` times (most of them masked to
   weight 0). If the common `ggml_mul_mat_id` path costs per row regardless of the mask, the MoE's compute could
   double at split 2. This is the single largest risk in the design and S2 exists to find it.
3. **Events per copy.** `split` events a layer instead of one: 48 x split host-side recordings per chunk. Cheap in
   principle, but the log already reports `dma_wait` in ns, so the instrumentation will show if it is not.
4. **Fragmentation.** `split` separate borrowed ranges per layer against one contiguous range today; the pool is carved
   into ranges already (M33 §3), so this is a matter of the carve, not new machinery — but it has to be checked
   against the "half of 1 536 MiB" degradation path.
5. **Whether the depth is even usable.** If the copy engine saturates at depth 2 the way `staging_ring_slots()` found
   for decode (16 and 64 measured identical), the depth knob buys nothing. That is a possible and useful outcome: it
   would say the remaining gap is the copy rate itself, not the pipeline.

## 8. Expected size, stated as a hypothesis

| what | expected | where it comes from |
|---|---|---|
| start-earlier at every layer boundary | **-0.4 s per chunk** | `min(copy, compute)` per layer x 48 = 9 ms x 48 |
| a continuous copy queue | **-0.2 s per chunk** | 11.1 -> 13.1 GB/s is the gap to strata |
| freed VRAM at `(4, 4)` -> a bigger chunk | unpredictable | §20.13: the chunk count is the wall clock, so this could dominate |

So the target is a 2 896-token prompt's first token at **3.7 s -> 3.0-3.2 s** with the same 2.5 GB cache and the loan
on, and the honest possibility that S0 shows the ceiling is lower than that, in which case this file's conclusion is
the measurement and the ring moves up instead.
