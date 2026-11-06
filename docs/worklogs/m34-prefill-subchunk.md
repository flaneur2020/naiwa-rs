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

---

## 9. S0 — the overlap is already there, and it is worth 8%

Before building anything, `NAIWA_STREAM_SYNC=1` (blocking uploads, every overlap removed) against the default, same
prompt, same session, `NAIWA_STREAM_DBG=1` on both:

| arm | TTFT | per-layer `gap` | sum `gap` | sum `copy` | sum `compute` |
|---|---:|---:|---:|---:|---:|
| async (default, 2 buffers) | 4 099 ms | **69.6 ms** | 3 273 ms | 3 499 ms | 228 ms |
| sync (no overlap at all) | 3 863 ms | **75.9 ms** | 3 570 ms | 5 434 ms | 30 ms |

The two arms differ by **8% per layer** (69.6 against 75.9), i.e. the async path already hides nearly all of the 6 ms a
layer that is not the copy. **The premise this milestone was written on - that the compute is exposed at layer
granularity and finer units would hide it - is false.** The per-layer wall time *is* the copy, and the copy is already
continuous: there is nothing left for a finer unit to overlap.

Two consequences:

1. the ceiling for the split half of this milestone is that same ~8%, not the 0.4 s a chunk §8 estimated;
2. the term that is actually wrong is the **rate**: this session's copies ran at **7.0-7.3 GB/s** (24 GiB in 3.3-3.5 s),
   against 11.1 GB/s when §20.13 measured and strata's 13.1. The gap to strata is in the copy, not in the pipeline.

## 10. S1 (the depth half) — built, and it says the same thing

`--prefill-stream-depth 1..4` is in the tree: the plan grants up to that many layer buffers
(`plan_with_lend_depth`), and `forward` issues `depth - 1` layers of upload ahead of the one it waits for. Depth 1 is
the arm without prefetch. Same prompt, same session:

| depth | plan granted | TTFT | per-layer |
|---:|---|---:|---:|
| 1 | 1 buffer | 4 353 ms | **77.9 ms** |
| 2 (default) | 2 buffers | 3 817 ms | **69.8 ms** |
| 3 | 2 buffers | 3 722 ms | 65.0 ms |
| 4 | 2 buffers | 3 722 ms | 65.0 ms |

- **depth 1 -> 2 is a real 12% per layer** - the double buffer is the whole of the available overlap, and 12% is the
  same number M18 measured for it;
- **depth 3 and 4 are not granted**: the loan is 1 526 MiB and a layer range is ~650 MiB, so the pool holds two. The
  step from 2 to 3 in the table is session drift, not mechanism - depth 3 and depth 4 produce identical numbers, and
  both log `2 layer buffer(s) loaned`.

That is the design's own prediction from §4.2 arriving as a measurement: **the deeper queue needs a smaller unit, so
the split has to exist before the depth can be spent.** At split 2 a unit is 325 MiB and four of them fit the same
1 526 MiB pool, which is exactly the `(split 4, depth 4) = one layer's worth of bytes` row in §8.

## 11. What S2 has to be, now that it is the only half left

The consumer has to be able to start on a sub-range, and the survey of the two sides says that is expressible without
touching a kernel:

- **candle has no `QTensor::narrow`**, so a sub-range tensor has to be built by hand: `QCudaStorage` wraps a byte
  tensor, so narrowing *that* by `a * bytes_per_expert` and rewrapping it with `QTensor::new(.., [n, ..])` gives a
  valid quantized view of experts `[a, a+n)` - and the buffers are ours (`SlotMemory`, owned or borrowed), so the
  bytes are addressable;
- **the MoE kernels already take per-row expert ids** (`indexed_moe_forward`, `ggml_mul_mat_id`), so the sub-launch
  is `MoEExperts::from_qtensors(gate_sub, up_sub, down_sub, n, top_k, ..)` with the ids remapped into `0..n` and the
  weights of the experts outside the range masked to 0, then the per-sub-chunk outputs summed.

The cost side, which S2 must measure rather than assume: `split` launches a layer instead of one, each with its own
combine of `[tokens, hidden]`, and most of the rows in each launch are masked to weight 0 - if `ggml_mul_mat_id` pays
per row regardless of the mask, the MoE's compute doubles at split 2. Against a 6 ms a layer budget that is the whole
of what is left to win, so it may well be a wash; the honest expectation after S0 is that this milestone's result is
**the measurement that the prompt path is copy-bound and rate-limited**, and that the next lever is the copy itself
(one merge window instead of three per layer, `io_submit`-style queueing, or a larger loan window so the depth can
grow without splitting).

## 12. The knob as built

```
--prefill-stream-depth <1-4>   (default 2, requires --prefill-stream)
```

`plan_with_lend_depth(free, layer_bytes, ceiling, borrowing, kv_reserve, lend_bytes, max_depth)` takes the ceiling;
the borrow branch grants `min(lend_bytes / layer_bytes, max_depth)` ranges, the owning branch walks `max_depth..=2`
down and takes the first that fits with the decode reserve. `(depth 1)` is the no-prefetch arm and `(depth 2)` is
byte-for-byte the behaviour before this milestone, which is why the oracle gate is unaffected - and it is: `frac 0` +
`--no-lookahead` is still byte-identical, and `cargo test` is 56 passed.

---

## 13. Where the copy rate actually goes: per-layer reallocation, then the pages

S0/S1 said the prompt path is copy-bound and that the copy runs at 7-9.8 GB/s where the link idles at 13.4. This is
the investigation of that gap. It is **not** the copy stream and **not** the number of parts - the repo's own
`h2d_` probes (in `src/host_memory.rs`) answer both on this machine, today:

| probe | rate |
|---|---:|
| 512 MiB, pageable source | 8.0 GB/s |
| 512 MiB, registered (`cuMemHostRegister`) | 13.4 GB/s |
| 511 MiB in 3 parts, registered, **4 KiB pages** | **5.5 GB/s** |
| 511 MiB in 3 parts, registered, **+ `MADV_HUGEPAGE`/`MADV_COLLAPSE`** | **13.4 GB/s** |
| the same 3 parts on the *default* stream | 13.3 GB/s |
| the same 3 parts on a **second stream** (the way `CopyStream` builds one) | 13.3 GB/s |
| a second pass over an already-written destination | 10.7 GB/s |

So: the copy stream costs nothing, one copy costs the same as three, and **the host page size of the source is worth
2.4x** - which is the mechanism strata's `pinned.cu` calls out ("large pages first, then the largest alignment the OS
will give us"). Our process does have 89% of its anonymous memory on 2 MiB pages
(`smaps_rollup: AnonHugePages 23.6 GiB of Anonymous 26.4 GiB`), so this is not the main term any more - worth about a
tenth of the time, not a doubling.

**The main term is per-layer reallocation.** Reading the per-layer rates out of one chunk's debug log, the pattern is
a strict function of whether that layer had to be reallocated:

| layer | copy | rate | what happened |
|---|---:|---:|---|
| 1 | **415 ms** | **0.9 GB/s** | the *second buffer's first* allocation (368 MiB into a fresh block) |
| 3, 7, 11, 15, ... 43 (every 4th) | 80-90 ms | **5.8-6.8 GB/s** | shape change -> reallocation |
| 4, 5, 6, 9, ... 46 | 54-63 ms | **9.2-9.8 GB/s** | the block is reused as-is |

A chunk pays **35 reallocations** of 48 layers, because this model has seven layer shapes (mixed quant) and two buffers
alternate between them. The reallocation call itself is cheap (15-82 us, ~2 ms a chunk) - what it costs is the copy
into **freshly allocated device memory**, which runs 35% slower than into a block that has already been written. The
first-touch case is the extreme: 415 ms for 368 MiB.

### So the ordering of the levers changes

| lever | expected | status |
|---|---|---|
| **no reallocation on a shape change**: the buffer is a borrowed range sized by the *largest* layer's span, so a smaller layer needs new *views*, not new memory - carve the range once and re-wrap the projection tensors with the layer's shape | **+35% on two thirds of the layers**, ~0.5 s a chunk, and it removes the 415 ms outlier | not built; needs the same "quantized view over our own bytes" helper S2 needs (§11) |
| complete 2 MiB coverage on the stacks (hugetlb first, then `MADV_HUGEPAGE`, and *verify* rather than best-effort) | up to 10%, since 89% of the pages are already large | not built |
| depth / split (S1, S2) | 0-12%, measured | depth built, split not |
| merging the 3 copies into fewer | **0**: 3 parts and 1 part both reach 13.4 | measured, closed |

### Calibration, which matters for what to expect

The expert set is streamed once per chunk whatever the chunk's token count: 48 layers x 531 MiB = 24 GiB. At the link's
13.4 GB/s that is **2.0 s a chunk, at 40 ms a layer** - a floor no pipeline work moves, because those bytes have to
cross PCIe. With the loan's two chunks for a 3 433-token prompt that floor is ~4.0 s, against strata's measured
3 741 ms (which pays the same bytes) and our 4 791 ms.

Which is the whole story in one line: **strata is at the byte floor and we are 28% above it, and the 28% is the
reallocation penalty on two thirds of the layers, not the pipeline's shape.**

---

## 14. §13 is wrong, and the correction changes the answer: the copies are already at the link's ceiling

§13 read the per-layer rates out of `NAIWA_STREAM_DBG` and concluded that reallocating on a shape change cost 35% of
the copy rate. **That conclusion does not survive a properly measured copy.** Two things were wrong with it:

1. the per-layer `copy=` in that log is **queue-to-consumer latency** (`buf.queued.elapsed()` read just before the
   compute), not the copy's duration, so it folds in everything the layer did in between;
2. the correlation was weak anyway: of the 15 slow layers 12 had reallocated, but so had 23 of the 33 fast ones - and
   the slow ones are exactly the **QSA layers** (index `3 mod 4`, `full_attention_interval = 4`), i.e. the ones with
   the expensive attention in front of the MoE. It was the metric measuring attention.

So the copy stream got its own timing pair, on the **per-buffer** `CopyDone` rather than the stream (the pipeline
queues the next layer before the consumer reads this one, so a shared pair is overwritten), behind
`NAIWA_COPY_TIME=1`. The number it reports is `cuEventElapsedTime` between an event recorded just before the batch and
one just after, both on the copy stream. Same prompt, same session:

| layer | bytes | the copy's own duration | rate | queue-to-consumer |
|---|---:|---:|---:|---:|
| 0 | 531 MiB | 41.6 ms | **13.4 GB/s** | 41.6 ms |
| 1 | 368 MiB | 28.9 ms | **13.4 GB/s** | 138.2 ms |
| 2 | 481 MiB | 37.7 ms | **13.4 GB/s** | 64.0 ms |
| 3 (QSA) | 481 MiB | 37.7 ms | **13.4 GB/s** | 80.1 ms |
| 4-6 | 481 MiB | 37.7 ms | **13.4 GB/s** | 54.3-54.8 ms |
| 7 (QSA) | 531 MiB | 41.6 ms | **13.4 GB/s** | 84.4 ms |

**Every layer copies at 13.4 GB/s - the same number the `h2d_` probes reach with no compute and no pipeline, and the
same number strata's streaming reads.** There is no copy-rate left to win: 48 x 531 MiB at 13.4 GB/s is 2.0 s a chunk
and every layer is already there.

**What is left is a start delay.** The copy's own duration is 37.7 ms, but the consumer sees 54-84 ms, i.e. **17-47 ms
a layer of the copy stream doing nothing before the copies begin**, which is ~0.8-2 s a chunk - larger than anything
§8 or §13 proposed to chase. Prime suspect: the ordering the pipeline inserts on the copy stream (`released` fence per
batch, and `after_device_work` on the 35 layers that reallocate), each of which pins the copy stream behind the device
stream's *current tail* - which by construction is the current layer's own kernels.

That is the next measurement, and it is cheap with the probe that now exists: record an extra event at enqueue time and
report `delay = (first copy starts) - (batch queued)`. If the delay is the fence, `NAIWA_STREAM_SYNC=1`'s arm should
show it too, and removing the per-layer `after_device_work` (by not reallocating on a shape change, which §13 proposed
for the wrong reason and is still worth doing for this one) should remove most of it.

---

## 15. Depth beyond two: granted, and it breaks the request. Which is the argument for splitting

With §14 establishing that every layer's copy already runs at 13.4 GB/s, the only thing left to improve is how much of
it is *hidden*, so the depth knob was tested properly. The borrow branch grants `lend_bytes / layer_bytes` ranges, so
depth needs the lending window to grow (`NAIWA_SLAB_MIB`, which is carved out of the expert-cache budget). Same prompt,
one cold request each:

| slab | `--prefill-stream-depth` | granted | request | per-layer |
|---:|---:|---|---|---:|
| 1 536 MiB | 2 | 2 ranges | ok | 70.2 ms |
| 2 048 MiB | 2 | 2 ranges | ok | **65.0 ms** |
| 2 048 MiB | 3 | **3 ranges** | **HTTP 500, twice of two** | 65.0 ms |
| 2 560 MiB | 4 | 3 ranges | **HTTP 500** | 65.0 ms |

Two separate findings, and neither is the one the knob was added for:

1. **The 70.2 -> 65.0 ms is the window, not the depth.** 2 048 MiB with depth *2* gives 65.0 as well. A bigger window
   means the pool is carved into fewer, larger contiguous ranges (117 of them) and the chunk's three projections land
   in one, which is worth 5 ms a layer (~0.25 s a chunk) - for cache budget, since the window is carved out of it.
2. **Depth 3 completes the chunk and then the request dies.** The prefill itself is clean: 48 layers, `stride 3`, the
   copy at 41.6 ms own, `borrowed 1 950 MiB of the pool in 3 range(s), 1051 slots emptied`, the loan given back with
   `1051 of 1051 displaced experts queued for refill`, and then 48 verify-step layers run and the stream produces
   **zero decode tokens** (`prefill 579.20, decode 0.00` in the engine's own summary). No CUDA error, no panic, twice
   out of two. The one number that stands out is the wipe: **1 051 of 1 145 slots emptied (92%)** against 719 at two
   ranges.

So the flag is **capped at 2 in the borrow branch** (`plan_with_lend_depth`, with the reason next to the clamp), which
is the behaviour that shipped plus a knob that can only make it shallower.

**And that is the strongest argument for the split half of this milestone that this file has produced.** The cost of
borrowing is not the bytes while the chunk runs, it is the wipe and the give-back afterwards, and that cost is
proportional to the *range* borrowed, not to what the chunk uses of it. Splitting the unit does not raise the bytes in
flight for a given VRAM (a layer still has to be whole to be consumed), but it does make each borrowed range 1/g of a
layer - so the same pool can hold g times as many units *and* the wipe, the refill and the give-back all shrink by g.
Depth 3 failed on exactly the term that splitting reduces.

---

## 16. Splitting the copy is free, so the whole price of the split is on the consumer

§14 showed every layer's copy already runs at 13.4 GB/s, so before designing the consumer side (S2) the obvious
question is whether a *smaller* copy unit costs throughput. If it does, the split is dead on arrival; if it does not,
the only thing S2 has to justify is the MoE's masked launches.

`NAIWA_COPY_SPLIT=g` answers it: the same bytes into the same destination, but each projection's copy is issued as `g`
commands of `1/g` of its length (4 KiB boundaries), with the same fence, the same single completion event and the same
call site. Same prompt, same session, the copy's own duration from the `NAIWA_COPY_TIME` pair, 48 layers each:

| `NAIWA_COPY_SPLIT` | median | max |
|---:|---:|---:|
| 1 (today) | **41.6 ms** | 51.0 ms |
| 2 | **41.6 ms** | 50.9 ms |
| 4 | **41.6 ms** | 50.9 ms |
| 8 | **41.7 ms** | 51.0 ms |

**Granularity is free**: the link reaches 13.4 GB/s whether a layer arrives as one copy or as twenty-four. So the split
has no bandwidth cost to pay, and the entire cost of S2 is the consumer side - the masked launches and the extra
combine - which is what S2 exists to measure.

Two consequences for the design:

1. the split can be introduced incrementally: the copy path is proven free, so the risk is all in the MoE, and the
   probe (`NAIWA_COPY_SPLIT`) stays in the tree as the control arm;
2. it also means **the depth-3 failure in §15 is not a granularity problem**: 92% of the cache was wiped and the
   request produced no decode tokens. That is the wipe and the give-back, and it has to be fixed on its own terms
   before a deeper queue can be measured at all - splitting the unit does not shrink the wipe, because the bytes a
   layer occupies while it is being consumed are the same however they are subdivided.

---

## 17. S2's consumer side is built - and blocked on one CUDA symbol lookup

The split needs the consumer to take a layer's experts a piece at a time, and the two sides of that are now in the
tree behind `--prefill-split g` (default 1 = the whole layer):

1. **A sub-range is a view, not a copy.** `SlotMemory::sub_view` wraps `QCudaStorage::borrow_from` over the slot's own
   storage - the same call the loan uses to carve ranges out of the pool - at `a * bytes_per_expert` for
   `n * cols` elements. `SlotMemory` had to keep its `QCudaStorage` for that, because `QTensor` does not expose one.
2. **The consumer runs `g` masked launches.** `mask_range` zeroes the weights of the experts outside the sub-range and
   rebases the ids into `0..n` (in i64, `relu` before the mask, so a rebase cannot wrap), and the sub-launch outputs are
   summed. `NAIWA_SPLIT_CHECK=1` runs the whole-layer launch *as well* and prints `worst |diff|` against it, which is the
   numerical gate this path will be held to.

**And it does not run yet**: the masked launch fails with `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not
found")` from candle's `indexed_moe_forward`, which is a kernel-module lookup rather than a shape check - so the
sub-view is built and passed, and the MoE's CUDA entry cannot find the symbol for it. The cause is not yet identified.

What is in the tree is therefore the **safe** form of it: `dispatch` falls back to the whole-layer launch on any error
from the split path, once, with the reason in the log
(`prefill split 2: DriverError(CUDA_ERROR_NOT_FOUND, ...); the chunk runs its experts as whole layers from here`), and
the arm measures exactly what the whole-layer arm measures (TTFT 3 829 ms and 69.7 ms a layer against split 1's
3 890 / 70.2 - it is the same code path, as the warning says).

Gates on the shipped defaults: oracle byte-identical, `cargo test` 56 passed.

**Next**: find what `indexed_moe_forward` looks up. The candidates, in order of likelihood, are (a) a kernel name built
from the stack's *first* dim, (b) a module compiled lazily per dtype where the borrowed offset breaks the code path's
expectation, (c) `MoEExperts::from_qtensors`' backend choice (`Fused` vs `Cutlass` vs `Fast`) switching on the expert
count. Each is a grep and a print, and the numeric gate (`NAIWA_SPLIT_CHECK`) is already in place to validate the first
one that works.

---

## 18. S2 works and closes its own route: the split costs more than the overlap can win

Three defects stood between §17's plumbing and a running split, and they were all found by making the fallback print
its reason instead of swallowing it:

1. **Integer tensor kernels.** `mask_range` first did the rebase in i64 with a `relu`, and candle compiles its unary
   kernels per dtype, so an i64 `relu` asks the CUDA module for a symbol it does not have -
   `CUDA_ERROR_NOT_FOUND: "named symbol not found"`, which is what §17 reported. Everything is F32 now: the ids are
   < 256 so they are exact, the mask is a 0/1 multiply, and an out-of-range id becomes `(id - a) * 0 = 0`, a valid
   index with zero weight.
2. **`MATRIX_ROW_PADDING` on the last sub-range.** `QCudaStorage::borrow_from` validates `offset + padded`, so a view of
   the *second* half of a stack needs 512 elements more than the stack holds and was refused by 196 bytes
   (`borrow of 80 281 796 bytes at 80 281 600 does not fit in a 160 563 200-byte slab`). The slot is now carved with
   `MATRIX_ROW_PADDING` extra elements and the upload's length check is `<=` rather than `==`.
3. With those two, the split runs: `split 2 in 2 sub-ranges`.

**Numerical gate** (`NAIWA_SPLIT_CHECK=1`, which also runs the whole-layer launch and prints the difference):

| layer | worst abs | scale | relative |
|---|---:|---:|---:|
| 0 | 9.77e-4 | 5.31e-1 | 1.8e-3 |
| 1 | 9.77e-4 | 2.20e-1 | 4.4e-3 |
| 2 | 1.78e-2 | 4.28e0 | 4.2e-3 |
| 3 | 9.77e-4 | 1.53e-1 | 6.4e-3 |

The absolute steps are `2^-10`-sized and the relative error sits at bf16's own resolution (`2^-8` = 3.9e-3), which is
what the two paths can differ by: the outputs are combined per sub-range and added in f32, where the whole layer
combines them in one pass and the result is rounded to bf16 either way. **So the split is numerically right and cannot
be bit-exact** - the same rule M33 §20.8 wrote down.

**And it costs more than it can ever win.** Same prompt, same session, whole-layer upload, only the consumer split:

| `--prefill-split` | TTFT | per layer | request |
|---:|---:|---:|---|
| 1 (whole layer) | 3 712 ms | **64.8 ms** | ok |
| 2 | - | 79.7 ms | **HTTP 500** |
| 4 | 4 788 ms | **85.5 ms** | ok |

+23% at split 2 and +32% at split 4 per layer, from the masked launches (every row is processed in every sub-range, the
mask only zeroes its weight) plus `g` combines instead of one. Against §9's measured ceiling - **8%** of the layer is
non-copy work that a finer unit could hide - this is not a trade, it is a loss. **So the split route is closed by
measurement on this model**, and the flag stays at its default of 1 with the whole-layer fallback in place.

The remaining question the split was supposed to answer - whether a *smaller unit* makes the depth affordable - is
answered by §16 instead: the copy is free at any granularity, and the bytes a layer occupies while it is consumed do
not shrink by subdividing them, so depth still costs wipe. The honest state of M34 is that both of its halves are
measured and neither pays on this model: the overlap (§9) is already realised, and the unit (§18) costs more than it
buys. What is left standing from the milestone is the instrument set - `NAIWA_COPY_TIME` for a copy's own duration,
`NAIWA_COPY_SPLIT` for granularity, `NAIWA_SPLIT_CHECK` for the numeric gate - and the two negative results, which is
what §13/§14/§16/§18 together cost.

---

## 19. Where the prompt's time goes, ours against strata's, and the PLE question

Asked for the stage distribution and a specific suspicion about PLE. Both are answerable from today's measurements,
and the first finding is about the machine rather than the code.

### 19.1 The card is shared with a 6.6 GB GPU process, and that is the variance

```
NVIDIA GeForce RTX 4070, 12282 MiB, used 7285, free 4591
  2174282  C+G  ...\game\game\sp26\sp26-cod.exe   6597 MiB
```

A game holds **6.6 GiB of this 12 GiB card** for the whole session. Consequences that were being read as engine
properties:

- the **chunk plan** (which is sized from free VRAM) lands anywhere between 768 and 2 816 tokens, and TTFT is
  `chunks x ~2.1 s` (M33 §20.13) - so the same request is 3.7 s or 8.3 s depending on what else is on the card;
- the §20.13 comparison against strata (4 791 ms against 3 741) was made in a squeezed state; §19.3 below has the
  un-squeezed one;
- depth 3 (§15) asks for 1 950 MiB of pool - unavailable;
- and as this was being written the shipped configuration **stopped loading at all** (`CUDA_ERROR_OUT_OF_MEMORY`, even at
  a 2.0 GiB cache with 4 596 MiB free), so no further profiling was possible without freeing the card.

Every VRAM-sensitive number in this file should be read with that in mind. It is also the honest answer to "why is our
prefill unstable": not the host, not the kernel, but a plan that re-derives itself from whatever VRAM is left.

### 19.2 Our distribution, one 2 896-token prompt, chunk 2 560, loan on

Measured with `NAIWA_STREAM_DBG` (per layer) and `NAIWA_COPY_TIME` (the copy's own duration), 48 layers:

| stage | per layer | per chunk | share of the chunk |
|---|---:|---:|---:|
| expert stream (copy, own duration) | 41.6 ms | **1 997 ms** | **65%** |
| everything else the layer does | ~23 ms | ~1 104 ms | 35% |
| of which: quant GEMM (nsys, scaled) | | ~740 ms | 24% |
| GDN recurrence (nsys) | | ~254 ms | 8% |
| prompt attention (nsys) | | ~152 ms | 5% |
| bf16 GEMM / hyper-connection / routing | | ~240 ms | 8% |
| **PLE** (host gather + upload, `PLE_DBG`) | | **~94 ms** | **3%** |
| chunk wall (`sum gap`) | 64.8 ms | 3 048 ms | 100% |
| **overlap achieved** | | **~19 ms/layer (45%)** | |

The copy and the layer's work are therefore **45% overlapped, not hidden**: 64.8 ms a layer against 41.6 of copy and
~42 of kernels means the pipeline alternates for about half of each layer. That - not the copy rate, which is at the
link's ceiling (§14) - is the structural difference to a ring that never lets the engine idle.

### 19.3 PLE is 3% of the chunk, not the gap

The suspicion is worth answering with numbers, because PLE is the one stage whose *source* is not contiguous.

- the host gather is **94 ms per prefill step** (14 344 rows in one call, `PLE_DBG`; two calls' worth of statistics in
  §20.12's 354 ms over 17 calls including decode);
- its traffic is tiny: 14 344 rows x a 160-value IQ4_NL row (~90 B) = **1.3 MB** per step, against 24 GiB of experts -
  five orders of magnitude smaller;
- and it is *serial* (a host memcpy loop plus an upload), so what it can buy is that 94 ms, i.e. **3% of the chunk**.

So even making PLE free would move a 3.05 s chunk by 0.09 s. It cannot be the term that separates us from strata, and
the table above says the term that is: the 65% of the chunk that is the expert stream, and the 55% of it that is not
overlapped.

### 19.4 Against strata, at the same chunk size, we are at parity

| | tokens | prefill wall | tok/s |
|---|---:|---:|---:|
| naiwa, chunk 2 560 (loan, unsqueezed) | 2 896 | 3 048 ms | **950** |
| naiwa, chunk 768-2 304 (VRAM squeezed) | 3 433 | 4 791 ms | 717 |
| strata (`serve`, chunk 2 816) | 3 433 | 3 741 ms | **918** |

**950 against 918 tok/s**: with the chunk the loan was designed to buy, the prefill rate is at parity, and the 28% in
§20.13 was the squeezed plan rather than the engine. What is left is not a rate but a *robustness* difference - strata's
plan does not collapse when the card is busy, ours does - and the 55% of the copy that is not yet overlapped, which is
what a ring would recover if the VRAM for it existed.

---

## 20. Re-measured with the card freed: the picture that survives

§19.1 found a 6.6 GiB game holding the 12 GiB card. It was killed and everything VRAM-sensitive was re-measured. What
survives, and what does not:

| | with the game | card freed |
|---|---:|---:|
| free VRAM before load | 4 591 MiB | **11 241 MiB** |
| free VRAM the plan sees at warmup | 2 002 MiB | **2 001 MiB** |
| chunk the plan affords | 3 840 | 6 656 |
| `--expert-cache-gb 4.0` | OOM | **still OOM** |

So **6.6 GiB of headroom changes nothing about the chunk**, because ~9 GiB of the card is *our* own device footprint
(2.5 GB expert cache + 0.7 MTP drafter + ~3.8 GB of non-expert weights + KV/recurrent state + ~1 GB of context and
retained scratch). A trim of the caching allocator before the plan (`NAIWA_PLAN_TRIM`, added here) also changes nothing
- the bytes are genuinely in use, not hoarded. 2.5 GB is the ceiling for this configuration on this card, and §19.1's
"the game is the variance" was wrong: the game removes headroom from a card that had none to spare, but freeing it does
not give the plan any.

### 20.1 The distribution, corrected

Every arm of this re-test sent the same **single step of 2 880 rows** (the plan's chunk sizes - 1 792, 4 864 - bind the
*next* request, not the one that re-planned), so the numbers are directly comparable:

| | prompt A (2 896) | prompt B (2 912) |
|---|---:|---:|
| TTFT | 3 914 / 3 787 / 3 733 ms | 3 425 ms |
| layer loop (sum of the 47 `gap`s) | 3 047 / 3 052 ms | 3 133 ms |
| **outside the layer loop** | **~740-870 ms** | **292 ms** |
| per layer | 64.8 ms | 66.7 ms |
| of which the copy's own duration | **41.6 ms (13.4 GB/s)** | 41.6 ms |
| PLE (host gather, `PLE_DBG`) | 35-41 ms a call | |

Two things this adds to §19: the layer loop is 3.05 s of a 3.73-3.91 s TTFT, so **0.3 s (a warm request) to 0.9 s (the
first request after load) is outside it** - tokenisation, the plan re-check, first-call module loads - and that is larger
than PLE's 3%. The layer loop itself is unchanged: 65 ms a layer, 41.6 of it the copy at the link's ceiling.

### 20.2 Depth 3 is closed for good, and it was never VRAM

§15's three-range arm produced no decode tokens (HTTP 500) on a squeezed card, which left open the possibility that it
was simply out of memory. On the freed card it does the same thing:

| ranges | granted | request | per layer | queue-to-consumer |
|---:|---|---|---:|---:|
| 2 | yes | ok | **64.9 ms** | 58.4 ms |
| 3 | yes (2 048 MiB pool) | **HTTP 500** | **70.4 ms** | 125.9 ms |

So it is the loan's bookkeeping, not the card - and the third buffer is not even faster when it runs: the copy is still
41.6 ms and the queue-to-consumer latency *doubles*, because the extra range lengthens the wipe and the wait. The cap
that §15 put in the code is now behind `NAIWA_DEPTH_CAP` (default 2) so that a configuration known to break is not
reachable by accident, with this table beside it.

### 20.3 Where the prompt stands

| | tokens | TTFT | tok/s |
|---|---:|---:|---:|
| naiwa, single step | 2 896 | 3.73-3.91 s | **742-777** (from TTFT) |
| naiwa, single step, layer loop only | 2 896 | 3.05 s | 950 |
| strata (`serve`, chunk 2 816) | 3 433 | 3.74 s | 918 |

The layer loop is at 950 tok/s against strata's 918; the whole TTFT is 742-777 because of the 0.3-0.9 s outside it. PLE
is 3%, the copy is at the ceiling and 45% overlapped, and the depth/ring route is closed both by the wipe (§15, §20.2)
and by the split's cost (§18). What is left visible is the per-request overhead outside the layer loop, which nobody has
instrumented yet and which is bigger than PLE.

### 20.4 The oracle arm was OOMing on `--max-seqs`, not on the engine

Re-running the §20 gates on the freed card turned up a third thing that had been mis-read as an engine problem: the M1
oracle arm (`harness/m1/capture_oracle.sh`) OOM'd at load with 11.2 GiB of free VRAM, twice, and does not with
`--max-seqs 4` - byte-identical to the reference. It was reserving the default 32 sequence slots of recurrent state
(~112 MiB each, 3.5 GiB) beside a 2 GB expert cache on a 12 GB card. The harness now sets `MAX_SEQS=4` by default,
which is the number the shipped chat config already uses (`engines/naiwa.json` records the measurement behind it).

So of this session's three "the shipped path is broken" moments - the depth-3 HTTP 500 (§15), the OOM at 2.0 GB with the
card free, and the oracle arm's OOM - only the first is a defect in our code. The second was the card's owner taking
6.6 GiB, and the third was a harness default that the `--max-seqs 4` note in the chat config had already solved for the
serving path.

---

## 21. The 0.3-0.9 s outside the layer loop, attributed: 60-160 ms of arrival, and a 284 ms prefix-cache snapshot

It is **per request, not per layer** - `sum(gap)` is entry-to-entry over layers 1..47, so layer 0's own work, the last
layer's, and everything before and after the loop are all outside it. `NAIWA_REQ_TIME=1` stamps absolute wall clock at
layer 0 and at the last layer, which lines up with a client that stamps its own send and first token, so the request
splits into three segments that add up exactly:

| | 772 tokens | 3 143 tokens | slope |
|---|---:|---:|---:|
| **A** send -> layer 0 (HTTP, tokenise, schedule, chunk assembly) | 162 ms | 95 ms | none (fixed, noisy) |
| **B** layer 0 -> last layer entry (the layer loop) | 2 149 ms | 3 364 ms | 0.51 ms/token |
| **C** last layer entry -> first token | 263 ms | 396 ms | 0.056 ms/token |
| TTFT | 2 574 ms | 3 855 ms | |

A + B + C = TTFT exactly (95 + 3 364 + 396 = 3 855). Layer 0's own work is inside B but outside `sum(gap)`: 3 364 -
3 221 = 143 ms.

**The token-scaled part of C is the prefix-cache snapshot.** The chat config runs `--prefix-cache-n 1` with
`MISTRALRS_ALIGN_CHECKPOINTS=1`, so the end of a prompt writes the recurrent state to a snapshot; with
`--prefix-cache-n 0` the same prompt gives:

| | `--prefix-cache-n 1` | `--prefix-cache-n 0` |
|---|---:|---:|
| A | 95 ms | 59 ms |
| B (layer loop) | 3 364 ms | 3 179 ms |
| **C** | **396 ms** | **112 ms** |
| TTFT | 3 855 ms | **3 351 ms** |

So the snapshot costs **284 ms on a 3 143-token prompt** (0.09 ms/token), which is the per-token term that could not be
explained and is 7% of TTFT. It is not waste: it is what makes the next turn of a conversation cheap (prefix reuse
0.9+ on the chat corpus), so the trade is right for chat and wrong for a single-shot request - which is why the oracle
arm already runs with `--prefix-cache-n 0`.

**The attribution in full, for a 3 143-token prompt at chunk 1 792:**

| segment | ms | share |
|---|---:|---:|
| arrival + tokenise + schedule (A) | 95 | 2.5% |
| expert stream, 48 layers (copy at 13.4 GB/s) | 1 997 | 52% |
| the layer's other work, 48 layers | ~1 100 | 29% |
| of which PLE (host gather) | 94 | 2.4% |
| layer 0 alone (outside `sum(gap)`) | 143 | 3.7% |
| prefix-cache snapshot (C) | 284 | 7.4% |
| first decode step + detokenise + stream (rest of C) | 112 | 2.9% |
| **TTFT** | **3 855** | |

Which answers the original question about PLE completely: it is 2.4%, and the term that had been hiding as
"unattributed overhead" - bigger than PLE - is the prefix-cache snapshot, a deliberate trade for multi-turn reuse
rather than a cost of the prompt path itself.

---

## 22. The same prompt on both engines, segment by segment: the work is at parity, our snapshot is not

§21's three-segment method applied to one prompt (3 143 tokens, identical text) on both engines, same session, cold
processes. strata reports its own prefill as `3143 tokens = 0 reused + 3143 read in 3 198 ms (982.9 tok/s), ... 1
checkpoints`, which is the number to compare against our layer loop rather than the client's TTFT.

| segment | naiwa, `--prefix-cache-n 0` | naiwa, shipped (`-n 1`) | strata |
|---|---:|---:|---:|
| arrival -> first work | 59 ms | 95 ms | 71 ms (all of its overhead) |
| **the prefill itself** | **3 179 ms** | 3 364 ms | **3 198 ms (983 tok/s)** |
| prefix snapshot / checkpoint | - | **284 ms** | inside the 71 ms |
| first decode + detokenise + stream | 112 ms | 112 ms | inside the 71 ms |
| **client TTFT** | **3 351 ms** | **3 855 ms** | **3 269 ms** |
| per prompt token | 1.066 ms | 1.226 ms | 1.040 ms |

Three things this settles:

1. **The prefill work is at parity - ours is 0.6% faster** (3 179 against 3 198 ms). The expert stream dominates both
   (1 997 ms of our 3 179 is the copy at 13.4 GB/s), so the byte floor is the same and both engines sit on it.
2. **strata's out-of-prefill overhead is 71 ms; ours is 171 ms** with the snapshot off. So a single-shot request is
   **+82 ms (+2.5%)** - which is the whole gap, not the 28% §20.13 recorded (that was a squeezed chunk plan).
3. **The shipped config's snapshot is the one real difference: 284 ms, against strata's checkpoint inside 71 ms.** Both
   engines checkpoint (strata logs `1 checkpoints`, we write the recurrent state for `--prefix-cache-n 1`), and both do
   it for the same reason - the next turn reuses the prompt - but strata's costs a quarter of ours. That is why our
   shipped TTFT is +18% here while our prefill is at parity.

So the ranking of what is left on the prompt path, measured rather than assumed:

| | ms on a 3 143-token prompt | share of shipped TTFT |
|---|---:|---:|
| expert stream (byte floor, at the link's ceiling) | 1 997 | 52% |
| other layer work (GEMM/GDN/attention, PLE 94 of it) | ~1 100 | 29% |
| **our prefix snapshot** | **284** | **7.4%** |
| layer 0 (outside `sum(gap)`) | 143 | 3.7% |
| arrival + decode + stream | 171 | 4.4% |

The snapshot is the only line that is both large and **not** at parity with strata, so it is the next thing worth doing
- and unlike the ring, it does not need VRAM, only that the write happen on the copy stream (or after the response
starts) instead of in the request's critical path.
