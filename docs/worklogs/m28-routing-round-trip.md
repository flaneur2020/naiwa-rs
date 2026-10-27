# M28 - the per-layer routing round trip

**Status:** open, **nothing built**. Opened by M27 §8's measurement.
**Parent:** M27 (§8: strata's hit rate is *lower* than ours, and halving PCIe traffic moved nothing), M24 (§7, whose
"71% of the step is the transfer" this milestone corrects: the engine is busy, but the step does not wait on it).
**One mechanism:** keep the **routing decision on the device** for the layers whose experts are all resident, so only
the layers that actually stage or fall back to the CPU pay a host round trip - strata's own architecture, measured at
0.24 ms/layer against our 0.7-0.8.

## 1. The measurement

`NAIWA_STEP_BUDGET=1`, `code`, 40 tokens, `--mtp-n-predict 1`, cache 2.2 GB:

```
layers 48 total 45.0 ms expert_dev 24.0 | plan 37.8 gpu 2.4 staging_wait 3.0 cpu_join 0.3
pairs 960 distinct 743
```

`plan` - the phase in which the host has the device's top-k and decides what to stage, promote, or send to the CPU -
is **37.8 ms of 45.0**, i.e. **0.7-0.8 ms per layer**. nsys agrees from the GPU side (`doorbell_wait_copy_f32`:
about 3 000 calls, 382 us each per 40 tokens).

Two controls show it is the round trip and not the bytes:

| control | result |
|---|---|
| `--pcie-frac 0.4 -> 0.15` (62% less PCIe, `expert_dev` 24.0 -> 10.9 ms) | **TPOT 38.4 -> 38.5 ms: nothing** |
| strata's own profiler, same workload | `GPU-reach wait 11.51 ms/window` = **0.24 ms/layer**, 3x cheaper |

So the step is serialised at one host/device hand-off per layer, 48 times per step, and the pipeline is idle for most
of it.

## 2. Why this is the right next mechanism

- It is **80% of the step**, so it dominates every other term on the ledger (experts 24 ms, mixer 8 ms, head 8 ms).
- It explains three earlier negatives at once: more PCIe bandwidth (M26 reordering) cannot help a step that is waiting
  on a host decision; a drafter cannot help (M25) because the extra tokens ride the same serialised step; and the cache
  hit rate (M27) cannot help because a hit still costs the same round trip.
- strata's number proves the floor is reachable in this architecture - it is the same hand-off, done 3x cheaper.

## 3. Plan

1. **Attribute the 0.7-0.8 ms first** (measurement): split `plan` into (a) the D2H of the top-k itself, (b) the host's
   decision work (route, pick misses, staging bookkeeping), (c) waiting for the previous layer's kernels to publish the
   router output. nsys's `cuda_gpu_trace` already timestamps the router kernel and the copy, so this may need no code.
   Gate: the three parts must sum to the phase within 10%, three runs.
2. **Then, if (c) dominates** - most likely, since 382 us is far too long for a 40-value copy - the fix is to stop
   *following* the router with a host round trip: read layer L+1's routing while layer L computes, or (strata's
   structure) keep the decision on the device and only involve the host for the layers that must change their
   residency.
3. **If (a) or (b) dominates**, the fix is cheaper still: batch the D2H across layers, or shrink the decision work.

## 4. Gates

| gate | requirement |
|---|---|
| numeric oracle | `frac 0 + --no-lookahead` **byte-identical**, plus a must-fail control if the routing read moves to a device buffer (a stale top-k must change the output) |
| **the metric** | **`plan` per layer below 0.35 ms** (halfway to strata's 0.24) with step and TPOT recorded, never gated |
| liveness | 3 rounds x 4 classes, 0 failures; `cargo test` 53 passed / 4 ignored |
| not a regression | the shipped configuration (`--pcie-frac 0.4`, 2.2 GB, auto MTP + greedy, promote-on-stage) stays the default |

## 5. Not in scope

Bytes per step (M26), hit rate (M27, closed: strata's is *lower* than ours), the denominator (M25, closed: the MTP
head already accepts 0.90 where a suffix drafter would fire).

## 2. Attribution (§3 step 1 done): it is **83 blocking D2H copies per token at 183 us each**

No new code was needed - nsys's runtime-API table timestamps every host CUDA call, and per token the decode window
shows where the host's `plan` time goes:

| host CUDA API | calls per token | ms per token | **us per call** |
|---|---:|---:|---:|
| **`cuMemcpyDtoHAsync_v2`** | **83** | **15.27** | **183.4** |
| `cudaLaunchKernel_ptsz` | 990 | 4.99 | 5.0 |
| `cuMemcpyHtoDAsync_v2` | 744 | 3.62 | 4.9 |
| `cuLaunchKernel` | 578 | 2.51 | 4.3 |
| `cuMemAllocAsync` | 1 554 | 1.44 | 0.9 |
| `cuEventSynchronize` | 3 | 1.44 | **443.9** |
| `cuMemFreeAsync` | 1 553 | 1.38 | 0.9 |

Multiplying by the measured **1.13 tokens/step**: D2H **17.3 ms**, kernel launches **8.5 ms**, allocations **3.2 ms**,
event syncs **1.6 ms** - about **30 of the 37.8 ms** that `plan` occupies in a 45 ms step. The GPU, meanwhile, is
**52% busy**.

The number that matters is **183 us for a copy whose payload is a few kilobytes** (the whole D2H traffic is 0.3 MB per
token). That is not bandwidth: it is a **synchronous round trip** - the host issues the copy and waits for the device to
produce the buffer, i.e. it waits for the layer's queue to reach the copy. 83 of those per token, at one to two per
layer, is the serialisation M27 §8 measured as "0.7-0.8 ms per layer".

So the mechanism is **not** "keep the router on the device" (§1's hypothesis): the router kernel itself is 0.08 ms per
token. It is **stop issuing blocking D2H copies on the critical path**:

1. **Pipelined read**: copy layer L's top-k asynchronously into a **pinned ring** and let the host consume it one layer
   later, after an event it already waits on for the staging DMA. Nothing about the decision changes; it just stops
   being synchronised with the device's progress.
2. **One copy, not 83**: the ids and weights of a layer are contiguous on the device; a single 160-byte copy per layer
   replaces whatever the 1.5-per-layer pattern is, and the count per token drops from 83 to ~48.
3. **Allocation traffic**: 1 554 `cuMemAllocAsync` + 1 553 frees per token is 3.2 ms of host time for what should be
   pooled buffers - the same "one extra tensor per layer" cost M22 §19 could not attribute. Worth a look once (1) lands.

The metric for §4 stays **`plan` per layer below 0.35 ms**; with 17.3 ms of D2H alone, the pipelined read is the only
change that can reach it.

### Where the 83 copies come from

1.5 per layer is the shape of "read the top-k ids, then read the weights" (or a per-row split of the same), plus the
doorbell's own result copy. Confirming it needs one grep-level trace rather than a run: the next step of this milestone
is to instrument `Route::submit` and `doorbell::wait_copy` with a per-call counter and see which of them fires 1.5
times per layer.

## 3. Where the 83 copies come from - and the doorbell that already exists but is bypassed

`Route::submit` (`src/moe/stages/route.rs`) **already has** the pipelined read this milestone was going to build: a
doorbell ring that `publish_and_read`s the top-k ids and weights out of host-mapped memory, so "no stream sync, no D2H
launch" (its own comment). Its fallback is the blocking pair

```rust
topk_ids.reshape((rows, ()))?.to_vec2::<u32>()?,          // D2H #1
topk_weights.reshape((rows, ()))?.to_dtype(F32)?.to_vec2::<f32>()?,   // D2H #2
```

but the doorbell is only consulted when `device_flip` is **false**, and

```rust
let device_flip = layer.top_k() > 0
    && topk_ids.elem_count() % layer.top_k() == 0
    && topk_ids.elem_count() / layer.top_k() == rows;
```

is true for exactly the normal case (`elem_count == rows * top_k`). So the common path takes the doorbell *out* and
pays **two blocking `to_vec2` round trips per layer** - 96 per step, which is the 83-per-token count nsys measured, and
each costs 183 us because the copy cannot complete until the device's queue reaches it.

That reframes §1's plan into something much smaller and better supported:

1. **Route the common case through the doorbell** (the machinery exists and is already sized at warmup). Only the rare
   non-`rows x top_k` shape should fall back. Expected: 96 blocking round trips per step -> 0, i.e. **~17 ms of the
   37.8 ms `plan` phase**, which at 1.13 tokens/step is most of the 15.3 ms/token of D2H.
2. **Then** look at the 1 554 allocator round trips per token (3.2 ms of host time) - the same "one extra tensor per
   layer" cost M22 §19 could not attribute.
3. Keep §1 step 1's pinned-ring pipeline only if the doorbell path turns out to be the slow half; the code comment
   suggests it is the fast one.

Why this is the highest-value change left on the line: it is **one condition**, it is already implemented and tested
machinery, and it targets 17 of the 30 ms of host API time that make up 80% of a decode step.

### Correction to §2's ordering

§2 said "the next step is to instrument `Route::submit` and `doorbell::wait_copy` with per-call counters". That
instrumentation is what this section would have produced, and reading the code answered it directly: the counter is
1.5 per layer because the *fallback* runs, not because the doorbell is slow. No run was needed.

## 4. Tried: route the common case through the doorbell - **neutral**, and that is the interesting part

Changed the gate so the doorbell (`publish_and_read`, "no stream sync, no D2H launch") is used for **every** window
that fits the ring, with the capacity check as the only condition, and the blocking `to_vec2` pair left as the fallback
for prefill-sized windows.

| check | result |
|---|---|
| oracle (`frac 0 + --no-lookahead`) | **byte-identical** |
| `plan` | **34.7-44.8 ms** (was 37.8) - unchanged |
| TPOT | 37.7 / 39.5 / 39.9 ms - unchanged |

**Why it had to be neutral**: `publish_and_read` is *publish then `wait_and_read`* - the host spins on the device's seq
counter until the copy lands. It removes the `cuMemcpyDtoH` **API call**, not the **round trip**. The 183 us per call was
never the call; it was the wait for the device's queue to reach the copy, and the doorbell waits for exactly the same
thing with the same latency, just more cheaply.

So §3's premise ("one condition, 17 ms") is refuted, and the change was reverted to keep the tree pristine. What
survives is the diagnosis: **the host cannot decide layer L's experts before the device has produced layer L's routing,
and it waits 0.7-0.8 ms for that every layer.** No API choice fixes it.

### The only change that can: pipeline the routing read by one layer

Issue the router for layer L while layer L-1's **experts** are still computing, publish it, and consume it one layer
later. The experts of layer L still need the routing of layer L, but they only need it *after* layer L-1's experts have
finished - which is exactly the window the round trip can hide in. That is a schedule change across layers (the router
kernel moves earlier in the submission order), not a copy change, and it is the remaining item on this milestone. Its
gate is unchanged (`plan` < 0.35 ms per layer), and its own must-fail control is that a deliberately stale routing read
must change the output.

## 5. Correcting §4: the router cannot move earlier - the pipeline is between the host read and the **resident** compute

A fair objection, and §4's phrasing was wrong: **layer L's routing needs layer L's post-mixer state**, so the router
kernel cannot be issued before layer L's attention/GDN has run. There is no "route one layer ahead". What can be
overlapped is something else, and the distinction is the whole mechanism:

```
per layer today (serialised):
  mixer -> router -> [device done, idles] -> host reads ids (0.7-0.8 ms)
        -> host decides resident / staged / CPU -> device runs ALL experts

what the dependency actually requires:
  mixer -> router -> device runs the RESIDENT experts (ids are already on the device)
                                      meanwhile: host reads ids, decides, queues staging DMA + CPU experts
        -> device runs the STAGED experts when their DMA lands
```

The device does **not** need the host to start the experts whose weights are already resident - it has the ids. Only the
misses need the host: a DMA has to be issued and a CPU worker has to be told what to compute. So the round trip can hide
under the resident experts' kernels instead of preceding every kernel of the layer.

That is also the shape of strata's number: `GPU-reach wait 11.51 ms/window` **plus** `per-layer host 19.26 [CPU 19.10]`
- their host is in the loop for the CPU experts, and that work runs *under* GPU compute, which is why the same hand-off
costs them 0.24 ms per layer instead of our 0.7-0.8.

### What implementing it requires

1. **A device-side residency map**: `[num_experts] u32` per layer, `slot` or `SENTINEL`, mirrored from the host's
   bookkeeping whenever the cache changes (a 1 KB buffer; the host writes it on promote/evict, which is already a
   device-side copy today).
2. **A device-side compaction**: turn the step's `[rows, topk]` ids into a compacted `(row, slot, weight)` list of the
   resident hits. The quant crate already has `moe_align_em` for exactly this shape.
3. **Run the resident wave first**, with no host involvement at all - the existing `Gpu::submit` over the resident bank,
   but with its hit list produced on the device instead of by `Route::submit`.
4. **Then** the host's `Route` result arrives (the same doorbell), and the staging/CPU path proceeds as today, joining
   as the second wave (M26 left the wave machinery in place for this).

Gates are unchanged: `plan` under 0.35 ms per layer as the metric, oracle byte-identical, and the must-fail control
that a deliberately stale residency map changes the output (it would make the device read a slot that holds a different
expert).

## 6. The two numbers §5 turns on: resident compute **0.33-0.43 ms/layer**, round trip **0.72-0.93 ms/layer**

Per layer, from the same `STEP_BUDGET` lines (48 layers):

| quantity | per layer | from |
|---|---:|---|
| **round trip** (`plan`) | **0.72-0.93 ms** (median 0.85) | `plan` 34.7-44.8 ms / 48 |
| **resident-expert compute** (78% of the touches) | **0.33-0.43 ms** | `expert_dev` 20.5-26.6 ms x 0.78 / 48 |
| all-expert compute (for scale) | 0.43-0.55 ms | `expert_dev` / 48 |
| mixer (lower bound on any wait for routing) | 0.17 ms | 8.1 ms / 48 |
| **strata's same round trip** | **0.24 ms** | `GPU-reach wait 11.51 ms / 48` |

So:

1. **The round trip is 2x the resident compute.** Hiding the resident wave under the read does *not* make the layer
   free: it covers ~0.4 of the ~0.6 ms of avoidable wait. What is left is the mixer+router latency itself, which is
   unavoidable and is exactly what strata's 0.24 ms is made of.
2. **The ceiling is therefore `plan` ~0.4 ms per layer**, i.e. **~19 ms off a 41.7 ms step** (0.45 x 48), which takes
   the step to ~22-26 ms and TPOT from 38.5 to roughly **21-23 ms**: **-40%**, and it needs no change to bytes, cache,
   drafting or kernels.
3. If the round trip can also be overlapped with the *staged* wave (it cannot, the staged wave needs the host's
   decision) the number would be better, but that is a second, separate idea.

Order of work is unchanged: device-side residency map first, because it is the piece that lets the resident wave start
without the host, and it is independently verifiable with the stale-map must-fail control.

## 7. What §5 and §6 missed - read this before building anything from them

Thinking it through properly, four things do not hold together, and the honest status of §5/§6 is **a hypothesis, not
a diagnosis**.

### (a) `plan` is not one thing, and §5's fix only touches one of them

`plan` measures `Route::submit` + `DmaMiss::submit`. Inside that window there are at least four separate costs:

| candidate | evidence | would §5's device-side residency fix it? |
|---|---|---|
| the device read (host waits for the router) | 183 us per D2H call in nsys | **yes** |
| host bookkeeping: 20 expert ids -> slot lookups, `hit_counts`, mutex + HashMap | unmeasured | no |
| **`copy_stream::wait_for_readers`** - `DmaMiss::submit` spins until the previous step's readers drop their `Arc`s | unmeasured; the code documents it as a spin | no |
| **allocator**: 1 554 `cuMemAllocAsync` + 1 553 frees per token = 3.2 ms/token | nsys | no |

So "the round trip" may be a fifth to a half of `plan`, and the rest would survive §5 untouched. **Splitting `plan`
into those four is the actual first step**, not the residency map.

### (b) The overlap arithmetic contradicts the strong form of the claim

`plan` 34.7 ms inside a 41.7 ms step, with kernels at ~24 ms/step: 34.7 + 24 - 41.7 = **17 ms**, so **at least half of
`plan` already overlaps device work**. That means the host is *ahead* of the device during much of `plan` - queueing the
next layer while the device finishes this one - which is the opposite of "the device sits idle waiting for the host".
Both can be true only if the idle is concentrated in specific windows; §5 assumed it was every layer's read, and that
was not measured.

### (c) There is a **second** serialisation that §5 ignores

The GPU-side spin kernel (`doorbell_wait_copy_f32`) runs about 21 times per step at 382 us = **8 ms/step**, and it is
*the device waiting for the CPU experts*. The `cpu_join 0.3 ms` in the budget does not capture that. So the step has
**two** hand-offs - host-for-routing and device-for-CPU-results - and §6's 0.4 ms ceiling only reasoned about the first.

### (d) The gate as written cannot test the new path

The oracle runs at `frac 0`, where **staging is disabled**, so a device-side residency map would be exercised by
neither the oracle nor the must-fail control at that setting. A wrong residency map means the device reads a *different
expert's weights* - the most dangerous silent failure in the whole design. Any implementation must therefore add a
`--pcie-frac > 0` correctness arm with **exact token comparison** against today's binary on the same prompt, in
addition to M1's oracle.

### What to do instead of building the residency map

1. Split `plan` with four timestamps (`Route::submit` in-/out-, and inside it the read vs the bookkeeping; plus
   `DmaMiss::submit` in-/out-). Cheap, and §3/§4 already showed how much a wrong hypothesis costs.
2. Cross-check with nsys: in the decode window, measure **device idle inside each layer's `plan` window** (kernel gaps
   that fall between the router kernel and the first expert kernel). That number either supports §5 or kills it.
3. Only then decide between the three candidate mechanisms, of which the residency map is one.

§6's ceiling arithmetic (0.85 -> 0.4 ms/layer, TPOT ~21-23 ms) is **conditional on (a) being the device read**, which is
exactly what has not been measured yet.

## 8. Measured from the existing trace: the device is **busy 74% of the time the host is reading routing**

No rebuild needed - the nsys trace already holds both sides. For every one of the **9 995** `cuMemcpyDtoHAsync`
calls in the decode windows, I intersected the *host*-blocked interval with the merged *device*-busy intervals:

| | value |
|---|---:|
| D2H calls | 9 995 (83 per token) |
| mean call | 183 us |
| **calls during which the device was completely idle** | **6 081 (61%)**, mean **78 us** |
| calls overlapping device work | 3 914 (39%), mean **346 us** |
| per token: D2H total | 15.27 ms |
| ... of which device **busy** | **11.30 ms** |
| ... of which device **idle** | **3.97 ms** |

So the picture is bimodal, and §6's model was the wrong one:

- **Most of the read time (11.3 of 15.3 ms per token) is the host waiting while the device is busy** - that is the
  host *following* the device, not the device idling for the host. §7(b) suspected this from the overlap arithmetic;
  now it is measured. Making the resident wave device-side does **not** shorten those waits, because the device is
  already working during them.
- **The genuinely wasted part is 3.97 ms/token = ~4.5 ms/step** - the short reads (78 us mean) where the device has
  nothing queued. That, not 19 ms, is what any routing-side change can win: **TPOT -10%, not -40%**.

### And the rest of the device idle is not routing at all

Device idle is ~17.5 ms per 41.7 ms step (52% busy), and only ~4.5 ms of it falls inside the routing reads. The other
**~13 ms** is elsewhere, and the host-API ledger says where: per token the host makes

```
15.3 ms  D2H            (83 calls)
 7.5 ms  kernel launches (1568 calls)
 2.8 ms  cuMemAlloc/Free (3107 calls)
```

**4 675 host->device calls per token, 25.6 ms of host time in a 45 ms step.** The launches and allocator calls are
**40% of the host's CUDA time** and they are pure overhead - 1 553 alloc/free *pairs* per token for buffers that could be
pooled, which is the same "one extra tensor per layer" cost M22 §19 could not attribute (it is 0.3 ms/layer there,
2.8 ms/token here).

### Revised order of work for M28

| # | change | ceiling | cost |
|---|---|---|---|
| 1 | **pool the per-layer temporaries** (1 553 alloc/free pairs/token) | up to 2.8 ms/token ≈ 3.2 ms/step | small, local |
| 2 | **fewer launches**: 1 568 per token is 7.5 ms of host time; the decode sequence is fixed per shape, which is what a graph removes | up to 7.5 ms/token ≈ 8.5 ms/step | medium; M8/M9.1 measured per-layer graphs neutral, a whole-step graph was never re-tested after M24 |
| 3 | routing reads (the original §5 idea) | ~4.5 ms/step (TPOT -10%) | large: device-side residency map, and the gate in §7(d) must be added first |
| 4 | device-side residency map as a *prerequisite* of (2)? only if the graph cannot capture the routing read - the doorbell is host-mapped memory, so it can | - | - |

So the campaign order inverts: the cheapest 12 ms/step is in **allocator pooling and launch count**, both of which were
dismissed earlier (M22 §19: "the pool threshold changed nothing"; M24: "the launch gap is the copy engine"). §8's
numbers say what §19 lacked: it is not the pool's *threshold*, it is the *number of allocations*.

## 9. Pooling vs "no allocations at all": what each is worth, and what the second one requires

### The arithmetic

Per token: **3 107 `cuMemAllocAsync` + 3 107 frees at 0.9 us = 2.82 ms**, and **1 568 kernel launches at 5 us =
7.50 ms**. At 1.13 tokens/step that is **3.2 + 8.5 = 11.7 ms of a 45 ms step**.

| change | allocations per token | host time removed | step | TPOT |
|---|---|---:|---:|---:|
| today | 3 107 | - | 45.0 ms | 38.5 ms |
| **pool the temporaries** (reuse by shape+dtype) | ~120 | 3.1 ms | ~41.9 | **~35.9** |
| **whole-step CUDA graph** (all allocs and launches baked) | **0** | **11.7 ms** | ~33.3 | **~28.5** |

So pooling is worth about **2.5 ms of TPOT (7%)**, and it is strictly dominated by the graph, which removes the
allocator *and* the launches. Pooling is only interesting if the graph turns out to be unavailable.

### Why the allocation count is what it is

Candle's CUDA backend allocates **one device buffer per tensor**, and a decode step builds roughly 900-1 000 tensors
(about 19 ops per layer x 48 layers, each with a fresh output). 0.9 us per `cuMemAllocAsync` is already near the floor
for the API call - the pool behind it makes the *allocator* cheap, which is exactly why M22 §19's "raise the pool's
release threshold" changed nothing. What costs is the **call per tensor**, not the pool's policy. The number to attack
is the count: 3 107 -> 0.

### To get to zero, three things have to be true at once (they are not independent)

1. **Allocations and launches must be baked**: a captured CUDA graph replay issues no `cuMemAlloc` and no
   `cuLaunchKernel`; every buffer has a fixed address. That is the only mechanism that takes both terms to zero
   together.
2. **Nothing inside the step may depend on the host**: today the step reads routing from the device per layer
   (83 D2H/token) and the host decides staging / promotion / CPU split. A graph cannot contain a host decision, so
   **the device-side residency map of §5 (or its equivalent) is a prerequisite of the graph**, not an alternative to it.
3. **Shapes must be fixed per variant**: rows x speculation depth vary per step (`--mtp` auto depth), so the graph must
   be cached per (rows, depth) - a handful of variants, evicted on shape change, and a fallback to eager for anything
   uncaptured (prefill, first token, CPU-heavy paths).

If (2) cannot be made to hold, the fallback is **a candle-level bump arena**: one device slab per decoder, handed out as
`Storage` sub-slices and reset at each step boundary. That removes the 3 107 allocations without touching launches
(worth 3.1 ms), and it needs no capture - but it is a fork-wide change to how `CudaStorage` is created, and it breaks the
moment any tensor outlives the step (the prefix cache, the MTP state, the promote queue all do).

### Recommendation

Graph first, pooling only as its fallback - and **before either**, the §7(a) instrumentation, because both are large
changes justified by a host-API ledger that has so far been read one phase at a time.

## 10. Yes - pooling is the small change, if it is done in candle rather than as an arena

§9 overstated pooling's cost by describing only the arena variant. There is a much smaller version, and it has none of
the graph's prerequisites:

| | **candle size-keyed pool** | candle bump arena | whole-step graph |
|---|---|---|---|
| where | **one place in candle**: the CUDA alloc path + `Drop` of the storage | candle, everywhere storage is created | naiwa + candle |
| size | ~100-150 lines | hundreds, fork-wide | capture plumbing + graph cache per (rows, depth) |
| needs device-side routing (§5)? | **no** | no | **yes** |
| needs fixed shapes / no host decisions? | **no** | no | **yes** |
| breaks long-lived tensors? | no (they simply never return to the pool) | **yes** - prefix cache, MTP state, promote queue all outlive the step | no, as long as capture is re-done on shape change |
| removes | `cuMemAlloc`/`cuMemFree` calls (~2.8 ms/token) | same | allocations **and** launches (11.7 ms/step) |
| TPOT | 38.5 -> **~35.9 (-7%)** | ~35.9 | ~28.5 (-26%) |

How it works, in one paragraph: candle allocates one device buffer per tensor through `CudaDevice::alloc_zeros` and frees
it when the last `Arc` drops. A decode loop re-creates the **same shapes every layer and every step** (each layer has one
shape), so a per-device free list keyed by `(bytes)` - or `(bytes, dtype)` - will hit on essentially every allocation
after the first step. The `Drop` returns the buffer to the list instead of calling `cuMemFreeAsync`; the alloc pops it.
Steady state: **no allocator API calls at all**, just a hash lookup.

The two things that must be right, and they are the whole risk:

1. **Stream safety.** We have a second stream (the expert staging copy stream). `cuMemAllocAsync` memory is
   stream-ordered; a pool pop hands out memory that was last used on *another* stream, so a reuse can race a kernel that
   is still reading it. The pool must therefore be **per stream**, or a returned buffer must carry the event of its last
   use and the alloc must wait on it. Getting this wrong is a silent data corruption, not a crash.
2. **Fragmentation and leaks.** A pool keyed by exact size is fine here because shapes repeat; but a tensor that is kept
   alive (prefix cache, the promote queue) never returns and must not be counted on. Pool growth is then unbounded in
   the worst case: cap it, and fall back to the plain allocator on a miss above the cap.

This is why it should be first: it is **independent** of everything else on the line (routing, residency map, graph),
it is measurable with the ledger already in place (count `cuMemAllocAsync` per token - it should fall from 3 107 to
~0), and it does not change any numerics, so the M1 oracle and a `frac > 0` exact-token comparison both apply directly.

Order: **pool -> measure the ledger again -> then decide whether the graph's extra 8.5 ms is worth the residency-map
chain.**

## 11. Where the candle pool would actually go - and why this session stops before writing it

Checked the interception points, and they are clear:

- **Allocation** funnels through exactly two methods: `CudaDevice::alloc_zeros` / `alloc_uninit`
  (`candle-core/src/cuda_backend/device.rs:85,89`), which call `self.stream.alloc_zeros::<T>(len)` - cudarc, one driver
  call per tensor. There is **no pool hook anywhere** between candle and cudarc.
- **Release** has no hook at all: the
  buffer lives in `CudaStorageSlice` (`cuda_backend/mod.rs:102`), an enum over **eight** dtypes
  (`U8/U32/I16/I32/I64/BF16/F16/F32/F64`), and it is freed by cudarc's own `Drop` when the storage's `Arc` falls.

So the patch is: a `Drop for CudaStorage` that moves each variant's slice into a per-`(stream, dtype, len)` free list, and
`alloc_zeros`/`alloc_uninit` popping from it. Two consequences decide its size:

1. Every variant must be moved out in `Drop`, so the enum's payloads become `Option<CudaSlice<T>>` (or the drop writes a
   placeholder) - that touches every `match` over `CudaStorageSlice` in candle, which is a lot of them.
2. **`Drop` does not know the stream.** The storage does not carry one, so "free list per stream" cannot be keyed
   directly; the safe form is a single list plus a recorded event per returned buffer, and an `alloc` that makes the
   caller's stream wait on it. That is a `CudaEvent` per recycled tensor - cheap, but it means the pool is only a win
   because the *same* stream reuses it in the steady state, and the event wait is the correctness guard for the other
   one.

That is a genuine day's work with a silent-corruption failure mode, and this session no longer has the room to land it
**validated** (build 2 min, server 4 min, M1 oracle, a `frac > 0` exact-token comparison, `cargo test`, three benchmark
rounds). Starting it now would leave a fork-wide allocator change half-tested, which is the one thing the repo's rules
are unambiguous about not doing.

### Handoff specification, in the order it should be done

1. Change `CudaStorageSlice` payloads to `Option<CudaSlice<T>>` and add `Drop for CudaStorage` that pushes each into a
   global `Mutex<HashMap<(usize /*bytes*/, u8 /*dtype tag*/), Vec<CudaSlice<..>>>>` **plus** the event of its last use.
2. `CudaDevice::alloc_zeros` / `alloc_uninit` pop a matching buffer, `stream.wait(&event)`, and zero it where the caller
   asked for zeros; miss -> the existing cudarc path, and count the miss.
3. Cap the pool (say 2 GB of pooled bytes) and drop half the list when it is exceeded, so a leak in the model cannot turn
   into an OOM.
4. Validation: `cuMemAllocAsync` per token from nsys (**3 107 -> ~0** is the gate), `cargo test`, M1 oracle
   byte-identical, `frac > 0` exact-token comparison against today's binary, then three benchmark rounds.
5. Expected by the ledger: **-2.8 ms/token = -3.2 ms/step, TPOT 38.5 -> ~35.9 (-7%)**. If the measured win is under 1 ms,
   the pool is not worth its corruption risk and should be reverted - the same standard M26 was held to.

## 12. Better than the pool: **preallocate once, hand out by index** (no `Drop` hook at all)

§11's pool needed a `Drop for CudaStorage` across eight dtype variants plus an event per recycled buffer. That whole
problem disappears if nothing is ever *returned*: allocate the decode step's buffers **at init**, and make every later
allocation a *carve* out of that slab, indexed by an offset that is reset at a known point. In that design:

| | pool (§11) | **preallocated slab, index by offset** |
|---|---|---|
| `Drop` hook needed | yes, all 8 variants | **none** - memory is never returned |
| per-use driver call | no (`cuMemAllocAsync` removed) | **no** (`CudaSlice::slice(off..)` is a pointer, not a call) |
| event per buffer | yes, for cross-stream reuse | **none** - one stream, and the reset point is chosen where nothing is live |
| failure mode | use-after-free across streams | **use-after-reset** (same class, but detectable by poisoning) |
| long-lived tensors | they just never return | **must be excluded** and allocated outside the slab |

The hook points are the same two functions as before (`CudaDevice::alloc_zeros` / `alloc_uninit`), but now they are
trivial: `off += bytes; return slab.slice(off..off+bytes)` - no map, no mutex (a `Cell<usize>` per device is enough, since
the token thread is the only allocator during decode), no events.

### What makes it safe, and how to prove it

The design's whole risk is a tensor that is still alive when the offset resets. Three rules make it checkable:

1. **Reset at a point where the previous step's temporaries are provably dead**: candle drops each op's output when the
   last `Arc` falls, and a decode layer's temporaries do not escape - the layer's output is written into the residual,
   which is allocated *outside* the slab. So the reset belongs at the **start of each layer**, not per step.
2. **Exclude the escapees by construction**: prefix-cache entries, MTP state, the promote queue and anything handed to
   the loader are allocated through the ordinary allocator (a one-line opt-out flag on the request), because they
   outlive the layer.
3. **Poison the slab on every reset** in a debug/`NAIWA_SLAB_POISON=1` build (write NaN / `0xDEADBEEF` into the region
   about to be handed out). A use-after-reset then shows up as NaN logits or a garbage token instead of a plausible
   wrong answer - and it is exactly the must-fail control this change needs, since neither the M1 oracle nor the
   `frac > 0` comparison would necessarily hit the boundary case.

### What has to be measured before it can be sized

The slab must be big enough for a layer's peak live bytes, and nothing yet says what that is. The 3 107 allocations per
token are ~57 per layer per step; their **sizes** were never logged. So step one is a byte counter in the same two
functions (`NAIWA_ALLOC_TRACE=1`: count, total bytes, peak live bytes, per layer), run for one probe. Two numbers come
out of it - the per-layer peak (the slab size, with headroom) and the shape histogram (whether one slab suffices or the
working set is dominated by a few big buffers that want their own slots).

If the per-layer peak turns out to be large (say 500 MB) the slab is still fine - it replaces memory that candle was
allocating and freeing every layer anyway - but it becomes a **fixed** cost that must fit the 12 GB card alongside the
2.2 GB expert cache, which is the same budget fight M20 lost at 2.3 GB. That is the number to know before writing the
allocator.
