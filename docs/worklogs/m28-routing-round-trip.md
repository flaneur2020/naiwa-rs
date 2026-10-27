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
