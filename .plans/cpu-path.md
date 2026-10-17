# CPU path: what is left to optimise

All numbers measured on this box (RTX 4070 12 GB, IQ1_M, `--cpu-moe`, `frac 0`, `NO_LOOKAHEAD`).

## Where the time actually goes

```
per pass 82-86 ms   = 48 layers x 1.53-1.63 ms of *host* time   (NAIWA_STEP_DBG dry time)
GPU idle share      ~98%
row sensitivity     1 row 82 ms -> 2 rows 106 ms   (+29%, i.e. +0.5 ms/layer per extra row)
hit rate @ frac 0   19.7%  (3402 hits, 13878 misses over 32 tokens)
```

So the CPU side is not one thing, it is three, and they scale differently:

| component | size | scales with |
|---|---|---|
| per-layer host bookkeeping (read the top-k back, compute the miss set, issue the DMA, write the payload) | ~1.0-1.2 ms/layer | layers (constant in rows) |
| activation publish spin (`wait_publish`) | 0.42 ms/layer at M5 time, now overlapped | layers |
| CPU experts: router gemv + one gemv per missed expert | ~0.2-0.5 ms/layer at 80% hit, much more at 20% | **misses, i.e. rows** |

That last row is why the 2-row pass costs +29%: doubling the rows roughly doubles the selection count, and
with a 20-80% hit rate that means 1.5-3 more experts per layer on the CPU plus their DMAs.

## The options, ranked by (gain x certainty) / effort

1. **Bigger expert cache - measured, and capped by the card.** At `frac 0` (no lookahead):

   | `--expert-cache-gb` | result |
   |---|---|
   | 2.0 | 12.3 tok/s (81.2 ms), **19.7% hit** |
   | 3.5 | `Invalid sampling probability ... NaN` (the flake below, not a hit-rate result) |
   | 4.5 | KV cache cannot be placed any more: "no usable token blocks after reserving block zero" |

   Shrinking the KV cache (`--pa-context-len 4096` -> 118 MB instead of 2 626 MB) makes 4 GB fit, and the
   answer is **negative**:

   | cache | slots/layer | result |
   |---|---|---|
   | 2.0 GB | 21 | **12.4 tok/s (80.8 ms)** |
   | 4.0 GB | 42 | 11.8 tok/s (84.7 ms) - *slower* |
   | 5.5 GB | 58 | `Failed to create hybrid cache: CUDA_ERROR_OUT_OF_MEMORY` |

   Doubling the slots costs ~4 ms a pass rather than saving any: the expert working set is
   `48 x 256 x 2.1 MB = 25.9 GB`, i.e. 6-13x this card, so 2 -> 4 GB barely moves the miss count, while
   every newly admitted slot is a 2.1 MB copy and the admission bookkeeping grows. **Cache size is not the
   lever on this model; the cost per miss is.** (The KV cache itself is cheap: 118 MB at a 4 k context, and
   it is not what limits a bigger cache - the weights plus the expert bank are.) **Note also that these `NO_LOOKAHEAD` numbers understate production**: with the prediction thread
   on, the hit rate is 81-84% instead of 19.7% (M5 measured 78.2-86.7 ms TPOT), because the preload fills
   the cache in the gap. `NO_LOOKAHEAD` is used for determinism, not because it is the shipping config.
2. **Take the miss decision off the host.** Today the token thread, per layer: spins for the top-k, hashes
   the ids into a resident map, builds the copy list, then writes the payload. A small device kernel could
   turn top-k into a *compact miss list + count* in mapped memory; the host would then only read that list
   (no set math, no per-id branching) and queue the copies. The copies stay host-issued - there is no way
   around that - but the decision cost per layer collapses to a memcpy of the list.
3. **Batch the CPU experts per layer.** Check first: if `run_hits` issues one gemv per missed expert, a
   single grouped gemv per layer (the same call the GPU path uses, with the CPU as "device") removes
   per-expert dispatch. Worth ~0.1-0.2 ms/layer at 20% hit, more at 0%.
4. **More workers for the CPU leg only.** M5 measured 2/4/6 prediction threads at 76.5/78.4-80.3/78.0-79.0%
   hit - i.e. the *prediction* was not thread-limited. The expert gemvs are already on rayon; whether they
   saturate is unmeasured and is the next thing to measure with `RAYON_NUM_THREADS`.

## What is *not* worth doing

- Kernel micro-optimisation on the GPU side: the GPU is idle 98% of the pass.
- More gates/captures (M6): worth 1-4 ms of 82 ms.
- Anything that makes the per-layer host path *wider* rather than *shorter*: the pass is host-bound, so
  only removing work (or moving it to the GPU) helps.

## NaN logits: first fix landed (payload zeroing), rate down from ~2-3/5 to 1/5

`ScheduleBuf`'s payload area was never initialised. `cudaHostAlloc` does not hand out zeroed memory, and a
consumer that passes its gate while a slot still holds garbage adds it to the layer's accumulator - garbage
f32 bits are frequently NaN/Inf. Zeroing the whole mapping (not just epoch + ready words) is correct
regardless, and it moved the failure rate from **~2-3 runs in 5 to 1 in 5** (5 runs after the fix:
85.1/82.7/82.2/83.9 ms clean, then one NaN).

So that was *a* source, not *the* source. What is left to look at, in order:

1. **`DoorbellBuf` and `FlagBuf`** - same `cudaHostAlloc` treatment; the top-k ring and the flag ring are
   read by kernels, and a stale-but-valid-looking ring can hand the layer a garbage expert id (that path
   has a sanity bail, so it would error rather than NaN - weaker suspect);
2. **a genuine lost wakeup / level-vs-edge slip** in the per-layer points: a layer that reads the
   *previous* pass's payload gives a wrong token, not a NaN, so this does not explain it either;
3. **uninitialised KV blocks**: a slot mapping that lands on a block never written would feed NaN into
   attention. This is the only remaining mechanism that produces a NaN directly, and it would be sensitive
   to the cache/KV layout - consistent with the failure being more common when the KV cache or expert cache
   size changes.

## Open risk seen today

`Invalid sampling probability ... NaN` has now been seen **three times**, at `--expert-cache-gb 2` (twice,
consecutively) and at `3.5` (once), always at `--prompt-len 512 --gen-len 32`, and every following re-run of
the same command was clean (12.1 / 11.9 / 12.3 tok/s). So it is rare, not cache-size-dependent, and it is a
*new* failure mode - a NaN in the logits rather than a different token - and it may be the same underlying
timing hazard as the one-off divergence recorded in `worklogs/m5.md` (which was characterised as a
two-variant rounding difference; a NaN does not fit that explanation). Treat it as an open correctness
risk and probe it with an output checksum over many runs before trusting long generations.
