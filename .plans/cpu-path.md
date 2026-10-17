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

1. **Bigger expert cache.** 2 GB gives 19.7% at `frac 0`; the card has headroom because IQ1_M is small.
   Each 10 points of hit rate removes roughly 1.5-3 experts/layer from the CPU leg. This is the cheapest
   experiment on the list (a flag), and it also *unlocks MTP*, whose break-even needs >65% acceptance at
   depth 1 given the +29%/row cost.
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

## Open risk seen today

Two consecutive runs at `--prompt-len 512 --gen-len 32` died with `Invalid sampling probability ... NaN`
and two immediate re-runs of the identical command were clean (12.1 and 11.9 tok/s). That is a *new*
failure mode - a NaN in the logits rather than a different token - and it may be the same underlying
timing hazard as the one-off divergence recorded in `worklogs/m5.md` (which was characterised as a
two-variant rounding difference; a NaN does not fit that explanation). Treat it as an open correctness
risk and probe it with an output checksum over many runs before trusting long generations.
