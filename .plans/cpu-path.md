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

   So the lever exists but the ceiling is ~3 GB on a 12 GB card: the KV cache and staging rings need the
   rest. **Note also that these `NO_LOOKAHEAD` numbers understate production**: with the prediction thread
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

## Open risk seen today

`Invalid sampling probability ... NaN` has now been seen **three times**, at `--expert-cache-gb 2` (twice,
consecutively) and at `3.5` (once), always at `--prompt-len 512 --gen-len 32`, and every following re-run of
the same command was clean (12.1 / 11.9 / 12.3 tok/s). So it is rare, not cache-size-dependent, and it is a
*new* failure mode - a NaN in the logits rather than a different token - and it may be the same underlying
timing hazard as the one-off divergence recorded in `worklogs/m5.md` (which was characterised as a
two-variant rounding difference; a NaN does not fit that explanation). Treat it as an open correctness
risk and probe it with an output checksum over many runs before trusting long generations.
