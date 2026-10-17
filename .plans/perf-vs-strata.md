# Where the 6x vs Strata comes from

Measured on 2026-10-17. Two engines, **the same model family** (Qwen3.8-Flash-Next), two cards.

## The numbers

| | Strata | naiwa-rs v2 |
|---|---|---|
| model | Qwen3.8-Flash-Next IQ3_XXS / IQ2_XS | Qwen3.8-Flash-Next GSQ-RCO **IQ1_M** |
| card | RTX 4070 Ti SUPER, 16 GB, sm_89 | RTX 4070, **12 GB**, sm_89 |
| decode | **76.0** tok/s (IQ3_XXS, 32k) / **97.9** (IQ2_XS) | **11.6-13.3** tok/s |
| per pass | ~33 ms (76 tok/s at ~2.5 tok/pass) | **78-86 ms, 1 token/pass** |
| speculation | `--spec 4 --mtp`, ~62-68% accepted | **none** - the artifact has no MTP head |

Strata's figures are from `~/code/strata/bench/results/2026-10-05-community-rtx-4070-ti-super/README.md`
(engine-reported `predicted_per_second`, temperature 0, 192 tokens, three SSE calls). Ours:
`bench --prompt-len 512 --gen-len 32 --depth 4 --pcie-frac 0 --expert-cache-gb 2` with
`NAIWA_NO_LOOKAHEAD=1`.

## The gap decomposes into two factors, not one

```
11.6 tok/s   x2.5 (speculation: 1 -> ~2.5 tokens/pass)   = 29 tok/s
             x2.6 (pass latency: 86 ms -> 33 ms)         = 76 tok/s   <- strata
```

**1. Tokens per pass: ~2.5x, and it is blocked by the artifact, not by code.** `--mtp` refuses on this
checkpoint:

```
Error: `--mtp` requested but the checkpoint has no built-in MTP head (`mtp.fc_embedding.weight`).
```

The engine has the path (`args.rs --mtp`, `qwen4_exp/mtp.rs`), and the requant shards simply do not carry
the head. With the head present, the same 86 ms pass would carry ~2.5 tokens - and the number would rise
by that factor minus the extra work the verify rows cost (2-3 rows instead of 1: our per-layer host work
barely moves, the GPU segments grow) - i.e. roughly **20-29 tok/s** without touching anything else. That is
also why "mtp2 -> 40 tok/s" is remembered: speculation is worth more than every kernel optimisation on this
list put together.

**2. Pass latency: 86 ms against ~33 ms, ~2.6x.** And we know exactly where ours goes, because it was
measured: `NAIWA_STEP_DBG` dry time is **1.53-1.63 ms per layer x 48 = 74-78 ms**, and the GPU is idle 98%
of that. The token thread is the bottleneck: per layer it reads the top-k back through the doorbell,
computes the miss set, issues the staging DMAs and writes the CPU payload. Strata's design instead keeps
the host *out* of the critical path - `BATCHED_DMA`, `KV_PREFETCH`, `DISJOINT_EXPERT_CACHE`, 23 expert-pool
workers - and the GPU runs the pass as one graph.

Card caveat, stated honestly: strata had 16 GB to our 12 GB, so it holds more experts resident; ours is
2 GB of expert cache (~21 slots/layer) with an **82-84% hit rate**, so ~1.5-3 experts per layer go to the
CPU, and that CPU leg is what the 1.6 ms/layer is made of. Its card also fits IQ3_XXS where we run IQ1_M
(fewer bytes per weight, which should favour *us* on the bandwidth-bound part).

## What this means for the roadmap

| lever | expected | cost | verdict |
|---|---|---|---|
| **MTP/spec decoding** | x2.5 tokens/pass (~20-29 tok/s) | needs a checkpoint with the MTP head (`mtp.fc_embedding.weight`), then accept-verify plumbing for the MTP site | **by far the biggest**; also the only one that gets us near 40 |
| **host-loop pipelining** (batch the staging DMAs, prefetch the top-k a layer ahead, cut the doorbell readback) | 86 -> ~50-60 ms/pass (x1.4-1.7) | medium: the doorbell protocol already supports "issue, don't perform" | second |
| **bigger expert cache / better hit rate** | fewer CPU experts -> less host work | VRAM-limited at 12 GB; IQ1_M leaves headroom | third |
| **M6 2c (attention capture)** | 1-4 ms/pass (5%) | ~150 lines, residents half-done | last |

So M6 is *not* where the 6x lives. It is worth doing for the mechanism (and it deletes work), but the
honest ranking is: get the MTP head, then take the host out of the inner loop.
