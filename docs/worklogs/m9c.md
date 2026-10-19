# M9c — doorbell: no gates, checked rings

Status: **done**. Implemented after the review at `ba33a28` (`~/code/strata` as the cross-reference).
Parent: [m15.md](m15.md).

Scope: `src/doorbell/` plus the call sites in `moe/{decode,cached_layer,provider}.rs`,
`moe/stages/{route,preload,dma_miss}.rs`, `executor.rs`, `copy_stream.rs`.

## What landed

| before | after |
|---|---|
| `GateSet` + `Gate`: epoch word, `ready[2*slots]`, `bump_epoch`, `publish_max` kernel, `wait_ready` kernel | **gone** (`gate.rs`, 196 lines) |
| `SYNC_PRELOAD` published on candle's stream by an `atomicMax` kernel, copies on the copy stream | one `CopyDone` + claim counter per layer, waited before the resident compute |
| `SYNC_STAGING` published with `cuStreamWriteValue32` on the copy stream | one `CopyDone` per layer, recorded after the batch, waited after the resident compute |
| `pass` threaded through `decode → Preload → DmaMiss → prefetch_predicted`, `PASS` static, layer-0 bump | **gone**: the claim cell owns its sequence |
| `preload_claimed: AtomicU32` + `== pass` | `PassCell` (claim count + event); the consumer waits for the claim it sees |
| `staging_enabled()` = `pcie_frac > 0 && copy_stream.is_some()`, gates failing open | `staging_enabled()` = "this layer has a staging event"; a missing event disables the path |
| `ResultDoorbell` ring depth assumed | device-written `consumed` word; `publish` waits (bounded) for the slot's previous lap and counts a drop |
| `executor::global_channel` grew at runtime, `retired` forever | sized once at warmup for `MAX_DECODE_ROWS * hidden`; growth only possible for the startup shape check and warns |
| `TopkDoorbell` lazily acquired at the first decode, cap `max(n, 8*top_k, 64)` | warmup-allocated per layer for `MAX_DECODE_ROWS * top_k`; a publish that does not fit is an error, not a silent `to_vec2` |
| `TopkDoorbell.seq: AtomicU32` mirror | the mapped device counter is the sequence (read before publishing) |
| a fresh `CudaEvent` per activation publish | one persistent event per channel |
| `ActivationDoorbell` side stream created with `.expect()` on the decode path | created in `acquire`; failure is `Ok(None)` → the existing blocking fallback |
| `doorbell_free_host` declared `()` though C returns `int` | declared `i32`, logged at debug at teardown |

`src/doorbell/`: 1091 lines (889 Rust + 202 `.cu`) → **991** (842 + 149) — the small net change is
documentation; the deleted gate module is replaced by `handoff.rs` (61) and explicit per-kernel
comments. `doorbell.cu` has two kernels instead of five. `git diff --stat`: 24 files, +556/−693.

## The two findings that were real

**1. `SYNC_PRELOAD` was published on the wrong stream — a live correctness bug.** The expert copies
were queued on the copy stream (`cached_layer.rs` `queue_async`) while
`GateSet::publish_stream_max` launched its `atomicMax` on candle's stream, so the consumer
(`decode.rs`) could be released onto half-written slots. The review left this as an open question
("confirm the two streams differ"); they differ by construction (`CopyStream::new` is the only
constructor and calls `new_stream()`), so no probe was needed. It is now a completion event recorded
on the copy stream.

**2. A late publish was skipped, not drained.** `decode.rs` waited only when
`preload_claimed == pass`, so a copy the lookahead queued late (the lookahead thread is FIFO and can
be stuck in `wait_for_readers`) left a pass reading slots mid-copy with no wait at all — moving the
publish to the right stream did not fix it. The fix is the shape of the handoff: the claim and its
event are one object, and the consumer waits for the claim it sees, including a stale one.

## Measurements

Gates (all met):

| gate | result |
|---|---|
| `cargo check --offline --features cuda` | 0 naiwa warnings |
| `cargo test --offline --features cuda --bin naiwa-rs` | **34/34** |
| oracle `frac 0` + `--no-lookahead` | byte-identical to `harness/m1/reference.oracle.txt` |
| … with `--prefill-stream` | byte-identical |
| np8 counting prompt | **parity**: base 52.4 T/s mean (10 runs, 51.0–53.7), new 52.6 (6 runs, 51.7–53.4); tail 8 base 44.6–48.6, new 44.3–49.7; hit 90.2–90.6% both |

Bench command (`~/bench-rel/m9c/`): `bench --cpu-moe --expert-cache-gb 4 --pcie-frac 0.2 --depth 4
--mtp --mmproj ~/bench-rel/mtp/mtp-component.gguf --mtp-n-predict 8 --prompt-file
~/bench-rel/pf/prompt-count.txt --gen-len 96 --iterations 3 --warmup 1`. A/B binaries were built
from the parent commit and from this tree in the same session and interleaved.

**The one measurement that changed the design.** A fresh `CopyDone` per claim is the obvious
encoding of "record this batch's completion", and it is wrong: `cuEventDestroy` on an event whose
record is still queued on the copy stream **blocks**, so each lookahead job paid ~0.95 ms and
`job_ms/step` went 1.98 → 2.25. That cost 1.2 points of hit rate and ~4% throughput. One
persistent event per layer, re-recorded per claim, is 1.92 ms/step — *faster* than the old
`atomicMax` kernel launch and its `as_cuda_device`. Reuse is safe because a new claim for a layer
comes from a job the token thread scheduled one layer earlier, i.e. after its wait for the previous
claim was enqueued. `CopyDone::record` binds the CUDA context itself, because a producer that
queued no copies (every prediction already resident, `--lookahead-swaps 0`) has not bound it on
the lookahead thread yet.

## Where the review was wrong

- **"Delete the epoch and the `PASS` static"** cannot both be done: with a numeric target the pass
  has to survive. The implemented design deletes *both* by giving the claim cell its own sequence
  and having the consumer wait for the claim it sees — which is also what fixes finding 2.
- **"The epoch is a latent hazard"** was not demonstrable: both the epoch and the host pass advance
  in the same thread at layer 0, so the device could not lead. The epoch was dead weight, not
  dangerous.
- **"`SYNC_STAGING` can become `CopyDone` with no replacement mechanism; preload needs one"** —
  right about the shape, wrong about the cost, and the review's "one persistent event per channel"
  advice for the activation publish does *not* transfer to the preload (it transfers to the
  activation publish only). See the measurement above.
- **"~990 → ~600 lines"** does not happen: the gate module was 196 lines and the kernels 53, and
  the handoff plus the new comments are ~140. The win is that the decode path has no spin kernel,
  no epoch, no pass, no `atomicMax`, and one fewer failure mode (a barrier that nobody publishes).
- **"Four mechanisms for two needs"** was rhetoric; it is three payload primitives plus one
  barrier. The ring merge (review item 4, optional) was **not** done: the two rings differ in
  direction, writer thread and consumption tracking, and only the layout is shared.

## Not done, deliberately

- **Ring merge** (`PayloadRing` + a `reserve` that owns backpressure for both directions): the
  layout is already shared and the three sequence owners are on three different threads, so the
  refactor buys the depth check and nothing else. `ResultDoorbell::publish` now does that check.
- **Per-step `Vec` allocation in `ActivationDoorbell::wait_publish`**: the buffer is one layer's
  activation per pass and the allocation is on the lookahead thread, off the token path.
- **`TopkDoorbell::publish_and_read` dtype/contiguity conversion**: it is cheap when the tensors
  already match (they do) and keeping it here keeps the call site one line.
- **Bound on the two device spins left**: `doorbell_wait_copy_f32` stays unbounded on purpose (a
  dropped CPU result corrupts the output, a hang does not); the host-side result-ring check is now
  bounded. The two gate spin kernels are gone, so the strata lesson ("every device spin needs a
  bound") now applies to one kernel, documented at it.

## Open

- `--pcie-frac 0` with a lookahead: the preload claim still exists (the layer has a copy stream) and
  is still waited for. Correct, and it removes the "no gate buffer" fail-open mode entirely.
- The startup shape-check run now allocates the result ring before warmup sizes it (it is the only
  growth). If a future change makes the decode window larger than `MAX_DECODE_ROWS`, warmup says so
  with a warning instead of reallocating silently.

---

## Appendix — the review as written before implementation

Kept verbatim so the proposal and the outcome can be read together; the section above is the
authority where they disagree.

> ### 1. `SYNC_PRELOAD` is published on the wrong stream (suspected bug)
> … copies queued on the copy stream, `publish_stream_max` on candle's stream → the consumer is
> released onto half-written slots; same-stream deadlock window from the claim/store ordering.
> Not in the file any more: this was the live bug and it is fixed.
>
> ### 2. The epoch is vestigial
> `GateSet` waits on `ready[id] >= *epoch`, which only existed because a captured graph freezes
> kernel arguments; M6 removed captures. Deleting it also removes `doorbell_epoch_bump_kernel`,
> `GateSet::bump_epoch`, the `PASS` static, the layer-0 special case and the design rule.
> **Partly wrong**: the `pass` cannot be deleted while the wait still takes a numeric target. The
> implemented design deletes both by moving the sequence into the claim.
>
> ### 3. The gate itself can be events
> Both gate users are host-side producers of stream-ordered work, which is exactly `CopyDone`.
> **Right about the shape, wrong about the cost**: a fresh event per claim blocks in
> `cuEventDestroy` (~1 ms/job). One persistent event per layer.
>
> ### 4. Backpressure is implemented for one ring and assumed away for the other
> `ResultDoorbell` had none, only a comment. A device-written `consumed = seq` turns the assumption
> into a check. **Done**, plus a bounded wait and a counted drop.
>
> ### 5. Failure opens instead of closing
> `provider.rs`: no gate buffer → `warn!` → `replay_gate` is a no-op while `dma_miss` still DMAs.
> **Done**: `staging_enabled()` is "this layer has a staging event", so a missing event disables
> the path; warmup allocation failures are errors.
>
> ### 6. Smaller items
> Header offset 8 written three times; `TopkDoorbell::wait_and_read` returning an unread value and
> doing dtype conversion; the capacity cliff (`max(n, 8*top_k, 64)`) versus `ActivationDoorbell`'s
> hard error above 16 rows; `executor::global_channel` growing at runtime; a fresh `CudaEvent` per
> publish; `Gate` public but not re-exported; `doorbell_free_host` returning `int` but declared
> `()`.
> **Done** except the dtype conversion (kept at the call site's convenience) and the capacity
> policy, which is now one warmup size for both (`MAX_DECODE_ROWS`).
>
> ### Cross-reference: strata
> A flat struct, no ring/gate/epoch/backpressure — a different host loop, not a better design.
> Two lessons imported: every device spin needs a bound, and every counter written to mapped memory
> is volatile. After this milestone only `doorbell_wait_copy_f32` spins, and the host→device
> direction documents its x86-TSO assumption.
>
> ### Proposed order
> 1. `SYNC_PRELOAD` on `cs.cu_stream()` — landed first, alone, after the stream identity was
>    established from the code instead of a probe.
> 2. Delete the epoch → **skipped**: item 3's shape deletes it together with the pass.
> 3. Staging → `CopyDone`; preload → claim + event. **Landed**; the claim and the event are one
>    object, which is what also fixes the late-publish race.
> 4. Optional ring merge — **not done** (see "Not done, deliberately").
