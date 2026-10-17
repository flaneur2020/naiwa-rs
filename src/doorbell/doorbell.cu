// Doorbell ring: publish this layer's top-k into a mapped pinned buffer so the host can
// read ids/weights with no stream sync and no D2H launch (Step 5 of the
// spin-pool/doorbell plan).
//
// The pattern is the one the harness proved (~/bench-rel/doorbell/harness.cu, 1000/1000
// on Ada): device writes the payload, __threadfence_system(), then a volatile store
// bumps seq; the host spins on seq and reads the payload behind it.
#include <cuda_runtime.h>
#include <stdint.h>

// One block. Copies `rows * topk` ids and weights from device tensors (produced by
// moe_router_topk earlier on the same stream, so this kernel is stream-ordered after
// them) into mapped host memory, then bumps the sequence counter.
__global__ void doorbell_ring_kernel(
    volatile uint32_t* d_ids,      // [rows * topk] u32, mapped
    volatile float* d_weights,     // [rows * topk] f32, mapped
    volatile uint32_t* d_seq,      // one u32, mapped: host polls this
    const uint32_t* src_ids,       // device, produced earlier on this stream
    const float* src_weights,      // device
    int n)                         // rows * topk
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d_ids[i] = src_ids[i];
        d_weights[i] = src_weights[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Order the payload writes host-visibly before the seq bump.
        __threadfence_system();
        *d_seq = *d_seq + 1u;
    }
}

extern "C" int doorbell_alloc_mapped(void** out, size_t bytes) {
    // cudaHostAllocMapped: the same bytes are visible from host and device.
    return (int)cudaHostAlloc(out, bytes, cudaHostAllocMapped);
}

extern "C" int doorbell_free_host(void* p) {
    return (int)cudaFreeHost(p);
}

extern "C" void doorbell_get_device_ptr(void* host, void** dev) {
    cudaHostGetDevicePointer(dev, host, 0);
}

extern "C" void doorbell_ring(
    void* d_seq,             // mapped seq counter, device view (also host-visible)
    void* d_ids,             // mapped ids buffer, device view
    void* d_weights,         // mapped weights buffer, device view
    const void* src_ids,     // device tensor of ids, stream-ordered before this launch
    const void* src_weights, // device tensor of weights
    int n,                   // rows * topk
    int64_t stream)
{
    doorbell_ring_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (volatile uint32_t*)d_ids, (volatile float*)d_weights,
        (volatile uint32_t*)d_seq,
        (const uint32_t*)src_ids, (const float*)src_weights, n);
}

// ---------------------------------------------------------------------------
// R2 flag channel (bidirectional doorbell):
//   GPU -> host: a kernel copies a payload into mapped memory and bumps seq
//                (the host spins, see doorbell_ring_kernel above).
//   host -> GPU: the host writes a payload and bumps seq; a stream-ordered
//                wait kernel spins on the mapped seq until it reaches the
//                target, replacing cudaLaunchHostFunc so the host never
//                blocks and the pattern stays graph-capturable (R2b/R5).
// ---------------------------------------------------------------------------

__global__ void doorbell_copy_f32_kernel(
    volatile uint32_t* d_seq,   // one u32, mapped: the host polls this
    volatile float* d_dst,      // payload, mapped
    const float* src,           // device tensor, stream-ordered before this launch
    int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d_dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Order the payload writes host-visibly before the seq bump.
        __threadfence_system();
        *d_seq = *d_seq + 1u;
    }
}

__global__ void doorbell_wait_seq_ge_kernel(
    const volatile uint32_t* d_seq,  // mapped, host bumps it after writing payload
    uint32_t target)
{
    if (threadIdx.x == 0) {
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
    }
}

extern "C" void doorbell_copy_f32(
    void* d_seq,          // mapped seq counter, device view
    void* d_dst,          // mapped payload, device view (float*)
    const void* src,      // device f32 tensor
    int n,                // float count
    int64_t stream)
{
    doorbell_copy_f32_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (volatile uint32_t*)d_seq, (volatile float*)d_dst, (const float*)src, n);
}

extern "C" void doorbell_wait_seq_ge(
    void* d_seq,          // mapped seq counter, device view
    uint32_t target,      // the host's seq value once its payload is complete
    int64_t stream)
{
    doorbell_wait_seq_ge_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, target);
}

// Host -> GPU result leg (R2b): the host writes a payload slot of a mapped
// ring and bumps seq; this kernel waits for seq >= target, then copies the
// slot into device memory. Replaces wait-for-worker host function + H2D
// memcpy: stream-ordered, no host involvement, graph-capturable. The slot
// stride keeps a slow GPU reading slot k while the host writes k+1 (depth >= 2).
__global__ void doorbell_wait_copy_f32_kernel(
    const volatile uint32_t* d_seq,   // mapped, host bumps it after the payload
    uint32_t target,                  // seq value that means "slot complete"
    const volatile float* d_payload,  // mapped payload ring, device view
    int64_t slot_off,                 // float offset of this publish's slot
    float* dst,                       // device destination
    int n)                            // float count
{
    if (threadIdx.x == 0) {
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = d_payload[slot_off + i];
    }
}

extern "C" void doorbell_wait_copy_f32(
    void* d_seq,          // mapped seq counter, device view
    uint32_t target,      // seq value that means "payload slot complete"
    void* d_payload,      // mapped payload ring, device view
    int64_t slot_off,     // float offset of the slot to read
    void* dst,            // device f32 destination, stream-ordered afterwards
    int n,                // float count
    int64_t stream)
{
    doorbell_wait_copy_f32_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, target,
        (const volatile float*)d_payload, slot_off,
        (float*)dst, n);
}

// ---------------------------------------------------------------------------
// F3 / v2 M0: zero-argument variants. `target` and `slot_off` are read from
// mapped memory instead of being kernel arguments, so one captured graph can be
// replayed with fresh values each step (a kernel argument is frozen at capture).
// Verified in a standalone harness on Ada (../bench-rel/naiwa-v2/m0).
// ---------------------------------------------------------------------------
__global__ void doorbell_wait_seq_ge_ptr_kernel(
    const volatile uint32_t* d_seq,     // mapped, host bumps it after payload
    const volatile uint32_t* d_target)  // mapped, host writes the seq to wait for
{
    if (threadIdx.x == 0) {
        const uint32_t target = *d_target;
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
}

__global__ void doorbell_wait_copy_f32_ptr_kernel(
    const volatile uint32_t* d_seq,     // mapped, host bumps it after payload
    const volatile uint32_t* d_target,  // mapped, host writes the seq to wait for
    const volatile float* d_payload,    // mapped payload ring, device view
    const volatile int64_t* d_slot_off, // mapped, host writes the slot's float offset
    float* dst,                         // device destination
    int n)                              // float count
{
    if (threadIdx.x == 0) {
        const uint32_t target = *d_target;
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    const int64_t slot_off = *d_slot_off;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = d_payload[slot_off + i];
    }
}

extern "C" void doorbell_wait_seq_ge_ptr(
    void* d_seq,     // mapped seq counter, device view
    void* d_target,  // mapped target cell, device view
    int64_t stream)
{
    doorbell_wait_seq_ge_ptr_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, (const volatile uint32_t*)d_target);
}

extern "C" void doorbell_wait_copy_f32_ptr(
    void* d_seq,      // mapped seq counter, device view
    void* d_target,   // mapped target cell, device view
    void* d_payload,  // mapped payload ring, device view
    void* d_slot_off, // mapped slot-offset cell, device view
    void* dst,        // device f32 destination
    int n,
    int64_t stream)
{
    doorbell_wait_copy_f32_ptr_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, (const volatile uint32_t*)d_target,
        (const volatile float*)d_payload, (const volatile int64_t*)d_slot_off,
        (float*)dst, n);
}

// M3: like doorbell_wait_copy_f32_ptr, but accumulates the published payload into `acc`
// instead of overwriting. The MoE down+aggregate already accumulated the GPU hits into that
// buffer, so one node folds the CPU experts' contribution in — the graph's output is then
// the completed MoE result, and `deferred` no longer needs a separate add.
__global__ void doorbell_wait_add_f32_ptr_kernel(
    const volatile uint32_t* d_seq,     // mapped, host bumps it after payload
    const volatile uint32_t* d_target,  // mapped, host writes the seq to wait for
    const volatile float* d_payload,    // mapped payload ring, device view
    const volatile int64_t* d_slot_off, // mapped, host writes the slot's float offset
    float* acc,                         // device accumulator (read-modify-write)
    int n)                              // float count
{
    if (threadIdx.x == 0) {
        const uint32_t target = *d_target;
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    const int64_t slot_off = *d_slot_off;
    // Each i is handled by exactly one thread, so the read-modify-write needs no atomics.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc[i] += d_payload[slot_off + i];
    }
}

extern "C" void doorbell_wait_add_f32_ptr(
    void* d_seq,      // mapped seq counter, device view
    void* d_target,   // mapped target cell, device view
    void* d_payload,  // mapped payload ring, device view
    void* d_slot_off, // mapped slot-offset cell, device view
    void* acc,        // device f32 accumulator
    int n,
    int64_t stream)
{
    doorbell_wait_add_f32_ptr_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, (const volatile uint32_t*)d_target,
        (const volatile float*)d_payload, (const volatile int64_t*)d_slot_off,
        (float*)acc, n);
}

// M3: zero-argument wait+accumulate whose **job number is claimed on the device** from a
// counter, instead of read from host-written cells. A host-written cell races: the token
// thread runs arbitrarily far ahead of the GPU, so by the time this node executes the host may
// already have written a later step's value (this was measured — it made --pcie-frac 0
// nondeterministic). Deriving the step on the device removes that entirely:
//   target = n, slot = (n % ring_depth) * max_n.
__global__ void doorbell_wait_add_indexed_f32_kernel(
    const volatile uint32_t* d_seq,     // mapped, host bumps it after payload
    uint32_t* d_counter,                // device u32, claimed per job with atomicAdd
    const volatile float* d_payload,    // mapped payload ring, device view
    float* acc,                         // device accumulator (read-modify-write)
    int n,                              // float count
    int ring_depth,                     // ring slots (matches the publisher's modulus)
    int max_n)                          // floats per slot
{
    __shared__ uint32_t job;
    if (threadIdx.x == 0) {
        job = atomicAdd(d_counter, 1u) + 1u;  // 1-based, matching the host's publish_seq
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t target = job;
        while (*d_seq < target) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    const int64_t slot_off =
        (int64_t)(job % (uint32_t)ring_depth) * (int64_t)max_n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc[i] += d_payload[slot_off + i];
    }
}

extern "C" void doorbell_wait_add_indexed_f32(
    void* d_seq,      // mapped seq counter, device view
    void* d_counter,  // device u32 job counter
    void* d_payload,  // mapped payload ring, device view
    void* acc,        // device f32 accumulator
    int n,
    int ring_depth,
    int max_n,
    int64_t stream)
{
    doorbell_wait_add_indexed_f32_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, (uint32_t*)d_counter,
        (const volatile float*)d_payload, (float*)acc, n, ring_depth, max_n);
}

// ---------------------------------------------------------------------------
// v2 §3.2: the device-epoch protocol for CPU -> GPU sync points that must work
// inside a replayed graph.
//
//   epoch    : device-written, host-read   (one bump per pass, in layer 0's graph)
//   progress : host-written, device-read   (monotonic sync points)
//   node     : p = (*epoch) * N + L, spin on progress >= p, then accumulate
//              payload[(p % R) * max_n ..] into acc.
//
// The host can only ever *lag* (it reads a past epoch), which is safe; it can never
// publish for a pass the device has not announced. That removes the whole class of
// "read the next step's data" races that a host-written target has.
// ---------------------------------------------------------------------------
__global__ void doorbell_epoch_bump_kernel(uint32_t* d_epoch) {
    if (threadIdx.x == 0) {
        *d_epoch = *d_epoch + 1u;
        __threadfence_system();
    }
}

extern "C" void doorbell_epoch_bump(void* d_epoch, int64_t stream) {
    doorbell_epoch_bump_kernel<<<1, 1, 0, (cudaStream_t)stream>>>((uint32_t*)d_epoch);
}

__global__ void doorbell_wait_add_ready_kernel(
    const volatile uint32_t* d_epoch,    // device-written, one bump per pass
    const volatile uint32_t* d_ready,    // [K] one word per sync point; value = ready pass
    const volatile float* d_payload,     // [slots][max_n], host-written
    float* acc,                          // device accumulator (read-modify-write)
    int n,                               // float count
    int id,                              // sync point index (constant, baked per node)
    int slot,                            // payload slot (constant, baked per node)
    int max_n)                           // floats per slot (constant)
{
    __shared__ uint32_t e;
    if (threadIdx.x == 0) {
        e = *d_epoch;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (d_ready[id] < e) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    const volatile float* src = d_payload + (int64_t)slot * (int64_t)max_n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc[i] += src[i];
    }
}

extern "C" void doorbell_wait_add_ready(
    void* d_epoch, void* d_ready, void* d_payload, void* acc,
    int n, int id, int slot, int max_n, int64_t stream)
{
    doorbell_wait_add_ready_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_epoch, (const volatile uint32_t*)d_ready,
        (const volatile float*)d_payload, (float*)acc, n, id, slot, max_n);
}

// M4: a sync point that only waits (no payload). Used for the two DMA-done points: the data is
// already in VRAM, written by the copy stream, so the node just has to be ordered after it.
__global__ void doorbell_wait_ready_kernel(
    const volatile uint32_t* d_epoch,
    const volatile uint32_t* d_ready,
    int id)
{
    __shared__ uint32_t e;
    if (threadIdx.x == 0) {
        e = *d_epoch;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (d_ready[id] < e) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);
#endif
        }
        __threadfence_system();
    }
}

extern "C" void doorbell_wait_ready(
    void* d_epoch, void* d_ready, int id, int64_t stream)
{
    doorbell_wait_ready_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_epoch, (const volatile uint32_t*)d_ready, id);
}
