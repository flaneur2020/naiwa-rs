// Doorbell kernels: mapped pinned memory the host and the device hand each other data through.
//
// The pattern is the one the harness proved (~/bench-rel/doorbell/harness.cu, 1000/1000 on Ada):
// the writer stores the payload, __threadfence_system(), then a volatile store bumps the counter;
// the reader spins on the counter and reads the payload behind it.
//
// There are no gate words and no epoch here. Both barriers in the decode path are CUDA events
// (`crate::copy_stream::CopyDone`), so the only device-side spins left are the two below.
#include <cuda_runtime.h>
#include <stdint.h>

// Host -> device: publish this layer's top-k into mapped memory so the host read needs no stream
// sync and no D2H launch. One block. Stream-ordered after the router kernel that produced the
// tensors.
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
// Payload ring: the same protocol over a ring of f32 slots.
//   device -> host: doorbell_copy_f32 bumps seq after copying a slot (the lookahead activation).
//   host -> device: the host writes a slot and bumps seq; doorbell_wait_copy_f32 spins until it
//                   reaches the target, copies the slot out, then publishes `consumed`.
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

// Host -> device result leg: the host writes a payload slot of a mapped ring and bumps seq; this
// kernel waits for seq >= target, copies the slot into device memory, and then publishes
// `consumed = target` so the host can check its ring depth instead of assuming it. The slot stride
// keeps a slow GPU reading slot k while the host writes k+1 (depth >= 2).
//
// The spin is unbounded on purpose: the producer is a host thread that always publishes, and a
// silently dropped payload would corrupt the model, which is worse than a hang.
__global__ void doorbell_wait_copy_f32_kernel(
    const volatile uint32_t* d_seq,   // mapped, host bumps it after the payload
    uint32_t target,                  // seq value that means "slot complete"
    const volatile float* d_payload,  // mapped payload ring, device view
    int64_t slot_off,                 // float offset of this publish's slot
    volatile uint32_t* d_consumed,    // mapped: set to `target` once the slot was read out
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
    __syncthreads();
    if (threadIdx.x == 0) {
        // Every read of the slot has retired; the host may overwrite it.
        __threadfence_system();
        *d_consumed = target;
    }
}

extern "C" void doorbell_wait_copy_f32(
    void* d_seq,           // mapped seq counter, device view
    uint32_t target,       // seq value that means "payload slot complete"
    void* d_payload,       // mapped payload ring, device view
    int64_t slot_off,      // float offset of the slot to read
    void* d_consumed,      // mapped word set once the copy is done
    void* dst,             // device f32 destination, stream-ordered afterwards
    int n,                 // float count
    int64_t stream)
{
    doorbell_wait_copy_f32_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const volatile uint32_t*)d_seq, target,
        (const volatile float*)d_payload, slot_off,
        (volatile uint32_t*)d_consumed,
        (float*)dst, n);
}
