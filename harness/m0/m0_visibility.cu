// naiwa-rs v2 — M0 harness: zero-argument doorbell primitives.
// See docs/roadmap.md (M0) and docs/worklogs/m0.md.
//
// Proves the two properties the v2 architecture rests on:
//
//   (A) host -> GPU visibility. The host publishes a payload slot and its slot
//       offset into mapped pinned memory, then bumps a mapped seq. A stream-ordered
//       kernel spins until *seq >= *target -- where *target is a mapped value, NOT a
//       kernel argument -- fences, then copies the slot into device memory.
//       `iters` iterations, all correct.
//
//   (B) graph replay. The SAME kernel is captured into one CUDA graph. The graph is
//       replayed `replays` times while the host rewrites payload / target / slot_off
//       in mapped memory between replays. Every per-iteration value comes from mapped
//       memory; the kernel parameters are frozen pointers. Anything accidentally baked
//       at capture time would make the per-iteration compare fail.
//
// Target: RTX 4070 (Ada, sm_89). strata reports RDNA needs a volatile store or a
// system fence; Ada is expected to need one fence. The negative-control arms
// (-DM0_NO_DEVICE_FENCE / -DM0_NO_HOST_FENCE) test whether each fence is load-bearing
// on this card -- they are evidence, not a gate.
//
// Build: harness/m0/build.sh
// Run:   ~/bench-rel/naiwa-v2/m0/m0_visibility [iters] [replays]

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define M0_CUDA(call)                                                        \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            std::printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_),  \
                        __FILE__, __LINE__);                                 \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

constexpr int kRing = 8;     // ring depth >= 2: host may write k+1 while the GPU reads k
constexpr int kFloats = 64;  // payload floats per slot (roughly topk*2)
constexpr int kThreads = 128;

// All per-iteration values live in mapped memory. The kernel parameters are fixed
// pointers, so one capture can be replayed indefinitely.
//
// `armed` is a test aid: the kernel raises it as its first action, and the host only
// publishes payload/seq after seeing it. That guarantees the kernel is inside the spin
// loop before the publish, so `waited` proves the live target was honored.
__global__ void wait_copy_ptr(const volatile uint32_t* seq,
                              const volatile uint32_t* target,
                              const volatile int*      slot_off,
                              const float*             payload,
                              float*                   dst,
                              int                      n,
                              volatile int*            armed,
                              volatile int*            waited,
                              volatile int*            timed_out) {
    if (threadIdx.x == 0) {
        *armed = 1;  // host waits for this before publishing
#ifdef M0_BAKE_TARGET
        // Negative control: simulate a target baked at capture time. The kernel then
        // never waits, so `waited` stays 0 and the pass criterion fails.
        const uint32_t tgt = 0;
#else
        const uint32_t tgt = *target;
#endif
        long spins = 0;
        while (*seq < tgt) {
            __nanosleep(200);
            if (++spins > 5000000L) {  // ~1 s bound: fail instead of hanging
                *timed_out = 1;
                break;
            }
        }
        if (spins > 0) {
            *waited = *waited + 1;  // proof the kernel honored the live target
        }
#ifndef M0_NO_DEVICE_FENCE
        __threadfence_system();
#endif
    }
    __syncthreads();
    const int off = *(volatile const int*)slot_off;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = payload[off + i];
    }
}

float expected_f(int k, int i) {
    return static_cast<float>(k * 1000 + i) + 0.5f;
}

void publish(volatile uint32_t* seq, uint32_t v) {
#ifndef M0_NO_HOST_FENCE
    std::atomic_thread_fence(std::memory_order_release);
#endif
    *seq = v;
}

// Spin until the kernel raises `armed` (it is then inside the seq spin loop).
bool await_armed(volatile int* armed) {
    for (long s = 0; s < 500000000L; ++s) {
        if (*armed == 1) return true;
        __builtin_ia32_pause();
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 1000;
    const int replays = argc > 2 ? std::atoi(argv[2]) : 200;

    cudaDeviceProp prop{};
    M0_CUDA(cudaGetDeviceProperties(&prop, 0));
#ifdef M0_NO_HOST_FENCE
    const char* host_f = "no host fence";
#else
    const char* host_f = "host fence";
#endif
#ifdef M0_NO_DEVICE_FENCE
    const char* dev_f = "no device fence";
#else
    const char* dev_f = "device fence";
#endif
#ifdef M0_BAKE_TARGET
    const char* bake = "  [BAKED-TARGET CONTROL: expected to FAIL]";
#else
    const char* bake = "";
#endif
    std::printf("device: %s  sm_%d%d  (%s, %s)%s\n", prop.name, prop.major, prop.minor,
                host_f, dev_f, bake);

    float*             h_payload = nullptr;
    float*             d_payload = nullptr;
    volatile uint32_t* h_seq     = nullptr;
    uint32_t*          d_seq     = nullptr;
    volatile uint32_t* h_target  = nullptr;
    uint32_t*          d_target  = nullptr;
    volatile int*      h_slot_off = nullptr;
    int*               d_slot_off = nullptr;
    volatile int*      h_armed    = nullptr;
    int*               d_armed    = nullptr;
    float*             h_dst     = nullptr;
    float*             d_dst     = nullptr;
    int*               d_waited  = nullptr;
    int*               d_timeout = nullptr;

    M0_CUDA(cudaHostAlloc((void**)&h_payload, (size_t)kRing * kFloats * sizeof(float),
                          cudaHostAllocMapped));
    M0_CUDA(cudaHostAlloc((void**)&h_seq, sizeof(uint32_t), cudaHostAllocMapped));
    M0_CUDA(cudaHostAlloc((void**)&h_target, sizeof(uint32_t), cudaHostAllocMapped));
    M0_CUDA(cudaHostAlloc((void**)&h_slot_off, sizeof(int), cudaHostAllocMapped));
    M0_CUDA(cudaHostAlloc((void**)&h_armed, sizeof(int), cudaHostAllocMapped));
    M0_CUDA(cudaHostGetDevicePointer((void**)&d_payload, (void*)h_payload, 0));
    M0_CUDA(cudaHostGetDevicePointer((void**)&d_seq, (void*)h_seq, 0));
    M0_CUDA(cudaHostGetDevicePointer((void**)&d_target, (void*)h_target, 0));
    M0_CUDA(cudaHostGetDevicePointer((void**)&d_slot_off, (void*)h_slot_off, 0));
    M0_CUDA(cudaHostGetDevicePointer((void**)&d_armed, (void*)h_armed, 0));
    M0_CUDA(cudaMalloc(&d_dst, kFloats * sizeof(float)));
    M0_CUDA(cudaMallocHost(&h_dst, kFloats * sizeof(float)));
    M0_CUDA(cudaMalloc(&d_waited, sizeof(int)));
    M0_CUDA(cudaMalloc(&d_timeout, sizeof(int)));

    cudaStream_t stream = nullptr;
    M0_CUDA(cudaStreamCreate(&stream));

    *h_seq = 0;
    *h_target = 0;
    *h_slot_off = 0;
    *h_armed = 0;
    std::memset(h_payload, 0, (size_t)kRing * kFloats * sizeof(float));
    M0_CUDA(cudaMemset(d_waited, 0, sizeof(int)));
    M0_CUDA(cudaMemset(d_timeout, 0, sizeof(int)));

    // ---------------------------------------------------------------- A: visibility
    std::printf("== A: host->GPU visibility, %d iters, ring %d ==\n", iters, kRing);
    int a_bad = 0, a_to = 0, a_waited = 0;
    const auto t_a0 = std::chrono::steady_clock::now();
    for (int k = 1; k <= iters; ++k) {
        const int slot = (k - 1) % kRing;

        // Announce the target, then launch. seq is still k-1, so the kernel spins.
        *h_target = (uint32_t)k;
        *h_armed = 0;
        std::atomic_thread_fence(std::memory_order_release);
        wait_copy_ptr<<<1, kThreads, 0, stream>>>(d_seq, d_target, d_slot_off, d_payload,
                                                  d_dst, kFloats, d_armed, d_waited,
                                                  d_timeout);
        if (!await_armed(h_armed)) {
            ++a_to;
            if (a_to < 4) std::printf("  iter %d: kernel never armed\n", k);
            M0_CUDA(cudaStreamSynchronize(stream));
            continue;
        }

        // Publish the payload while the kernel is provably spinning.
        float* slotp = h_payload + (size_t)slot * kFloats;
        for (int i = 0; i < kFloats; ++i) {
            slotp[i] = expected_f(k, i);
        }
        *h_slot_off = slot * kFloats;
        publish(h_seq, (uint32_t)k);

        M0_CUDA(cudaStreamSynchronize(stream));
        M0_CUDA(cudaMemcpy(h_dst, d_dst, kFloats * sizeof(float), cudaMemcpyDeviceToHost));
        int to = 0, waited = 0;
        M0_CUDA(cudaMemcpy(&to, d_timeout, sizeof(int), cudaMemcpyDeviceToHost));
        M0_CUDA(cudaMemcpy(&waited, d_waited, sizeof(int), cudaMemcpyDeviceToHost));
        if (to) {
            ++a_to;
            if (a_to < 4) std::printf("  iter %d: kernel spin timeout\n", k);
            continue;
        }
        a_waited = waited;
        for (int i = 0; i < kFloats; ++i) {
            if (h_dst[i] != expected_f(k, i)) {
                ++a_bad;
                if (a_bad < 4) {
                    std::printf("  iter %d: dst[%d]=%g want %g\n", k, i, h_dst[i],
                                expected_f(k, i));
                }
                break;
            }
        }
    }
    const double a_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_a0)
            .count();
    std::printf("A: %d/%d correct, %d timeout, kernel waited %d/%d, %.1f ms (%.1f us/iter)\n",
                iters - a_bad, iters, a_to, a_waited, iters, a_ms, a_ms * 1000.0 / iters);

    // ------------------------------------------------------------- B: graph replay
    std::printf("== B: graph replay, %d replays ==\n", replays);
    *h_target = 0;
    *h_seq = 0;
    *h_armed = 0;
    M0_CUDA(cudaMemset(d_waited, 0, sizeof(int)));

    M0_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    wait_copy_ptr<<<1, kThreads, 0, stream>>>(d_seq, d_target, d_slot_off, d_payload, d_dst,
                                              kFloats, d_armed, d_waited, d_timeout);
    cudaGraph_t graph = nullptr;
    M0_CUDA(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    M0_CUDA(cudaGraphInstantiateWithFlags(&exec, graph, 0));

    const uint32_t base = (uint32_t)iters;  // continue the seq counter from A
    int b_bad = 0, b_to = 0, b_waited = 0;
    const auto t_b0 = std::chrono::steady_clock::now();
    for (int k = 1; k <= replays; ++k) {
        const int slot = (k - 1) % kRing;

        *h_target = base + (uint32_t)k;  // the live target the replay must honor
        *h_armed = 0;
        std::atomic_thread_fence(std::memory_order_release);
        M0_CUDA(cudaGraphLaunch(exec, stream));
        if (!await_armed(h_armed)) {
            ++b_to;
            if (b_to < 4) std::printf("  replay %d: kernel never armed\n", k);
            M0_CUDA(cudaStreamSynchronize(stream));
            continue;
        }

        float* slotp = h_payload + (size_t)slot * kFloats;
        for (int i = 0; i < kFloats; ++i) {
            slotp[i] = expected_f(1000 + k, i);
        }
        *h_slot_off = slot * kFloats;
        publish(h_seq, base + (uint32_t)k);

        M0_CUDA(cudaStreamSynchronize(stream));
        M0_CUDA(cudaMemcpy(h_dst, d_dst, kFloats * sizeof(float), cudaMemcpyDeviceToHost));
        int to = 0, waited = 0;
        M0_CUDA(cudaMemcpy(&to, d_timeout, sizeof(int), cudaMemcpyDeviceToHost));
        M0_CUDA(cudaMemcpy(&waited, d_waited, sizeof(int), cudaMemcpyDeviceToHost));
        if (to) {
            ++b_to;
            if (b_to < 4) std::printf("  replay %d: kernel spin timeout\n", k);
            continue;
        }
        b_waited = waited;
        for (int i = 0; i < kFloats; ++i) {
            if (h_dst[i] != expected_f(1000 + k, i)) {
                ++b_bad;
                if (b_bad < 4) {
                    std::printf("  replay %d: dst[%d]=%g want %g\n", k, i, h_dst[i],
                                expected_f(1000 + k, i));
                }
                break;
            }
        }
    }
    const double b_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_b0)
            .count();
    std::printf("B: %d/%d correct, %d timeout, kernel waited %d/%d, %.1f ms (%.1f us/replay)\n",
                replays - b_bad, replays, b_to, b_waited, replays, b_ms,
                b_ms * 1000.0 / replays);

    const bool pass = (a_bad == 0 && a_to == 0 && a_waited == iters) &&
                      (b_bad == 0 && b_to == 0 && b_waited == replays);
    std::printf("result: %s\n", pass ? "PASS" : "FAIL");

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_dst);
    cudaFree(d_waited);
    cudaFree(d_timeout);
    cudaFreeHost(h_dst);
    cudaFreeHost((void*)h_payload);
    cudaFreeHost((void*)h_seq);
    cudaFreeHost((void*)h_target);
    cudaFreeHost((void*)h_slot_off);
    cudaFreeHost((void*)h_armed);
    return pass ? 0 : 1;
}
