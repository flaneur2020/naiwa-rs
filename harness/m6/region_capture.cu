// M6 harness: does a captured *region* replayed with fresh host-written inputs produce exactly what a
// fresh eager run of the same kernels produces?
//
// This is the whole assumption behind M6 step 2 (per-layer attention segments): the region is captured
// once into a graph whose buffers live in the capture-time pool, and every step the host copies the new
// input into the *same* fixed buffer before replaying. If that is not bit-exact, the plan is wrong and
// no amount of fork plumbing will save it.
//
// Three kernels, deliberately the shape of an attention region: a gemv-shaped matvec, a scale+add, and
// an elementwise square (a stand-in for the norm/mix/inject chain).
//
// Build/run: ./run.sh
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x)                                                                       \
    do {                                                                            \
        cudaError_t e = (x);                                                        \
        if (e != cudaSuccess) {                                                     \
            printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

constexpr int N = 4096;   // rows
constexpr int K = 256;    // reduction

// y[i] = sum_k w[i*K+k] * x[k]
__global__ void gemv_like(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += w[i * K + k] * x[k];
    y[i] = acc;
}

__global__ void scale_add(const float* __restrict__ src, float* __restrict__ dst, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[i] * a + b;
}

__global__ void square(const float* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[i] * src[i];
}

// Every launch must go to the stream being captured: a launch on the legacy default stream while a
// capture is open is itself an error ("previous error during capture").
static void launch_region(cudaStream_t stream, const float* w, const float* x, float* y, float* z,
                          float* out) {
    gemv_like<<<(N + 127) / 128, 128, 0, stream>>>(w, x, y);
    scale_add<<<(N + 127) / 128, 128, 0, stream>>>(y, z, 1.5f, 0.25f);
    square<<<(N + 127) / 128, 128, 0, stream>>>(z, out);
}

int main() {
    float *w, *x, *y, *z, *out;
    CK(cudaMalloc(&w, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&x, K * sizeof(float)));
    CK(cudaMalloc(&y, N * sizeof(float)));
    CK(cudaMalloc(&z, N * sizeof(float)));
    CK(cudaMalloc(&out, N * sizeof(float)));

    std::vector<float> hw((size_t)N * K), hx(K);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = std::sin((float)i * 0.001f) * 0.5f;
    CK(cudaMemcpy(w, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Host-side trial inputs: 4 "steps", each a different vector.
    auto make_x = [](int trial) {
        std::vector<float> v(K);
        for (int k = 0; k < K; ++k) v[k] = std::cos((float)(k + 1) * (float)(trial + 1) * 0.01f);
        return v;
    };

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    const int trials = 4;
    // ---- eager reference, per trial, into a private buffer ----
    std::vector<std::vector<float>> ref(trials);
    {
        std::vector<float> hx_t(K), hout(N);
        for (int t = 0; t < trials; ++t) {
            auto v = make_x(t);
            CK(cudaMemcpy(x, v.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            launch_region(stream, w, x, y, z, out);
            CK(cudaStreamSynchronize(stream));
            CK(cudaMemcpy(hout.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost));
            ref[t] = hout;
        }
    }

    // ---- capture the region once ----
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    launch_region(stream, w, x, y, z, out);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));

    size_t nodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nodes));
    printf("captured nodes: %zu (expected 3)\n", nodes);

    // ---- replay with fresh inputs copied into the SAME buffer ----
    int bad = 0;
    std::vector<float> hout(N);
    for (int t = 0; t < trials; ++t) {
        auto v = make_x(t);
        CK(cudaMemcpy(x, v.data(), K * sizeof(float), cudaMemcpyHostToDevice));  // the copy-in
        CK(cudaGraphLaunch(exec, stream));
        CK(cudaMemcpy(hout.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            if (hout[i] != ref[t][i]) {  // bit-exact, not approximate
                printf("MISMATCH trial %d at %d: %a vs %a\n", t, i, hout[i], ref[t][i]);
                ++bad;
                break;
            }
        }
    }
    printf("replays: %d/%d bit-exact\n", trials - bad, trials);

    // ---- control: the output must TRACK the buffer we write ----
    // Replaying with the last trial's input still in place (no copy-in) must reproduce that trial, and
    // replaying after poising the buffer with trial 0 must reproduce trial 0 *and* differ from trial
    // 3's result. If either half fails, the graph is not reading the buffer we claim to update and the
    // bit-exactness above proved nothing.
    CK(cudaGraphLaunch(exec, stream));
    CK(cudaMemcpy(hout.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool same_as_last = true;
    for (int i = 0; i < N; ++i)
        if (hout[i] != ref[trials - 1][i]) { same_as_last = false; break; }

    std::vector<float> hout0(N);
    auto v0 = make_x(0);
    CK(cudaMemcpy(x, v0.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, stream));
    CK(cudaMemcpy(hout0.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool matches_trial0 = true, differs_from_last = false;
    for (int i = 0; i < N; ++i) {
        if (hout0[i] != ref[0][i]) matches_trial0 = false;
        if (hout0[i] != ref[trials - 1][i]) differs_from_last = true;
    }
    printf("control: stale replay == last trial: %s; re-armed replay == trial 0: %s; differs from last: %s\n",
           same_as_last ? "yes" : "NO", matches_trial0 ? "yes" : "NO", differs_from_last ? "yes" : "NO");
    bool control_ok = same_as_last && matches_trial0 && differs_from_last;

    bool ok = (nodes == 3) && bad == 0 && control_ok;
    printf("%s\n", ok ? "M6 region-capture harness: PASS" : "M6 region-capture harness: FAIL");
    return ok ? 0 : 1;
}
