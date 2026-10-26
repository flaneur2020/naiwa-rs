// Isolated A/B for M22.4 §5's aligned repack: same vec_dot, same weights, same codebook, same q8 activation.
// Only the block layout and therefore the load width changes (2-byte uint16 loads vs 4-byte uint32 loads).
#include "cuda_fp16.h"
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <random>

#define QK_K 256
#define QK8_0 32
#define MIN_CC_DP4A 610

typedef struct { half2 ds; int8_t qs[QK8_0]; } block_q8_1;

// ---- today's layout: 98 bytes, `d` first, so `qs` sits at +2 and is 2-byte aligned
typedef struct { half d; uint8_t qs[3 * QK_K / 8]; } block98;
// ---- repacked: 100 bytes, `qs` first (4-byte aligned), `d` at +96, 2 bytes of padding
typedef struct { uint8_t qs[3 * QK_K / 8]; half d; uint8_t pad[2]; } block100;
static_assert(sizeof(block98) == 98, "");
static_assert(sizeof(block100) == 100, "");

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
  return __dp4a(a, b, c);
}
static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}
// today: 2-byte loads out of necessity (qs base is only 2-byte aligned)
static __device__ __forceinline__ int iq3_get_int_b2(const void *x, const int &i32) {
  const uint16_t *x16 = (const uint16_t *)x;
  return (x16[2 * i32 + 0] << 0) | (x16[2 * i32 + 1] << 16);
}
// repacked: one 4-byte load
static __device__ __forceinline__ int iq3_get_int_b4(const void *x, const int &i32) {
  return ((const int *)x)[i32];
}
// calibration: what if we make the loads *narrower* (4 x 1-byte per int)? If the bench is sensitive to load
// width at all, this must be visibly slower than today's 2-byte variant.
static __device__ __forceinline__ int iq3_get_int_b1(const void *x, const int &i32) {
  const uint8_t *x8 = (const uint8_t *)x + 4 * i32;
  return (int)x8[0] | ((int)x8[1] << 8) | ((int)x8[2] << 16) | ((int)x8[3] << 24);
}
static __device__ __forceinline__ uint32_t iq3_unpack_ksigns(const uint8_t v) {
  const uint32_t p = __popc(v) & 1;
  return (v ^ (p << 7)) * 0x01010101u;
}

#include "iq3xxs_grid.cuh"

static __device__ __forceinline__ float dot_body(const uint8_t *qs, const block_q8_1 *bq8_1, const int &iqs,
                                                int i0, int i1, uint32_t aux32_in, float d) {
  const int2 q3_packed = make_int2(i0, i1);
  const uint8_t *q3 = (const uint8_t *)&q3_packed;
  int sumi = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
    const uint32_t signs = iq3_unpack_ksigns(aux32_in >> (7 * l0 / 2));
    const int signs0 = __vcmpne4(signs & 0x08040201, 0);
    const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
    const int u0 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 0);
    const int signs1 = __vcmpne4(signs & 0x80402010, 0);
    const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
    const int u1 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 1);
    sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
    sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
  }
  const int ls = aux32_in >> 28;
  sumi = (ls * sumi + sumi / 2) / 2;
  return d * sumi;
}

__device__ __forceinline__ float vec_dot_today(const void *vbq, const block_q8_1 *bq8_1, const int &iqs) {
  const block98 *bq3 = (const block98 *)vbq;
  const uint32_t aux32 = iq3_get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
  const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
  return dot_body(bq3->qs, bq8_1, iqs, iq3_get_int_b2(bq3->qs, iqs), iq3_get_int_b2(bq3->qs, iqs + 1), aux32, d);
}

__device__ __forceinline__ float vec_dot_repacked(const void *vbq, const block_q8_1 *bq8_1, const int &iqs) {
  const block100 *bq3 = (const block100 *)vbq;
  const uint32_t aux32 = iq3_get_int_b4(bq3->qs, QK_K / 16 + iqs / 2);
  const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
  return dot_body(bq3->qs, bq8_1, iqs, iq3_get_int_b4(bq3->qs, iqs), iq3_get_int_b4(bq3->qs, iqs + 1), aux32, d);
}

__device__ __forceinline__ float vec_dot_narrow(const void *vbq, const block_q8_1 *bq8_1, const int &iqs) {
  const block98 *bq3 = (const block98 *)vbq;
  const uint32_t aux32 = (uint32_t)iq3_get_int_b1(bq3->qs, QK_K / 16 + iqs / 2);
  const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
  return dot_body(bq3->qs, bq8_1, iqs, iq3_get_int_b1(bq3->qs, iqs), iq3_get_int_b1(bq3->qs, iqs + 1), aux32, d);
}

// One thread walks whole super-blocks, exactly as the MoE kernel's warp loop does per (row, block) pair.
template <int LAYOUT>
__global__ void dot_kernel(const void *__restrict__ w, const block_q8_1 *__restrict__ q8, long nblk,
                           int stride_bytes, float *__restrict__ out) {
  const char *base = (const char *)w;
  float acc = 0.f;
  for (long b = blockIdx.x * (long)blockDim.x + threadIdx.x; b < nblk; b += (long)gridDim.x * blockDim.x) {
    const void *blk = base + b * stride_bytes;
#pragma unroll
    for (int iqs = 0; iqs < 16; iqs += 2) {
      if (LAYOUT == 0) acc += vec_dot_today(blk, q8, iqs);
      else if (LAYOUT == 1) acc += vec_dot_repacked(blk, q8, iqs);
      else acc += vec_dot_narrow(blk, q8, iqs);
    }
  }
  if (acc == 1.2345e30f) out[0] = acc;  // keep the loop alive; never true
}

static float run(int layout, const void *w, int stride, long nblk, const block_q8_1 *q8, int iters) {
  float *out; cudaMalloc(&out, 4);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int threads = 256, blocks = 132 * 2;
  // warm up
  if (layout == 0) dot_kernel<0><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
  else if (layout == 1) dot_kernel<1><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
  else dot_kernel<2><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for (int i = 0; i < iters; i++) {
    if (layout == 0) dot_kernel<0><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
    else if (layout == 1) dot_kernel<1><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
    else dot_kernel<2><<<blocks, threads>>>((const void *)w, q8, nblk, stride, out);
  }
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaFree(out);
  return ms / iters;
}

// Build & run:  nvcc -O3 -std=c++17 -arch=sm_89 iq3_repack_ab.cu -o /tmp/iq3_ab && /tmp/iq3_ab
// Answers M22.4 §5's weak link: does halving the load-instruction count (2 B -> 4 B per load, the aligned
// repack) actually halve the load *time*? Three arms, same codebook, same weights, same q8 activation:
//   today     98 B block, `d` first, qs at +2  -> 4 x uint16 per 8 B (2 B per load instruction)
//   repacked 100 B block, qs first at +0      -> 2 x uint32 per 8 B (4 B per load instruction)
//   narrow    98 B block, 1-byte loads        -> calibration: is the bench sensitive to load width at all?
int run_one(long nblk) {
  std::vector<uint8_t> src(nblk * 96);
  std::mt19937 rng(0);
  for (auto &v : src) v = (uint8_t)rng();
  std::vector<block98> w98(nblk);
  std::vector<block100> w100(nblk);
  for (long i = 0; i < nblk; i++) {
    w98[i].d = __float2half(0.01f);
    memcpy(w98[i].qs, &src[i * 96], 96);
    w100[i].d = __float2half(0.01f);
    memcpy(w100[i].qs, &src[i * 96], 96);
  }
  block_q8_1 q8[8];
  for (int i = 0; i < 8; i++) { q8[i].ds = __floats2half2_rn(0.02f, 0.f); for (auto &v : q8[i].qs) v = (int8_t)(rng() % 15) - 7; }

  block98 *d98; block100 *d100; block_q8_1 *dq8;
  cudaMalloc(&d98, sizeof(block98) * nblk); cudaMalloc(&d100, sizeof(block100) * nblk);
  cudaMalloc(&dq8, sizeof(block_q8_1) * 8);
  cudaMemcpy(d98, w98.data(), sizeof(block98) * nblk, cudaMemcpyHostToDevice);
  cudaMemcpy(d100, w100.data(), sizeof(block100) * nblk, cudaMemcpyHostToDevice);
  cudaMemcpy(dq8, q8, sizeof(block_q8_1) * 8, cudaMemcpyHostToDevice);

  printf("working set: today %.1f MB / repacked %.1f MB   (%ld super-blocks)\n",
         nblk * 98.0 / 1e6, nblk * 100.0 / 1e6, nblk);
  run(0, d98, 98, nblk, dq8, 1); run(1, d100, 100, nblk, dq8, 1); run(2, d98, 98, nblk, dq8, 1);
  for (int rep = 0; rep < 3; rep++) {
    float t0 = run(0, d98, 98, nblk, dq8, 20);
    float t1 = run(1, d100, 100, nblk, dq8, 20);
    float t2 = run(2, d98, 98, nblk, dq8, 20);
    printf("  rep%d  today(2B) %7.2f ms (%5.1f GB/s)   repacked(4B) %7.2f ms (%5.1f GB/s) = %.3fx   "
           "narrow(1B) %7.2f ms = %.3fx vs today\n",
           rep, t0, nblk * 98.0 / t0 / 1e6, t1, nblk * 100.0 / t1 / 1e6, t0 / t1,
           t2, t2 / t0);
  }
  cudaFree(d98); cudaFree(d100); cudaFree(dq8);
  return 0;
}

int main() {
  // (a) 19.6 MB: lives in L2 (36 MB on this part) -> the loop is issue-bound, which is the regime the real kernel
  //     is in at 82 GB/s (16% of peak).
  run_one(200000);
  // (b) 160 MB: far beyond L2 -> DRAM-bound, the pure streaming case.
  run_one(6400L * 256L);
  return 0;
}
