// Qwen4-Exp (Qwen3.8-Flash-Next) kernels: hyper-connections, hashed n-gram PLE
// and QSA sparse attention.
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define Q4_PAD_SLOT 0xffffffffu
#define Q4_MAX_PLE_NGRAM 8
#define Q4_MAX_PLE_HEADS 64
#define Q4_MAX_CONV_STATE 16
#define Q4_QSA_MAX_INDEX_HEADS 8
#define Q4_QSA_HEAD_DIM 128
#define Q4_ATTN_HEAD_DIM 256
#define Q4_ATTN_WARPS 4
#define Q4_ATTN_MAX_HEADS_PER_WARP 4
#define Q4_ATTN_ITEMS 4

// The prompt attention kernel works on a 16-row tile (the GQA group is 12 heads, the mma is 16 rows)
// and stages its Q/K/V rows in shared memory with this stride: 256 dims plus 8 halves of padding so
// every row stays 16-byte aligned for the vector copies the staging does.
// Room for the padded head rows the MMA needs (the GQA group is 12, the tile is 16).
#define Q4_ATTN_MMA_ROWS 16
#define Q4_ATTN_QSTRIDE (Q4_ATTN_HEAD_DIM + 8)

// One 16x8x16 tile, f32 accumulate. A is 16 rows x 16 k (heads x dims), B is 16 k x 8 n (items x
// dims, i.e. column-major for the instruction's `.col`), C is 16 rows x 8 items.
template <typename T>
__device__ __forceinline__ void q4_mma16x8x16(float c[4], const unsigned int a[4],
                                              const unsigned int b[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same<T, __half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
#else
  (void)c; (void)a; (void)b;
#endif
}

// Two T as one 32-bit register, low element first - the order a fragment expects.
template <typename T> __device__ __forceinline__ unsigned int q4_pack2t(T x, T y) {
  unsigned int lo = 0, hi = 0;
  memcpy(&lo, &x, 2);
  memcpy(&hi, &y, 2);
  return lo | (hi << 16);
}

template <typename T> __device__ __forceinline__ float q4_f(T v) {
  return (float)v;
}
template <typename T> __device__ __forceinline__ T q4_t(float v);
template <> __device__ __forceinline__ float q4_t<float>(float v) { return v; }
template <> __device__ __forceinline__ __half q4_t<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 q4_t<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float q4_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float q4_silu(float x) { return x * q4_sigmoid(x); }

__device__ __forceinline__ float q4_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, o);
  }
  return v;
}

// All threads must call; blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float q4_block_sum(float v, float *smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  v = q4_warp_sum(v);
  __syncthreads();
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  v = threadIdx.x < warps ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = q4_warp_sum(v);
    if (lane == 0) {
      smem[0] = v;
    }
  }
  __syncthreads();
  return smem[0];
}

template <typename T> struct Q4Vec8 {
  T v[8];
};

template <typename T>
__device__ __forceinline__ void q4_load8(const T *__restrict__ p, float *out) {
  const Q4Vec8<T> vec = *reinterpret_cast<const Q4Vec8<T> *>(p);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = q4_f(vec.v[i]);
  }
}

// Flattened token -> logical sequence mapping. A null tok_seq is the
// rectangular layout: every sequence has q_len consecutive new tokens (decode
// is q_len 1). tok_seq < 0 marks padding.
struct Q4Tokens {
  const int *tok_seq;
  const int *tok_local;
  const int *seq_start;
  const int *seq_len;
  int n_tokens;
  int n_seqs;
  int q_len;
};

__device__ __forceinline__ int q4_seq_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_seq[t] : t / l.q_len;
}
__device__ __forceinline__ int q4_local_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_local[t] : t % l.q_len;
}
__device__ __forceinline__ int q4_start_of(const Q4Tokens &l, int seq) {
  return l.seq_start ? l.seq_start[seq] : seq * l.q_len;
}
__device__ __forceinline__ int q4_len_of(const Q4Tokens &l, int seq) {
  return l.seq_len ? l.seq_len[seq] : l.q_len;
}

template <typename T>
static __global__ void
q4_hc_norm_kernel(const T *__restrict__ x, const float *__restrict__ weight,
                  T *__restrict__ out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const int stream = row % hc;
  const T *xr = x + row * hidden;
  const float *wr = weight + (size_t)stream * hidden;
  T *orow = out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = q4_t<T>(q4_f(xr[i]) * inv * wr[i]);
  }
}

template <typename T>
static __global__ void
q4_hc_mix_kernel(const T *__restrict__ xn, const T *__restrict__ gate,
                 T *__restrict__ out, size_t total, int hc, int hidden) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const size_t t = idx / hidden;
  const int i = idx - t * hidden;
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    const size_t o = (t * hc + c) * hidden + i;
    acc = fmaf(q4_f(xn[o]), q4_sigmoid(q4_f(gate[o])), acc);
  }
  out[idx] = q4_t<T>(acc / (float)hc);
}

// res_out[t, c] = res[t, c] + block_out[t] * 2 * sigmoid(inject[t, c] / hc);
// when next_weight is set, xn_out is the grouped RMSNorm of res_out for the
// next mixer.
template <typename T>
static __global__ void
q4_hc_combine_kernel(const T *__restrict__ res, const T *__restrict__ block_out,
                     const T *__restrict__ inject, T *__restrict__ res_out,
                     const float *__restrict__ next_weight,
                     T *__restrict__ xn_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const size_t t = row / hc;
  const int stream = row - t * hc;
  const float w = 2.0f * q4_sigmoid(q4_f(inject[row]) / (float)hc);
  const T *rr = res + row * hidden;
  const T *br = block_out + t * hidden;
  T *orow = res_out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const T v = q4_t<T>(fmaf(q4_f(br[i]), w, q4_f(rr[i])));
    orow[i] = v;
    const float vf = q4_f(v);
    ss = fmaf(vf, vf, ss);
  }
  if (next_weight == nullptr) {
    return;
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  const float *wr = next_weight + (size_t)stream * hidden;
  T *xr = xn_out + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    xr[i] = q4_t<T>(q4_f(orow[i]) * inv * wr[i]);
  }
}

struct Q4PleHashParams {
  unsigned long long multipliers[Q4_MAX_PLE_NGRAM];
  unsigned long long vocab_sizes[Q4_MAX_PLE_HEADS];
  unsigned long long offsets[Q4_MAX_PLE_HEADS];
  int ngram_size;
  int heads_per_ngram;
  int num_heads;
  unsigned int eos;
};

// hist_pool rows hold the last `hist_len` token ids of each sequence, oldest
// first, stored as id + 1 so a zeroed slot reads as "no token" (EOS).
static __global__ void q4_ple_hash_kernel(const unsigned int *__restrict__ tokens,
                                   Q4Tokens l,
                                   const float *__restrict__ hist_pool,
                                   const unsigned int *__restrict__ slots,
                                   int hist_len, Q4PleHashParams p,
                                   long long *__restrict__ rows_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= l.n_tokens) {
    return;
  }
  long long *out = rows_out + (size_t)t * p.num_heads;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    for (int h = 0; h < p.num_heads; ++h) {
      out[h] = (long long)p.offsets[h];
    }
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  unsigned long long ctx[Q4_MAX_PLE_NGRAM];
  ctx[0] = tokens[t];
  bool cut = false;
  for (int s = 1; s < p.ngram_size; ++s) {
    const int j = local - s;
    bool none = false;
    unsigned int tok = p.eos;
    if (j >= 0) {
      tok = tokens[t - s];
    } else {
      const int h = hist_len + j;
      if (h < 0 || slot == Q4_PAD_SLOT) {
        none = true;
      } else {
        const float stored = hist_pool[(size_t)slot * hist_len + h];
        if (stored <= 0.0f) {
          none = true;
        } else {
          tok = (unsigned int)stored - 1u;
        }
      }
    }
    cut = cut || none || tok == p.eos;
    ctx[s] = cut ? p.eos : tok;
  }
  for (int n = 2; n <= p.ngram_size; ++n) {
    unsigned long long mixed = ctx[0] * p.multipliers[0];
    for (int j = 1; j < n; ++j) {
      mixed ^= ctx[j] * p.multipliers[j];
    }
    const int base = (n - 2) * p.heads_per_ngram;
    for (int g = 0; g < p.heads_per_ngram; ++g) {
      const int h = base + g;
      out[h] = (long long)(mixed % p.vocab_sizes[h] + p.offsets[h]);
    }
  }
}

// New history = last hist_len tokens of (history ++ this chunk), per sequence.
static __global__ void
q4_ple_hist_update_kernel(const unsigned int *__restrict__ tokens, Q4Tokens l,
                          float *__restrict__ hist_pool,
                          const unsigned int *__restrict__ slots,
                          int hist_len) {
  const int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= l.n_seqs) {
    return;
  }
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  float *h = hist_pool + (size_t)slot * hist_len;
  float next[Q4_MAX_PLE_NGRAM];
  for (int i = 0; i < hist_len; ++i) {
    const int e = len + i;
    next[i] = e < hist_len ? h[e] : (float)tokens[start + e - hist_len] + 1.0f;
  }
  for (int i = 0; i < hist_len; ++i) {
    h[i] = next[i];
  }
}

// One warp per (token, head): copy a 16-bit row of the n-gram table. The shards
// may be host memory the GPU reads through ATS/HMM.
static __global__ void
q4_ple_gather_kernel(const long long *__restrict__ rows, int n_lookups,
                     const unsigned long long *__restrict__ shard_ptrs,
                     const long long *__restrict__ shard_starts, int n_shards,
                     int head_dim, unsigned short *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const long long r = rows[lookup];
  int lo = 0;
  int hi = n_shards - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (shard_starts[mid] <= r) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const unsigned short *src =
      reinterpret_cast<const unsigned short *>(shard_ptrs[lo]) +
      (size_t)(r - shard_starts[lo]) * head_dim;
  unsigned short *dst = out + (size_t)lookup * head_dim;
  for (int i = lane; i < head_dim; i += 32) {
    dst[i] = src[i];
  }
}

// Gather from the device-resident quantized table. Q8/Q4: per row, symmetric
// int8 or offset-8 nibbles with one f16 scale per Q4_PLE_GROUP values in
// `scales`. IQ4_NL: ggml block_iq4_nl rows (f16 scale + 16 nibble bytes per
// Q4_PLE_GROUP values, low nibbles first) straight from a GGUF; `scales` unused.
#define Q4_PLE_GROUP 32
#define Q4_PLE_FMT_Q8 0
#define Q4_PLE_FMT_Q4 1
#define Q4_PLE_FMT_IQ4_NL 2
#define Q4_IQ4_NL_BLOCK_BYTES 18
__constant__ signed char q4_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static __global__ void
q4_ple_gather_quant_kernel(const long long *__restrict__ rows, int n_lookups,
                           const unsigned char *__restrict__ data,
                           const __half *__restrict__ scales, int head_dim,
                           int format, __nv_bfloat16 *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const size_t r = (size_t)rows[lookup];
  const int groups = head_dim / Q4_PLE_GROUP;
  __nv_bfloat16 *dst = out + (size_t)lookup * head_dim;
  if (format == Q4_PLE_FMT_IQ4_NL) {
    const unsigned char *src =
        data + r * (size_t)groups * Q4_IQ4_NL_BLOCK_BYTES;
    for (int i = lane; i < head_dim; i += 32) {
      const unsigned char *blk =
          src + (size_t)(i / Q4_PLE_GROUP) * Q4_IQ4_NL_BLOCK_BYTES;
      const int j = i % Q4_PLE_GROUP;
      const unsigned char byte = blk[2 + (j & 15)];
      const int idx = j < 16 ? (byte & 15) : (byte >> 4);
      const float d = __half2float(*reinterpret_cast<const __half *>(blk));
      dst[i] = __float2bfloat16_rn(d * (float)q4_kvalues_iq4nl[idx]);
    }
    return;
  }
  const int bits = format == Q4_PLE_FMT_Q8 ? 8 : 4;
  const size_t row_bytes = (size_t)head_dim * bits / 8;
  const unsigned char *src = data + r * row_bytes;
  const __half *sc = scales + r * groups;
  for (int i = lane; i < head_dim; i += 32) {
    int q;
    if (bits == 8) {
      q = (int)(signed char)src[i];
    } else {
      const unsigned char byte = src[i >> 1];
      q = (int)((i & 1) ? (byte >> 4) : (byte & 15)) - 8;
    }
    dst[i] = __float2bfloat16_rn((float)q * __half2float(sc[i / Q4_PLE_GROUP]));
  }
}

// Per token: gate each hyper-connection stream by the normalized key/query dot
// product, scale the shared value, and RMS-normalize the gated value for the
// conv. One block per token.
template <typename T>
static __global__ void
q4_ple_gate_kernel(const T *__restrict__ key,
                   const T *__restrict__ hidden_states,
                   const T *__restrict__ value, const float *__restrict__ w_key,
                   const float *__restrict__ w_query,
                   const float *__restrict__ w_conv, T *__restrict__ gated_out,
                   T *__restrict__ normed_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  __shared__ float stats[3 * 8 + 1];
  const size_t t = blockIdx.x;
  const T *kr = key + t * hc * hidden;
  const T *hr = hidden_states + t * hc * hidden;
  const T *vr = value + t * hidden;
  float vv = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(vr[i]);
    vv = fmaf(v, v, vv);
  }
  vv = q4_block_sum(vv, smem);
  if (threadIdx.x == 0) {
    stats[3 * 8] = vv;
  }
  for (int c = 0; c < hc; ++c) {
    float kk = 0.0f, qq = 0.0f, kq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const size_t o = (size_t)c * hidden + i;
      const float k = q4_f(kr[o]);
      const float q = q4_f(hr[o]);
      kk = fmaf(k, k, kk);
      qq = fmaf(q, q, qq);
      kq = fmaf(k * w_key[o], q * w_query[o], kq);
    }
    kk = q4_block_sum(kk, smem);
    qq = q4_block_sum(qq, smem);
    kq = q4_block_sum(kq, smem);
    if (threadIdx.x == 0) {
      stats[3 * c] = kk;
      stats[3 * c + 1] = qq;
      stats[3 * c + 2] = kq;
    }
  }
  __syncthreads();
  const float mean_vv = stats[3 * 8] / (float)hidden;
  for (int c = 0; c < hc; ++c) {
    const float inv_k = rsqrtf(stats[3 * c] / (float)hidden + eps);
    const float inv_q = rsqrtf(stats[3 * c + 1] / (float)hidden + eps);
    const float g = stats[3 * c + 2] * inv_k * inv_q * rsqrtf((float)hidden);
    const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    const float sig = q4_sigmoid(g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
    const float inv_g = rsqrtf(sig * sig * mean_vv + eps);
    T *gr = gated_out + (t * hc + c) * hidden;
    T *nr = normed_out + (t * hc + c) * hidden;
    const float *wc = w_conv + (size_t)c * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float gv = sig * q4_f(vr[i]);
      gr[i] = q4_t<T>(gv);
      nr[i] = q4_t<T>(gv * inv_g * wc[i]);
    }
  }
}

// out = residual + gated + silu(dilated causal depthwise conv(normed)); history
// before the chunk comes from the pooled conv state `[slot, channels,
// state_len]` (oldest first).
template <typename T>
static __global__ void
q4_ple_conv_kernel(const T *__restrict__ normed, const T *__restrict__ gated,
                   const T *__restrict__ residual, const T *__restrict__ weight,
                   const T *__restrict__ state_pool,
                   const unsigned int *__restrict__ slots, T *__restrict__ out,
                   Q4Tokens l, int channels, int kernel_size, int dilation,
                   int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_tokens * channels) {
    return;
  }
  const int t = idx / channels;
  const int c = idx - (size_t)t * channels;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    out[idx] = residual[idx];
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  float acc = 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    const int j = local - (kernel_size - 1 - k) * dilation;
    float x = 0.0f;
    if (j >= 0) {
      x = q4_f(normed[(size_t)(t - (local - j)) * channels + c]);
    } else if (slot != Q4_PAD_SLOT && state_len + j >= 0) {
      x = q4_f(state_pool[((size_t)slot * channels + c) * state_len +
                          state_len + j]);
    }
    acc = fmaf(q4_f(weight[(size_t)c * kernel_size + k]), x, acc);
  }
  out[idx] = q4_t<T>(q4_f(residual[idx]) + q4_f(gated[idx]) + q4_silu(acc));
}

template <typename T>
static __global__ void
q4_ple_conv_state_update_kernel(const T *__restrict__ normed,
                                T *__restrict__ state_pool,
                                const unsigned int *__restrict__ slots,
                                Q4Tokens l, int channels, int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_seqs * channels) {
    return;
  }
  const int seq = idx / channels;
  const int c = idx - (size_t)seq * channels;
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  T *s = state_pool + ((size_t)slot * channels + c) * state_len;
  float next[Q4_MAX_CONV_STATE];
  for (int i = 0; i < state_len; ++i) {
    const int e = len + i;
    next[i] =
        e < state_len
            ? q4_f(s[e])
            : q4_f(normed[(size_t)(start + e - state_len) * channels + c]);
  }
  for (int i = 0; i < state_len; ++i) {
    s[i] = q4_t<T>(next[i]);
  }
}

// QSA aux cache rows are [raw indexer key | cos | sin] per token slot; once a
// block completes, its first token's key becomes the pooled, rotated block key.

struct Q4Paged {
  const int *block_tables;
  const unsigned int *kv_lens;
  int max_blocks_per_seq;
  int block_size;
};

__device__ __forceinline__ long long q4_slot(const Q4Paged &pg, int seq,
                                             int pos) {
  const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                  pos / pg.block_size];
  return (long long)blk * pg.block_size + pos % pg.block_size;
}

// Absolute kv position of flattened token t, or -1 for padding.
__device__ __forceinline__ int q4_pos_of(const Q4Tokens &l, const Q4Paged &pg,
                                         int t, int &seq) {
  seq = q4_seq_of(l, t);
  if (seq < 0) {
    return -1;
  }
  return (int)pg.kv_lens[seq] - q4_len_of(l, seq) + q4_local_of(l, t);
}

template <typename T>
static __global__ void q4_qsa_aux_write_kernel(
    const T *__restrict__ raw_key, const T *__restrict__ cos,
    const T *__restrict__ sin, const long long *__restrict__ slot_mapping,
    T *__restrict__ aux, int n_tokens, int key_dim, int half_rot, int aux_dim) {
  const int t = blockIdx.x;
  if (t >= n_tokens) {
    return;
  }
  const long long slot = slot_mapping[t];
  if (slot < 0) {
    return;
  }
  T *dst = aux + slot * aux_dim;
  for (int i = threadIdx.x; i < key_dim; i += blockDim.x) {
    dst[i] = raw_key[(size_t)t * key_dim + i];
  }
  for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
    dst[key_dim + i] = cos[(size_t)t * half_rot + i];
    dst[key_dim + half_rot + i] = sin[(size_t)t * half_rot + i];
  }
}

// One warp per token; the token that completes a block writes the block key.
// key_dim must be 128.
template <typename T>
static __global__ void
q4_qsa_finalize_kernel(const T *__restrict__ aux, T *__restrict__ block_keys,
                       Q4Tokens l, Q4Paged pg,
                       const float *__restrict__ norm_weight, int ratio,
                       int half_rot, int aux_dim, float eps) {
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= l.n_tokens) {
    return;
  }
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  if (pos < 0 || (pos + 1) % ratio != 0) {
    return;
  }
  const int start = pos + 1 - ratio;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int m = 0; m < ratio; ++m) {
    const T *src = aux + q4_slot(pg, seq, start + m) * aux_dim;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] += q4_f(src[lane * 4 + i]);
    }
  }
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // HF pools in f32 and rounds to the checkpoint dtype before the norm
    v[i] = q4_f(q4_t<T>(v[i] / (float)ratio));
    ss = fmaf(v[i], v[i], ss);
  }
  ss = q4_warp_sum(ss);
  const float inv = rsqrtf(ss / (float)Q4_QSA_HEAD_DIM + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = q4_f(q4_t<T>(v[i] * inv * norm_weight[lane * 4 + i]));
  }
  const T *cs = aux + q4_slot(pg, seq, start) * aux_dim + Q4_QSA_HEAD_DIM;
  // Keep raw keys intact when speculative rollback reopens a completed block.
  T *dst = block_keys + q4_slot(pg, seq, start) / ratio * Q4_QSA_HEAD_DIM;
  const T *sn = cs + half_rot;
  float out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = v[i];
  }
  // rotate_half partner is half_rot dims away
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const float partner = __shfl_sync(
        0xffffffff, v[i],
        d < half_rot ? lane + half_rot / 4 : (lane - half_rot / 4) & 31);
    if (d < half_rot) {
      out[i] = v[i] * q4_f(cs[d]) - partner * q4_f(sn[d]);
    } else if (d < 2 * half_rot) {
      out[i] = v[i] * q4_f(cs[d - half_rot]) + partner * q4_f(sn[d - half_rot]);
    }
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[lane * 4 + i] = q4_t<T>(out[i]);
  }
}

// scores[t, j] = sum_h relu(q[t, h] . block_key[j]) / sqrt(d) for rows with
// more complete blocks than the budget selects; other rows are skipped.
template <typename T>
static __global__ void
q4_qsa_score_kernel(const T *__restrict__ q, const T *__restrict__ block_keys,
                    Q4Tokens l, Q4Paged pg, float *__restrict__ scores,
                    int score_stride, int n_heads, int ratio, int topk,
                    int key_dim) {
  __shared__ float qs[Q4_QSA_MAX_INDEX_HEADS * Q4_QSA_HEAD_DIM];
  const int t = blockIdx.y;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  if (nb <= topk || (int)(blockIdx.x * blockDim.x) >= nb) {
    return;
  }
  for (int i = threadIdx.x; i < n_heads * Q4_QSA_HEAD_DIM; i += blockDim.x) {
    qs[i] = q4_f(q[(size_t)t * n_heads * Q4_QSA_HEAD_DIM + i]);
  }
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nb) {
    return;
  }
  const T *key = block_keys + q4_slot(pg, seq, j * ratio) / ratio * key_dim;
  float dots[Q4_QSA_MAX_INDEX_HEADS];
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    dots[h] = 0.0f;
  }
  for (int d = 0; d < Q4_QSA_HEAD_DIM; d += 8) {
    float k[8];
    q4_load8(key + d, k);
#pragma unroll
    for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
      if (h < n_heads) {
        const float *qh = qs + h * Q4_QSA_HEAD_DIM + d;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          dots[h] = fmaf(qh[i], k[i], dots[h]);
        }
      }
    }
  }
  float total = 0.0f;
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    if (h < n_heads) {
      total += fmaxf(dots[h], 0.0f);
    }
  }
  scores[(size_t)t * score_stride + j] = total * rsqrtf((float)Q4_QSA_HEAD_DIM);
}

__device__ __forceinline__ unsigned int q4_ordered(float v) {
  if (v != v) {
    return 0u;
  }
  const unsigned int bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Per query row: the indices of the topk largest scores among the first nb
// complete blocks (all of them when nb <= topk). Radix select over the
// order-preserving float bits.
static __global__ void q4_qsa_topk_kernel(const float *__restrict__ scores,
                                   int score_stride, Q4Tokens l, Q4Paged pg,
                                   int ratio, int topk,
                                   int *__restrict__ selected,
                                   int *__restrict__ n_selected) {
  __shared__ unsigned int hist[256];
  __shared__ unsigned int s_prefix, s_mask, s_remaining;
  __shared__ unsigned int s_gt, s_eq;
  const int t = blockIdx.x;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  int *out = selected + (size_t)t * topk;
  if (nb <= topk) {
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
      out[i] = i;
    }
    if (threadIdx.x == 0) {
      n_selected[t] = nb;
    }
    return;
  }
  const float *row = scores + (size_t)t * score_stride;
  if (threadIdx.x == 0) {
    s_prefix = 0u;
    s_mask = 0u;
    s_remaining = topk;
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      hist[i] = 0u;
    }
    __syncthreads();
    const unsigned int prefix = s_prefix;
    const unsigned int mask = s_mask;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) {
      const unsigned int key = q4_ordered(row[j]);
      if ((key & mask) == prefix) {
        atomicAdd(&hist[(key >> shift) & 255u], 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int remaining = s_remaining;
      unsigned int above = 0u;
      int digit = 255;
      for (; digit > 0; --digit) {
        if (above + hist[digit] >= remaining) {
          break;
        }
        above += hist[digit];
      }
      s_remaining = remaining - above;
      s_prefix = prefix | ((unsigned int)digit << shift);
      s_mask = mask | (255u << shift);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_gt = 0u;
    s_eq = 0u;
  }
  __syncthreads();
  const unsigned int threshold = s_prefix;
  const unsigned int want_eq = s_remaining;
  const unsigned int n_gt = topk - want_eq;
  for (int j = threadIdx.x; j < nb; j += blockDim.x) {
    const unsigned int key = q4_ordered(row[j]);
    if (key > threshold) {
      out[atomicAdd(&s_gt, 1u)] = j;
    } else if (key == threshold) {
      const unsigned int e = atomicAdd(&s_eq, 1u);
      if (e < want_eq) {
        out[n_gt + e] = j;
      }
    }
  }
  if (threadIdx.x == 0) {
    n_selected[t] = topk;
  }
}

// Sparse GQA attention of each query over its selected blocks plus the
// incomplete tail. Grid: (tokens, kv_heads, splits); each split covers a
// contiguous chunk of the token's item list and writes partial (max, sum, acc)
// when splits > 1. Head dim is fixed at 256: each lane owns 8 dims.
// flashinfer_layout: K and V are [blocks, kv_heads, block_size, 256]; otherwise
// K is [blocks, kv_heads, 256/8, block_size, 8] and V is [blocks, kv_heads,
// 256, block_size].
template <typename T>
static __global__ void __launch_bounds__(Q4_ATTN_WARPS * 32) q4_qsa_attention_kernel(
    const T *__restrict__ q, const T *__restrict__ k_cache,
    const T *__restrict__ v_cache, Q4Tokens l, Q4Paged pg,
    const int *__restrict__ selected, const int *__restrict__ n_selected,
    int topk, int ratio, int n_q_heads, int n_kv_heads, int items_per_split,
    float scale, int flashinfer_layout, T *__restrict__ out,
    float *__restrict__ part_acc, float *__restrict__ part_ml) {
  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int split = blockIdx.z;
  const int splits = gridDim.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = n_q_heads / n_kv_heads;

  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  int n_block_items = 0;
  int n_items = 0;
  int tail_start = 0;
  if (pos >= 0) {
    const int nb = (pos + 1) / ratio;
    n_block_items = n_selected[t] * ratio;
    tail_start = nb * ratio;
    n_items = n_block_items + (pos + 1 - tail_start);
  }
  const int begin = split * items_per_split;
  const int end = min(n_items, begin + items_per_split);

  float qv[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float acc[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float m[Q4_ATTN_MAX_HEADS_PER_WARP];
  float s[Q4_ATTN_MAX_HEADS_PER_WARP];
  int n_heads_here = 0;
#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    const int h = warp + hh * Q4_ATTN_WARPS;
    m[hh] = -FLT_MAX;
    s[hh] = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[hh][i] = 0.0f;
      qv[hh][i] = 0.0f;
    }
    if (h < group) {
      n_heads_here = hh + 1;
      const int qh = g * group + h;
      q4_load8(q + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8,
               qv[hh]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qv[hh][i] *= scale;
      }
    }
  }

  const size_t head_stride_fi = (size_t)pg.block_size * Q4_ATTN_HEAD_DIM;
  for (int it0 = begin; it0 < end; it0 += Q4_ATTN_ITEMS) {
    // Resolve and load a batch of items up front so the dependent loads overlap
    const T *kp[Q4_ATTN_ITEMS];
    const T *vp[Q4_ATTN_ITEMS];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      const int it = min(it0 + j, end - 1);
      int p;
      if (it < n_block_items) {
        p = selected[(size_t)t * topk + it / ratio] * ratio + it % ratio;
      } else {
        p = tail_start + (it - n_block_items);
      }
      const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                      p / pg.block_size];
      const int off = p % pg.block_size;
      const size_t head_base = (size_t)blk * n_kv_heads + g;
      if (flashinfer_layout) {
        const size_t base = head_base * head_stride_fi +
                            (size_t)off * Q4_ATTN_HEAD_DIM + lane * 8;
        kp[j] = k_cache + base;
        vp[j] = v_cache + base;
      } else {
        kp[j] =
            k_cache +
            (head_base * (Q4_ATTN_HEAD_DIM / 8) + lane) * pg.block_size * 8 +
            (size_t)off * 8;
        vp[j] = v_cache +
                (head_base * Q4_ATTN_HEAD_DIM + lane * 8) * pg.block_size + off;
      }
    }
    float kf[Q4_ATTN_ITEMS][8];
    float vf[Q4_ATTN_ITEMS][8];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      q4_load8(kp[j], kf[j]);
      if (flashinfer_layout) {
        q4_load8(vp[j], vf[j]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          vf[j][i] = q4_f(vp[j][(size_t)i * pg.block_size]);
        }
      }
    }
#pragma unroll
    for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
      if (hh < n_heads_here) {
        float dot[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          dot[j] = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dot[j] = fmaf(qv[hh][i], kf[j][i], dot[j]);
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            dot[j] += __shfl_xor_sync(0xffffffff, dot[j], o);
          }
        }
        float m_new = m[hh];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          if (it0 + j < end) {
            m_new = fmaxf(m_new, dot[j]);
          }
        }
        const float corr = __expf(m[hh] - m_new);
        float psum = 0.0f;
        float pj[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          pj[j] = it0 + j < end ? __expf(dot[j] - m_new) : 0.0f;
          psum += pj[j];
        }
        s[hh] = s[hh] * corr + psum;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float a = acc[hh][i] * corr;
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            a = fmaf(pj[j], vf[j][i], a);
          }
          acc[hh][i] = a;
        }
        m[hh] = m_new;
      }
    }
  }

#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    if (hh >= n_heads_here) {
      continue;
    }
    const int qh = g * (n_q_heads / n_kv_heads) + warp + hh * Q4_ATTN_WARPS;
    if (splits == 1) {
      const float inv = s[hh] > 0.0f ? 1.0f / s[hh] : 0.0f;
      T *o = out + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[i] = q4_t<T>(acc[hh][i] * inv);
      }
    } else {
      const size_t row = ((size_t)t * n_q_heads + qh) * splits + split;
      float *a = part_acc + row * Q4_ATTN_HEAD_DIM + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = acc[hh][i];
      }
      if (lane == 0) {
        part_ml[row * 2] = m[hh];
        part_ml[row * 2 + 1] = s[hh];
      }
    }
  }
}

template <typename T>
static __global__ void
q4_qsa_attention_reduce_kernel(const float *__restrict__ part_acc,
                               const float *__restrict__ part_ml, int splits,
                               T *__restrict__ out) {
  const size_t row = blockIdx.x;
  const int d = threadIdx.x;
  float m = -FLT_MAX;
  for (int sp = 0; sp < splits; ++sp) {
    const float ms = part_ml[(row * splits + sp) * 2];
    if (part_ml[(row * splits + sp) * 2 + 1] > 0.0f) {
      m = fmaxf(m, ms);
    }
  }
  float total = 0.0f;
  float acc = 0.0f;
  for (int sp = 0; sp < splits; ++sp) {
    const float ss = part_ml[(row * splits + sp) * 2 + 1];
    if (ss <= 0.0f) {
      continue;
    }
    const float w = __expf(part_ml[(row * splits + sp) * 2] - m);
    total = fmaf(ss, w, total);
    acc = fmaf(part_acc[(row * splits + sp) * Q4_ATTN_HEAD_DIM + d], w, acc);
  }
  out[row * Q4_ATTN_HEAD_DIM + d] = q4_t<T>(total > 0.0f ? acc / total : 0.0f);
}

// Launcher dtype codes: 0 = f16, 1 = bf16, 2 = f32.

#define Q4_DISPATCH(dtype, ...)                                                \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else if ((dtype) == 1) {                                                 \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = float;                                                  \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

#define Q4_DISPATCH_HALF(dtype, ...)                                           \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

static inline int q4_row_threads(int hidden) {
  int threads = 32;
  while (threads < 512 && threads * 4 < hidden) {
    threads <<= 1;
  }
  return threads;
}

extern "C" void naiwa_qwen4_hc_norm(const void *x, const float *weight, void *out,
                              int rows, int hc, int hidden, float eps,
                              int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_norm_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>((const scalar_t *)x,
                                                       weight, (scalar_t *)out,
                                                       hc, hidden, eps));
}

extern "C" void naiwa_qwen4_hc_mix(const void *xn, const void *gate, void *out,
                             int tokens, int hc, int hidden, int dtype,
                             int64_t stream) {
  const size_t total = (size_t)tokens * hidden;
  if (total == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const unsigned int blocks = (total + threads - 1) / threads;
  Q4_DISPATCH(dtype, q4_hc_mix_kernel<scalar_t><<<blocks, threads, 0, s>>>(
                         (const scalar_t *)xn, (const scalar_t *)gate,
                         (scalar_t *)out, total, hc, hidden));
}

extern "C" void naiwa_qwen4_hc_combine(const void *res, const void *block_out,
                                 const void *inject, void *res_out,
                                 const float *next_weight, void *xn_out,
                                 int rows, int hc, int hidden, float eps,
                                 int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_combine_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>(
                  (const scalar_t *)res, (const scalar_t *)block_out,
                  (const scalar_t *)inject, (scalar_t *)res_out, next_weight,
                  (scalar_t *)xn_out, hc, hidden, eps));
}

extern "C" void naiwa_qwen4_ple_hash(const unsigned int *tokens, Q4Tokens layout,
                               const float *hist_pool,
                               const unsigned int *slots, int hist_len,
                               const Q4PleHashParams *params,
                               long long *rows_out, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 128;
  q4_ple_hash_kernel<<<(layout.n_tokens + threads - 1) / threads, threads, 0,
                       s>>>(tokens, layout, hist_pool, slots, hist_len, *params,
                            rows_out);
}

extern "C" void naiwa_qwen4_ple_hist_update(const unsigned int *tokens,
                                      Q4Tokens layout, float *hist_pool,
                                      const unsigned int *slots, int hist_len,
                                      int64_t stream) {
  if (layout.n_seqs == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  q4_ple_hist_update_kernel<<<(layout.n_seqs + 63) / 64, 64, 0, s>>>(
      tokens, layout, hist_pool, slots, hist_len);
}

extern "C" void naiwa_qwen4_ple_gather(const long long *rows, int n_lookups,
                                 const unsigned long long *shard_ptrs,
                                 const long long *shard_starts, int n_shards,
                                 int head_dim, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0, s>>>(
      rows, n_lookups, shard_ptrs, shard_starts, n_shards, head_dim,
      (unsigned short *)out);
}

extern "C" void naiwa_qwen4_ple_gather_quant(const long long *rows, int n_lookups,
                                       const unsigned char *data,
                                       const void *scales, int head_dim,
                                       int format, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_quant_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0,
                               s>>>(rows, n_lookups, data,
                                    (const __half *)scales, head_dim, format,
                                    (__nv_bfloat16 *)out);
}

extern "C" void naiwa_qwen4_ple_gate(const void *key, const void *hidden_states,
                               const void *value, const float *w_key,
                               const float *w_query, const float *w_conv,
                               void *gated_out, void *normed_out, int tokens,
                               int hc, int hidden, float eps, int dtype,
                               int64_t stream) {
  if (tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_ple_gate_kernel<scalar_t><<<tokens, 512, 0, s>>>(
                         (const scalar_t *)key, (const scalar_t *)hidden_states,
                         (const scalar_t *)value, w_key, w_query, w_conv,
                         (scalar_t *)gated_out, (scalar_t *)normed_out, hc,
                         hidden, eps));
}

extern "C" void naiwa_qwen4_ple_conv(const void *normed, const void *gated,
                               const void *residual, const void *weight,
                               void *state_pool, const unsigned int *slots,
                               void *out, Q4Tokens layout, int channels,
                               int kernel_size, int dilation, int state_len,
                               int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const size_t total = (size_t)layout.n_tokens * channels;
  const unsigned int blocks = (total + threads - 1) / threads;
  const size_t state_total = (size_t)layout.n_seqs * channels;
  const unsigned int state_blocks = (state_total + threads - 1) / threads;
  Q4_DISPATCH(dtype, {
    q4_ple_conv_kernel<scalar_t><<<blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (const scalar_t *)gated,
        (const scalar_t *)residual, (const scalar_t *)weight,
        (const scalar_t *)state_pool, slots, (scalar_t *)out, layout, channels,
        kernel_size, dilation, state_len);
    q4_ple_conv_state_update_kernel<scalar_t><<<state_blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (scalar_t *)state_pool, slots, layout,
        channels, state_len);
  });
}

extern "C" void naiwa_qwen4_qsa_aux_write(const void *raw_key, const void *cos,
                                    const void *sin,
                                    const long long *slot_mapping, void *aux,
                                    int n_tokens, int key_dim, int half_rot,
                                    int aux_dim, int dtype, int64_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH_HALF(dtype,
                   q4_qsa_aux_write_kernel<scalar_t><<<n_tokens, 128, 0, s>>>(
                       (const scalar_t *)raw_key, (const scalar_t *)cos,
                       (const scalar_t *)sin, slot_mapping, (scalar_t *)aux,
                       n_tokens, key_dim, half_rot, aux_dim));
}

extern "C" void naiwa_qwen4_qsa_finalize(const void *aux, void *block_keys,
                                   Q4Tokens layout, Q4Paged paged,
                                   const float *norm_weight, int ratio,
                                   int half_rot, int aux_dim, float eps,
                                   int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  Q4_DISPATCH_HALF(
      dtype, q4_qsa_finalize_kernel<scalar_t>
      <<<(layout.n_tokens + warps - 1) / warps, warps * 32, 0, s>>>(
          (const scalar_t *)aux, (scalar_t *)block_keys, layout, paged,
          norm_weight, ratio, half_rot, aux_dim, eps));
}

extern "C" void naiwa_qwen4_qsa_select(const void *q, const void *block_keys,
                                 Q4Tokens layout, Q4Paged paged, float *scores,
                                 int score_stride, int n_heads, int ratio,
                                 int topk, int key_dim, int *selected,
                                 int *n_selected, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  if (score_stride > topk) {
    const int threads = 128;
    const dim3 grid((score_stride + threads - 1) / threads, layout.n_tokens);
    Q4_DISPATCH_HALF(
        dtype, q4_qsa_score_kernel<scalar_t><<<grid, threads, 0, s>>>(
                   (const scalar_t *)q, (const scalar_t *)block_keys, layout,
                   paged, scores, score_stride, n_heads, ratio, topk, key_dim));
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, s>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

// ---------------------------------------------------------------------------------------------------
// Prompt attention with the whole tile on tensor cores (M33 §20.9).
//
// The naive port in §20.8 was correct and 56% slower, and the accounting of why is the design brief
// here: it filled 4 of the tile's 8 item columns, re-staged K into shared memory every 4 items, and
// built fragments with per-element reads. This kernel does the opposite on all three counts:
//
//   * batches of 16 items - two 8-column n-tiles for the scores, and exactly the k the value phase
//     wants, so nothing is padding;
//   * K and V staged once per batch for the whole block (they are the same for every warp: same kv
//     head, same items), which also removes the 4x redundant KV read the scalar kernel does;
//   * Q staged once per token, and the score fragments handed to the value phase **without** going
//     through shared memory at all: the mma C layout (row g, g+8 x columns 2t, 2t+1) is exactly the
//     A layout the next mma reads (rows g, g+8 x columns 2t, 2t+1 and 2t+8, 2t+9), so p never leaves
//     the registers only the *value* half needs a dimension split - each warp owns HEAD_DIM/4 of the
//     output dims, which is what keeps the accumulator inside the register budget.
//
// That split is also why the softmax needs no cross-warp communication: every warp computes the same
// scores for the same items, so every warp derives the same row max, row sum and rescale factor, and
// each rescales its own quarter of the accumulator. The only block-wide barrier per batch guards the
// staged K/V buffers.
//
// Only used when `splits == 1` (the prompt shape); the split-K path keeps the scalar kernel.
// ---------------------------------------------------------------------------------------------------
#define Q4_ATTN_ITEMS_TILE 16
#define Q4_ATTN_DIMS_PER_WARP (Q4_ATTN_HEAD_DIM / Q4_ATTN_WARPS)
#define Q4_ATTN_MMA_SMEM                                                       \
  ((Q4_ATTN_MMA_ROWS + 2 * Q4_ATTN_ITEMS_TILE) * Q4_ATTN_QSTRIDE * 2)

// Whether to run the tile kernel at all. A one-word read of the environment per launch.
//
// **On by default since M35 §23**, and the reason is a measurement rather than a preference: on a 3 405-token cold
// prompt the tile kernel takes `q4_qsa_attention` from 512.7 ms to 185.5 ms (153 launches either way) and the whole
// kernel timeline from 2.30 s to 2.07 s, against strata's last capture of 1.97 s - so it moves this engine from +17% to
// +5% on the one number that does not depend on the host's memory pressure. Part of that 327 ms comes back as
// `doorbell_wait_copy` (129 -> 236 ms): the GPU stops waiting on attention and starts waiting on the CPU experts, which
// is the next thing to attack and is untouched by this switch.
//
// It does change the arithmetic order (see the header above), so the byte-exact oracle arm no longer matches its
// reference unless it is taken with `NAIWA_QSA_MMA_KERNEL=0` - the rule this repo adopted for exactly this case (worklog
// §20.8: an arithmetic reordering cannot pass a byte-identical gate, so it is gated by an equal-value probe plus the
// text prefix). `NAIWA_QSA_MMA_KERNEL=0` is the scalar arm, bit-for-bit what shipped before.
static bool q4_qsa_tile_enabled() {
  const char *e = getenv("NAIWA_QSA_MMA_KERNEL");
  if (e == nullptr) {
    return true;
  }
  return !(e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F');
}

static bool q4_qsa_tile_supported() {
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
    return false;
  }
  return major >= 8;
}

template <typename T>
static __global__ void __launch_bounds__(Q4_ATTN_WARPS * 32) q4_qsa_attention_tile_kernel(
    const T *__restrict__ q, const T *__restrict__ k_cache,
    const T *__restrict__ v_cache, Q4Tokens l, Q4Paged pg,
    const int *__restrict__ selected, const int *__restrict__ n_selected,
    int topk, int ratio, int n_q_heads, int n_kv_heads, float scale,
    int flashinfer_layout, T *__restrict__ out) {
  constexpr int ST = Q4_ATTN_QSTRIDE;
  constexpr int HW = Q4_ATTN_HEAD_DIM / 8;      // 16-byte vectors per row
  constexpr int NT = Q4_ATTN_DIMS_PER_WARP / 8; // value n-tiles per warp

  extern __shared__ __align__(16) char q4_tile_smem[];
  T *qs = reinterpret_cast<T *>(q4_tile_smem);            // [16][256] unscaled Q
  T *ks = qs + Q4_ATTN_MMA_ROWS * ST;                     // [16][256] K of this batch
  T *vs = ks + Q4_ATTN_ITEMS_TILE * ST;                   // [16][256] V of this batch

  const int t = blockIdx.x, gh = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = n_q_heads / n_kv_heads;
  // Fragment rows gr and gr+8, columns 2tc and 2tc+1 (the C, A and B layouts all agree on this).
  const int gr = lane >> 2, tc = lane & 3;

  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  int n_block_items = 0, n_items = 0, tail_start = 0;
  if (pos >= 0) {
    const int nb = (pos + 1) / ratio;
    n_block_items = n_selected[t] * ratio;
    tail_start = nb * ratio;
    n_items = n_block_items + (pos + 1 - tail_start);
  }

  // Q, once per token. Staged **unscaled** so the f32 scale is applied to the score rather than
  // rounded into a bf16 operand; rows past the group stay zero and are simply never written out.
  for (int v = threadIdx.x; v < Q4_ATTN_MMA_ROWS * HW; v += blockDim.x) {
    const int row = v / HW, off = (v - row * HW) * 8;
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (row < group) {
      const size_t src =
          ((size_t)t * n_q_heads + gh * group + row) * Q4_ATTN_HEAD_DIM + off;
      val = *reinterpret_cast<const uint4 *>(q + src);
    }
    *reinterpret_cast<uint4 *>(qs + row * ST + off) = val;
  }

  // This warp's quarter of the output: 16 rows x Q4_ATTN_DIMS_PER_WARP dims, in NT n-tiles.
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
  }
  float mrow[2] = {-INFINITY, -INFINITY}; // rows gr, gr+8
  float srow[2] = {0.0f, 0.0f};

  const size_t head_stride_fi = (size_t)pg.block_size * Q4_ATTN_HEAD_DIM;
  __syncthreads(); // Q is read by every warp, written by every warp

  for (int it0 = 0; it0 < n_items; it0 += Q4_ATTN_ITEMS_TILE) {
    // ---- stage this batch's K and V, once for the block ----
    for (int v = threadIdx.x; v < Q4_ATTN_ITEMS_TILE * HW; v += blockDim.x) {
      const int j = v / HW, off = (v - j * HW) * 8;
      const int it = min(it0 + j, n_items - 1);
      int p;
      if (it < n_block_items) {
        p = selected[(size_t)t * topk + it / ratio] * ratio + it % ratio;
      } else {
        p = tail_start + (it - n_block_items);
      }
      const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                      p / pg.block_size];
      const int off_item = p % pg.block_size;
      const size_t head_base = (size_t)blk * n_kv_heads + gh;
      T *kd = ks + j * ST + off;
      T *vd = vs + j * ST + off;
      if (flashinfer_layout) {
        const size_t base =
            head_base * head_stride_fi + (size_t)off_item * Q4_ATTN_HEAD_DIM + off;
        *reinterpret_cast<uint4 *>(kd) =
            *reinterpret_cast<const uint4 *>(k_cache + base);
        *reinterpret_cast<uint4 *>(vd) =
            *reinterpret_cast<const uint4 *>(v_cache + base);
      } else {
        const T *kp = k_cache +
                      (head_base * (Q4_ATTN_HEAD_DIM / 8) + (off / 8)) *
                          pg.block_size * 8 +
                      (size_t)off_item * 8;
        const T *vp = v_cache +
                      (head_base * Q4_ATTN_HEAD_DIM + off) * pg.block_size +
                      off_item;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          kd[i] = kp[i];
          vd[i] = vp[(size_t)i * pg.block_size];
        }
      }
    }
    __syncthreads();

    // ---- scores: A = Q (16 heads x 256 dims), B = K (256 dims x 16 items) ----
    float c[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) c[nt][i] = 0.0f;
    }
    for (int base = 0; base < Q4_ATTN_HEAD_DIM; base += 16) {
      const int col = base + tc * 2;
      unsigned int a[4];
      a[0] = q4_pack2t<T>(qs[gr * ST + col], qs[gr * ST + col + 1]);
      a[1] = q4_pack2t<T>(qs[(gr + 8) * ST + col], qs[(gr + 8) * ST + col + 1]);
      a[2] = q4_pack2t<T>(qs[gr * ST + col + 8], qs[gr * ST + col + 9]);
      a[3] =
          q4_pack2t<T>(qs[(gr + 8) * ST + col + 8], qs[(gr + 8) * ST + col + 9]);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const T *kb = ks + (nt * 8 + gr) * ST + col;
        unsigned int b[2];
        b[0] = q4_pack2t<T>(kb[0], kb[1]);
        b[1] = q4_pack2t<T>(kb[8], kb[9]);
        q4_mma16x8x16<T>(c[nt], a, b);
      }
    }

    // ---- online softmax, per row, inside the four lanes that share it ----
    // Scale and mask first: items past the end keep -inf, so their p is exactly zero and they
    // contribute nothing to the value phase.
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int j = nt * 8 + tc * 2 + (i & 1);
        c[nt][i] = (it0 + j < n_items) ? c[nt][i] * scale : -INFINITY;
        mx[i >> 1] = fmaxf(mx[i >> 1], c[nt][i]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], o));
      mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], o));
    }
    float corr[2], psum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      // -inf stays -inf when a batch is all masked; exp(-inf - -inf) is NaN, so guard it.
      const float mn = fmaxf(mrow[r], mx[r]);
      corr[r] = (mrow[r] == -INFINITY) ? 0.0f : __expf(mrow[r] - mn);
      mrow[r] = mn;
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = i >> 1;
        const float p = (c[nt][i] == -INFINITY) ? 0.0f : __expf(c[nt][i] - mrow[r]);
        c[nt][i] = p;
        psum[r] += p;
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      psum[0] += __shfl_xor_sync(0xffffffffu, psum[0], o);
      psum[1] += __shfl_xor_sync(0xffffffffu, psum[1], o);
    }
    srow[0] = srow[0] * corr[0] + psum[0];
    srow[1] = srow[1] * corr[1] + psum[1];

    // ---- values: A = p, straight out of the score fragments (same layout), B = V ----
    // Clearing the accumulator's own rescale first, then accumulating, is the online-softmax
    // update: acc = acc * corr + p * V.
    unsigned int pa[4];
    pa[0] = q4_pack2t<T>(q4_t<T>(c[0][0]), q4_t<T>(c[0][1]));
    pa[1] = q4_pack2t<T>(q4_t<T>(c[0][2]), q4_t<T>(c[0][3]));
    pa[2] = q4_pack2t<T>(q4_t<T>(c[1][0]), q4_t<T>(c[1][1]));
    pa[3] = q4_pack2t<T>(q4_t<T>(c[1][2]), q4_t<T>(c[1][3]));
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int dim = warp * Q4_ATTN_DIMS_PER_WARP + nt * 8 + gr;
      unsigned int b[2];
      b[0] = q4_pack2t<T>(vs[(tc * 2) * ST + dim], vs[(tc * 2 + 1) * ST + dim]);
      b[1] = q4_pack2t<T>(vs[(tc * 2 + 8) * ST + dim],
                          vs[(tc * 2 + 9) * ST + dim]);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[nt][i] *= corr[i >> 1];
      q4_mma16x8x16<T>(acc[nt], pa, b);
    }
    __syncthreads(); // the staged K/V are about to be overwritten
  }

  // ---- normalise and write out ----
  const int col0 = warp * Q4_ATTN_DIMS_PER_WARP + tc * 2;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = gr + r * 8;
    if (row < group) {
      const float inv = srow[r] > 0.0f ? 1.0f / srow[r] : 0.0f;
      T *dst = out + ((size_t)t * n_q_heads + gh * group + row) * Q4_ATTN_HEAD_DIM +
               col0;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        // Two adjacent dims are four bytes: one store, and a plain conversion at the end.
        const T pair[2] = {q4_t<T>(acc[nt][r * 2 + 0] * inv),
                           q4_t<T>(acc[nt][r * 2 + 1] * inv)};
        memcpy(dst + nt * 8, pair, 4);
      }
    }
  }
}

// Top-k over scores produced elsewhere (the cuTile scoring kernel).
extern "C" void naiwa_qwen4_qsa_topk(const float *scores, int score_stride,
                               Q4Tokens layout, Q4Paged paged, int ratio,
                               int topk, int *selected, int *n_selected,
                               int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, (cudaStream_t)stream>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

extern "C" void naiwa_qwen4_qsa_attention(
    const void *q, const void *k_cache, const void *v_cache, Q4Tokens layout,
    Q4Paged paged, const int *selected, const int *n_selected, int topk,
    int ratio, int n_q_heads, int n_kv_heads, int splits, int items_per_split,
    float scale, int flashinfer_layout, void *out, float *part_acc,
    float *part_ml, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  // The tile kernel covers the prompt shape only: one split, one block per (token, kv head), the
  // group fitting the 16-row tile, and tensor cores to run it on.
  if (splits == 1 && n_q_heads / n_kv_heads <= Q4_ATTN_MMA_ROWS &&
      q4_qsa_tile_enabled() && q4_qsa_tile_supported()) {
    const dim3 tgrid(layout.n_tokens, n_kv_heads);
    Q4_DISPATCH_HALF(dtype, {
      q4_qsa_attention_tile_kernel<scalar_t>
          <<<tgrid, Q4_ATTN_WARPS * 32, Q4_ATTN_MMA_SMEM, s>>>(
              (const scalar_t *)q, (const scalar_t *)k_cache,
              (const scalar_t *)v_cache, layout, paged, selected, n_selected,
              topk, ratio, n_q_heads, n_kv_heads, scale, flashinfer_layout,
              (scalar_t *)out);
    });
    return;
  }
  const dim3 grid(layout.n_tokens, n_kv_heads, splits);
  Q4_DISPATCH_HALF(dtype, {
    q4_qsa_attention_kernel<scalar_t><<<grid, Q4_ATTN_WARPS * 32, 0, s>>>(
        (const scalar_t *)q, (const scalar_t *)k_cache,
        (const scalar_t *)v_cache, layout, paged, selected, n_selected, topk,
        ratio, n_q_heads, n_kv_heads, items_per_split, scale, flashinfer_layout,
        (scalar_t *)out, part_acc, part_ml);
    if (splits > 1) {
      q4_qsa_attention_reduce_kernel<scalar_t>
          <<<layout.n_tokens * n_q_heads, Q4_ATTN_HEAD_DIM, 0, s>>>(
              part_acc, part_ml, splits, (scalar_t *)out);
    }
  });
}

// ---------------------------------------------------------------------------------------------------
// Tensor-core primitives for the prompt path's attention (M33 §20.7).
//
// The scalar kernel spends the prompt's attention time on instruction issue, not on any one of its
// parts: five controlled experiments (worklog §20.6) could not move it by more than 13.6%, because
// one item costs 16 FMA a lane plus a five-step reduction, a max, an exp and a rescale. One
// `mma.sync.m16n8k16` is 2048 FMA in a single instruction - more than a whole item costs today.
//
// The risky part of that rewrite is not the loop structure, it is the fragment layout: get it wrong
// and the kernel still runs, just with quietly wrong numbers. So the primitive lives here with its
// own check (`qwen4_qsa_mma_check`), which computes the same 16x8 score tile two ways - this MMA and
// a plain per-element dot product - and prints the difference. `NAIWA_QSA_MMA_CHECK=1` runs it at
// startup.
// ---------------------------------------------------------------------------------------------------

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define Q4_HAS_MMA 1
#else
#define Q4_HAS_MMA 0
#endif

// D = A[16x16] * B[16x8] + C, bf16 inputs, f32 accumulate. A is row-major, B column-major (i.e. B is
// given k-major, which is what a [n][k] K tile is), C/D are f32.
__device__ __forceinline__ void q4_mma_m16n8k16_bf16(float c[4], const unsigned int a[4],
                                                     const unsigned int b[2]) {
#if Q4_HAS_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c; (void)a; (void)b;
#endif
}

// Two bf16 as one 32-bit register, low half first (what the fragment expects).
__device__ __forceinline__ unsigned int q4_pack2(__nv_bfloat16 x, __nv_bfloat16 y) {
  __nv_bfloat162 h = __halves2bfloat162(x, y);
  return *reinterpret_cast<unsigned int *>(&h);
}

// One warp: scores[16][8] = Q[16][head_dim] * K[8][head_dim]^T, all head-dim steps accumulated.
//
// Fragment layouts, PTX ISA "mma.m16n8k16 with .f16/.bf16":
//   A: a0,a1 = (row g,   col 2t, 2t+1)     a2,a3 = (row g+8, col 2t, 2t+1)
//      a4,a5 = (row g,   col 2t+8, 2t+9)   a6,a7 = (row g+8, col 2t+8, 2t+9)
//   B: b0,b1 = (k 2t, 2t+1; col n=g)       b2,b3 = (k 2t+8, 2t+9; col n=g)
//   C: c0,c1 = (row g,   col 2t, 2t+1)     c2,c3 = (row g+8, col 2t, 2t+1)
// with g = lane >> 2 and t = lane & 3.
__device__ __forceinline__ void q4_qk_mma_warp(const __nv_bfloat16 *__restrict__ q,
                                               const __nv_bfloat16 *__restrict__ k, int head_dim,
                                               float *__restrict__ out /* [16][8] */) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int base = 0; base < head_dim; base += 16) {
    const int cA = base + t * 2;
    unsigned int a[4], b[2];
    a[0] = q4_pack2(q[(size_t)g * head_dim + cA], q[(size_t)g * head_dim + cA + 1]);
    a[1] = q4_pack2(q[(size_t)(g + 8) * head_dim + cA], q[(size_t)(g + 8) * head_dim + cA + 1]);
    a[2] = q4_pack2(q[(size_t)g * head_dim + cA + 8], q[(size_t)g * head_dim + cA + 9]);
    a[3] = q4_pack2(q[(size_t)(g + 8) * head_dim + cA + 8],
                    q[(size_t)(g + 8) * head_dim + cA + 9]);
    // The K tile is given as [n][k], i.e. row n = item, which is exactly the column-major B.
    b[0] = q4_pack2(k[(size_t)g * head_dim + cA], k[(size_t)g * head_dim + cA + 1]);
    b[1] = q4_pack2(k[(size_t)g * head_dim + cA + 8], k[(size_t)g * head_dim + cA + 9]);
    q4_mma_m16n8k16_bf16(c, a, b);
  }
  out[(size_t)g * 8 + t * 2 + 0] = c[0];
  out[(size_t)g * 8 + t * 2 + 1] = c[1];
  out[(size_t)(g + 8) * 8 + t * 2 + 0] = c[2];
  out[(size_t)(g + 8) * 8 + t * 2 + 1] = c[3];
}

// The reference: the same tile by plain per-element f32 dot products, one thread each.
static __global__ void q4_qk_ref_kernel(const __nv_bfloat16 *__restrict__ q,
                                 const __nv_bfloat16 *__restrict__ k, int head_dim,
                                 float *__restrict__ out) {
  const int r = threadIdx.x / 8, n = threadIdx.x % 8;
  float acc = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    acc = fmaf((float)q[(size_t)r * head_dim + i], (float)k[(size_t)n * head_dim + i], acc);
  }
  out[(size_t)r * 8 + n] = acc;
}

static __global__ void q4_qk_mma_kernel(const __nv_bfloat16 *__restrict__ q,
                                 const __nv_bfloat16 *__restrict__ k, int head_dim,
                                 float *__restrict__ out) {
  q4_qk_mma_warp(q, k, head_dim, out);
}

}  // namespace

// Returns 0 when the MMA and the reference agree, 1 otherwise. Prints the numbers either way.
extern "C" int naiwa_qwen4_qsa_mma_check(int head_dim) {
  const int rows = 16, items = 8;
  const size_t n_q = (size_t)rows * head_dim, n_k = (size_t)items * head_dim;
  __nv_bfloat16 *hq = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
  __nv_bfloat16 *hk = (__nv_bfloat16 *)malloc(n_k * sizeof(__nv_bfloat16));
  float *href = (float *)malloc((size_t)rows * items * sizeof(float));
  float *hmma = (float *)malloc((size_t)rows * items * sizeof(float));
  if (!hq || !hk || !href || !hmma) {
    free(hq); free(hk); free(href); free(hmma);
    return 2;
  }
  // Exactly representable in bf16, so both paths see identical inputs and the only difference left
  // is the order the 256 products are summed in.
  for (int i = 0; i < rows; ++i)
    for (int d = 0; d < head_dim; ++d)
      hq[(size_t)i * head_dim + d] = __float2bfloat16((float)((i * 7 + d * 3) % 13 - 6) * 0.25f);
  for (int i = 0; i < items; ++i)
    for (int d = 0; d < head_dim; ++d)
      hk[(size_t)i * head_dim + d] = __float2bfloat16((float)((i * 5 + d * 11) % 17 - 8) * 0.125f);

  void *dq = nullptr, *dk = nullptr, *dref = nullptr, *dmma = nullptr;
  const int rc = (int)(cudaMalloc(&dq, n_q * sizeof(__nv_bfloat16)) ||
                       cudaMalloc(&dk, n_k * sizeof(__nv_bfloat16)) ||
                       cudaMalloc(&dref, (size_t)rows * items * sizeof(float)) ||
                       cudaMalloc(&dmma, (size_t)rows * items * sizeof(float)));
  if (rc) {
    fprintf(stderr, "QSA_MMA_CHECK: cudaMalloc failed\n");
    free(hq); free(hk); free(href); free(hmma);
    return 2;
  }
  cudaMemcpy(dq, hq, n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk, n_k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  q4_qk_ref_kernel<<<1, rows * items>>>((const __nv_bfloat16 *)dq, (const __nv_bfloat16 *)dk,
                                        head_dim, (float *)dref);
  q4_qk_mma_kernel<<<1, 32>>>((const __nv_bfloat16 *)dq, (const __nv_bfloat16 *)dk, head_dim,
                              (float *)dmma);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "QSA_MMA_CHECK: kernel failed: %s\n", cudaGetErrorString(err));
    cudaFree(dq); cudaFree(dk); cudaFree(dref); cudaFree(dmma);
    free(hq); free(hk); free(href); free(hmma);
    return 2;
  }
  cudaMemcpy(href, dref, (size_t)rows * items * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hmma, dmma, (size_t)rows * items * sizeof(float), cudaMemcpyDeviceToHost);
  double worst = 0.0, worst_rel = 0.0;
  int worst_r = 0, worst_n = 0;
  for (int r = 0; r < rows; ++r) {
    for (int n = 0; n < items; ++n) {
      const size_t i = (size_t)r * items + n;
      const double diff = fabs((double)hmma[i] - (double)href[i]);
      const double rel = diff / (fabs((double)href[i]) + 1e-6);
      if (diff > worst) { worst = diff; worst_rel = rel; worst_r = r; worst_n = n; }
    }
  }
  printf("QSA_MMA_CHECK head_dim=%d: worst |mma - ref| = %.6g (relative %.3g) at row %d item %d; "
         "ref[0][0] = %.6g, mma[0][0] = %.6g\n",
         head_dim, worst, worst_rel, worst_r, worst_n, (double)href[0], (double)hmma[0]);
  const int bad = worst > 1e-2 * (1.0 + fabs((double)href[(size_t)worst_r * items + worst_n]));
  // What the primitive is worth: the same kernel over many blocks, timed with events. Each block does
  // one 16x8 score tile of `head_dim` dims = 16 * 8 * head_dim * 2 flops.
  {
    const int blocks = 16384, iters = 8;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int w = 0; w < 2; ++w) {  // one warm pass, then the timed one
      for (int i = 0; i < iters; ++i)
        q4_qk_mma_kernel<<<blocks, 32>>>((const __nv_bfloat16 *)dq, (const __nv_bfloat16 *)dk,
                                         head_dim, (float *)dmma);
    }
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
      q4_qk_mma_kernel<<<blocks, 32>>>((const __nv_bfloat16 *)dq, (const __nv_bfloat16 *)dk,
                                       head_dim, (float *)dmma);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double flops = (double)blocks * iters * 16.0 * 8.0 * (double)head_dim * 2.0;
    printf("QSA_MMA_CHECK throughput: %.2f TFLOP/s over %d tiles x %d iters (%.1f ms)\n",
           flops / (ms * 1e-3) / 1e12, blocks, iters, ms);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }
  cudaFree(dq); cudaFree(dk); cudaFree(dref); cudaFree(dmma);
  free(hq); free(hk); free(href); free(hmma);
  return bad;
}
