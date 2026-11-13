#include <cuda_runtime.h>
#include <stdint.h>
__global__ void naiwa_fetch_projection_kernel(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    if ((((uintptr_t)dst | (uintptr_t)src) & 15) == 0) {
        for (size_t j=i; j<n/16; j+=stride) ((uint4*)dst)[j] = ((const uint4*)src)[j];
        for (size_t j=n/16*16+i; j<n; j+=stride) dst[j]=src[j];
    } else {
        for (;i<n;i+=stride) dst[i]=src[i];
    }
}
extern "C" int naiwa_fetch_projection(uint64_t dst, uint64_t src, size_t n, void* stream) {
    naiwa_fetch_projection_kernel<<<384,256,0,(cudaStream_t)stream>>>((uint8_t*)dst,(const uint8_t*)src,n);
    return (int)cudaGetLastError();
}

// One launch for a whole layer's staged experts: descriptors are [dst, src, len] u64 triples in
// device memory. 384 blocks over n entries keeps enough loads in flight to reach link bandwidth
// even when a layer stages only one or two experts.
__global__ void naiwa_fetch_batch_kernel(const unsigned long long* __restrict__ desc, int n) {
    const int per = (384 + n - 1) / n;
    const int e = blockIdx.x % n, b = blockIdx.x / n;
    const unsigned long long* d = desc + (size_t) e * 3;
    unsigned char* dst = (unsigned char*) (uintptr_t) d[0];
    const unsigned char* src = (const unsigned char*) (uintptr_t) d[1];
    const size_t len = (size_t) d[2];
    const size_t stride = (size_t) per * blockDim.x;
    const size_t i = (size_t) b * blockDim.x + threadIdx.x;
    const size_t words = len / 16;
    if ((((uintptr_t) dst | (uintptr_t) src) & 15) == 0) {
        // Index by 16-byte units so both sides stay aligned whatever the thread offset is.
        for (size_t j = i; j < words; j += stride) ((uint4*) dst)[j] = ((const uint4*) src)[j];
        for (size_t j = words * 16 + i; j < len; j += stride) dst[j] = src[j];
    } else {
        for (size_t j = i; j < len; j += stride) dst[j] = src[j];
    }
}

extern "C" int naiwa_fetch_batch(unsigned long long desc_dev, int n, void* stream) {
    if (n <= 0) return 0;
    const int blocks = 384;
    naiwa_fetch_batch_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
        (const unsigned long long*) desc_dev, n);
    return (int) cudaGetLastError();
}
