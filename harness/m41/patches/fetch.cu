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
