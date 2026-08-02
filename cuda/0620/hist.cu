#include <algorithm>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

__global__ void hist_kernal(const int* input, const int N, const int num_bins, int* histogram) {
    extern __shared__ int sm[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        sm[i] = 0;
    }
    __syncthreads();
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        atomicAdd(&sm[input[i]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(histogram + i, sm[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int blocks = std::min(1024, (N + 255) / 256);
    hist_kernal<<<blocks, 256, num_bins * sizeof(int)>>>(input, N, num_bins, histogram);
}