#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <functional>


__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int shared_his[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_his[i] = 0;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        const int bin = input[i];
        atomicAdd(&shared_his[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (shared_his[i] > 0) {
            atomicAdd(&histogram[i], shared_his[i]);
        }
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    const int threads_per_block = 256;
    const int blocks = std::min(4096, (N + threads_per_block - 1) / threads_per_block);
    size_t shared_mem_bytes = num_bins * sizeof(int);
    histogram_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(input, histogram, N, num_bins);
}