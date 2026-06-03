#include <cuda_runtime.h>

__global__ void hist_kernal(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int sm[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) sm[i] = 0;
    __syncthreads();
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gtid; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&sm[input[i]], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_bins; i++) {
            atomicAdd(&histogram[i], 1);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int block_size = 1024;
    size_t sm_size = num_bins * sizeof(float) * num_bins;
    hist_kernal<<<(N + 1024) / 1024, 1024>>>(input, histogram, N, num_bins);
}