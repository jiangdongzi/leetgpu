#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void scan_block(const int* input, const int N, int* output, int* block_sums) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int gtid = tid + b * blockDim.x;
    __shared__ int sm[256];
    if (gtid < N) {
        sm[tid] = input[gtid];
    } else {
        sm[tid] = 0;
    }
    __syncthreads();
    for (int step = 1; step < 256; step *= 2) {
        int tmp = 0;
        if (step <= tid) {
            tmp = sm[tid - step];
        }
        __syncthreads();
        sm[tid] += tmp;
        __syncthreads();
    }
    if (gtid < N) {
        output[gtid] = sm[tid];
    }
    if (tid == 0) {
        block_sums[b] = sm[255];
    }
}

__global__ void merge(int* output, const int N, const int* block_sums) {
    const int b = blockIdx.x;
    const int gtid = threadIdx.x + (b + 1) * blockDim.x;
    if (gtid < N) {
        output[gtid] += block_sums[b];
    }
}

void scan(const int* input, const int N, int* output) {
    const int blocks = (N + 255) / 256;
    int* block_sums;
    cudaMalloc(&block_sums, sizeof(int) * blocks);
    scan_block<<<blocks, 256>>>(input, N, output, block_sums);
    if (blocks == 1) {
        cudaFree(block_sums);
        return;
    }
    scan(block_sums, blocks, block_sums);
    merge<<<blocks - 1, 256>>>(output, N, block_sums);
    cudaFree(block_sums);
}

__global__ void window_max_kernal(const int* prefix_sum, const int N, const int window_size, int* output) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int maxVal = INT32_MIN;
    for (int i = gtid; i < N; i += stride) {
        const int cur_window_sum = prefix_sum[i] - prefix_sum[i - window_size];
        maxVal = max(maxVal, cur_window_sum);
    }
    __shared__ int sm[256];
    sm[tid] = maxVal;
    __syncthreads();
    for (int offset = 256 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] = max(sm[tid], sm[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(output, sm[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int* prefix_sum;
    cudaMalloc(&prefix_sum, sizeof(int) * (N + 1));
    cudaMemset(prefix_sum, 0, 4);
    scan(input, N, prefix_sum + 1);
    int int32min = INT32_MIN;
    cudaMemcpy(output, &int32min, 4, cudaMemcpyHostToDevice);
    const int blocks = std::min(1024, (N + 255) / 256);
    window_max_kernal<<<blocks, 256>>>(prefix_sum + window_size, N - window_size + 1, window_size, output);
    cudaFree(prefix_sum);
}