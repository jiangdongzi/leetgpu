#include <cuda_runtime.h>

__global__ void scan_block(const float* intput, float* output, const int N, float* block_sum) {
    const int g_idx = blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;
    const int gtid = g_idx + tid;
    __shared__ float sm[256];
    sm[tid] = gtid < N ? intput[gtid] : 0.f;
    __syncthreads();
    int step = 1;
    while (step < 256) {
        float tmp = 0.f;
        if (tid >= step) {
            tmp = sm[tid - step];
        }
        __syncthreads();
        sm[tid] += tmp;
        __syncthreads();
        step *= 2;
    }
    if (gtid < N) {
        output[gtid] = sm[tid];
    }
    if (tid == 0) {
        block_sum[blockIdx.x] = sm[255];
    }
}

__global__ void merge_sum(float* output, const int N, const float* block_sum) {
    const int b = blockIdx.x;
    const int g_idx = (b + 1) * blockDim.x;
    const int my_idx = g_idx + threadIdx.x;
    if (my_idx < N) {
        output[my_idx] += block_sum[b];
    }
}

void scan(const float* input, float* output, int N) {
    const int blocks = (N + 255) / 256;
    float* block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(float));
    scan_block<<<blocks, 256>>>(input, output, N, block_sums);
    if (blocks == 1) return;
    scan(block_sums, block_sums, blocks);
    merge_sum<<<blocks - 1, 256>>>(output, N, block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
}