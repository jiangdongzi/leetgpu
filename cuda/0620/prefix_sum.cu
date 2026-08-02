#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

__global__ void scan_block(const float* input, const int N, float* output, float* block_sum) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockDim.x * blockIdx.x;
    __shared__ float sm[1024];
    if (gtid < N) {
        sm[tid] = input[gtid];
    } else {
        sm[tid] = 0.f;
    }
    __syncthreads();
    for (int step = 1; step < 1024; step *= 2) {
        float tmp = 0.f;
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
        block_sum[blockIdx.x] = sm[1023];
    }
}

__global__ void merge(float* output, const int N, const float* block_sum) {
    const int b = blockIdx.x;
    const int gtid = (b + 1) * blockDim.x + threadIdx.x;
    if (gtid < N) {
        output[gtid] += block_sum[b];
    }
}

void scan(const float* input, const int N, float* output) {
    const int blocks = (N + 1023) / 1024;
    float* block_sum;
    cudaMalloc(&block_sum, sizeof(float) * blocks);
    scan_block<<<blocks, 1024>>>(input, N, output, block_sum);
    if (blocks == 1) {
        cudaFree(block_sum);
        return;
    }
    scan(block_sum, blocks, block_sum);
    merge<<<blocks - 1, 1024>>>(output, N, block_sum);
    cudaFree(block_sum);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, N, output);
}