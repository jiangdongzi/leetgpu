#include <algorithm>
#include <cuda_runtime.h>

__global__ void scan_block(const int* input, const int N, int* output, int* block_sums) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
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

__global__ void merge(int* output, const int N, int* block_sums) {
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

__global__ void max_kernal(const int* prefix_sum, const int N, const int window_size, int* output) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    int local_max = 0x80000000;
    for (int i = gtid; i < N; i += stride) {
        const int tmp = prefix_sum[i] - prefix_sum[i - window_size];
        local_max = max(tmp, local_max);
    }
    __shared__ int s_max[256];
    s_max[tid] = local_max;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) {
            s_max[tid] = max(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(output, s_max[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int* prefix_sum;
    cudaMalloc(&prefix_sum, sizeof(int) * (N + 1));
    scan(input, N, prefix_sum + 1);
    cudaMemset(prefix_sum, 0, sizeof(int));
    const int eles = N - window_size + 1;
    const int blocks = std::min((eles + 255) / 256, 1024);
    int init = INT_MIN;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);
    max_kernal<<<blocks, 256>>>(prefix_sum + window_size, eles, window_size, output);
    cudaFree(prefix_sum);
}