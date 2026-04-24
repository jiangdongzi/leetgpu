#include <cuda_runtime.h>
#include <stdio.h>

#define cdiv(a, b) (a+b+1)/b

#define BLOCK 256
#define SIZE 10000
// #define PRINT

__global__ void scan_block(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    if (n >= N) return;
    __shared__ float sh[BLOCK];
    sh[t] = input[n];
    __syncthreads();
    int step = 1;
    while (step < BLOCK) {
        float tmp = sh[t];
        if (t >= step)
            tmp += sh[t-step];
        step *= 2;
        __syncthreads();
        sh[t] = tmp;
        __syncthreads();
    }
    output[n] = sh[t];
    if (t == BLOCK-1) block_sums[blockIdx.x] = sh[t];

}

__global__ void merge_with_block_sums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];

}

void scan(const float* input, float* output, int N) {
    float* blocksums;
    int blocksums_len = cdiv(N, BLOCK);
    cudaMalloc((void**)&blocksums, blocksums_len * sizeof(float));
    scan_block<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, blocksums);
    if (N <= BLOCK) return;
    scan(blocksums, blocksums, blocksums_len);
    merge_with_block_sums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, blocksums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
