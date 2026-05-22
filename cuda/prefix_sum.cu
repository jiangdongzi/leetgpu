#include <cuda_runtime.h>
#include <stdio.h>

#define cdiv(a, b) (((a) + (b) - 1) / (b))

#define BLOCK 256
#define SIZE 10000
// #define PRINT

__global__ void scan_block(const float* input, float* output, int N, float* block_sums) {
    int block_start = blockDim.x * blockIdx.x;
    int n = threadIdx.x + block_start;
    int t = threadIdx.x;

    // 这里不能让越界线程提前 return，因为同一个 block 内所有线程都必须执行到
    // 每一个 __syncthreads()。越界线程只贡献 0，并跳过最后的全局内存写回。
    bool valid = n < N;

    // 当前 block 实际负责的输入元素数量。最后一个 block 可能不足 BLOCK 个元素。
    int block_count = N - block_start;
    if (block_count > blockDim.x) block_count = blockDim.x;

    __shared__ float sh[BLOCK];
    sh[t] = valid ? input[n] : 0.0f;
    __syncthreads();

    // 在一个 CUDA block 内做 Hillis-Steele inclusive scan。
    // step = 1 后，每个 lane 最多包含 2 个元素的和。
    // step = 2 后，每个 lane 最多包含 4 个元素的和。
    // step = 4 后，每个 lane 最多包含 8 个元素的和，依此类推。
    int step = 1;
    while (step < BLOCK) {
        // 先基于旧的 shared memory 值计算到 tmp。等所有线程都读完旧值后再写回，
        // 否则某个 lane 可能读到另一个 lane 在本轮刚更新过的值。
        float tmp = sh[t];
        if (t >= step)
            tmp += sh[t-step];
        step *= 2;
        __syncthreads();
        sh[t] = tmp;
        __syncthreads();
    }

    if (valid) output[n] = sh[t];

    // 保存当前 block 的总和。满 block 时是 lane BLOCK - 1；
    // 最后一个 partial block 时是最后一个有效 lane。
    if (t == block_count - 1) block_sums[blockIdx.x] = sh[t];

}

__global__ void merge_with_block_sums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;

    // block_sums 已经被 scan 过。对于逻辑 block (b + 1)，需要加上它之前
    // 所有 block 的总和，也就是 block_sums[b]。
    output[n] += block_sums[b];

}

void scan(const float* input, float* output, int N) {
    if (N <= 0) return;

    float* blocksums;
    int blocksums_len = cdiv(N, BLOCK);
    cudaMalloc((void**)&blocksums, blocksums_len * sizeof(float));

    // 第一遍：每个 block 内部独立做 scan，同时把每个 block 的总和收集到 blocksums。
    scan_block<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, blocksums);

    // 如果只有一个 block，block 内 scan 的结果就是完整结果，不需要递归合并。
    if (N <= BLOCK) {
        cudaFree(blocksums);
        return;
    }

    // 递归 scan 每个 block 的总和，使 blocksums[i] 变成 block 0..i 的累计总和。
    scan(blocksums, blocksums, blocksums_len);

    // 除了第 0 个 block，其余 block 都加上自己之前所有 block 的累计总和。
    merge_with_block_sums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, blocksums);

    cudaFree(blocksums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
