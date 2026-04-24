#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// =====================================================================
// 核心数学与线程通信模块 (面试官最看重的地方)
// =====================================================================

// Warp 级别的 Online Softmax 归约
__device__ __forceinline__ void warpReduceOnlineSoftmax(float& max_val, float& sum_val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);

        float new_max = fmaxf(max_val, other_max);
        // Online Softmax 的核心状态转移公式
        sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// Block 级别的 Online Softmax 归约
__device__ __forceinline__ void blockReduceOnlineSoftmax(float& val_max, float& val_sum) {
    // 1. 先在各自的 Warp 内部规约
    warpReduceOnlineSoftmax(val_max, val_sum);

    // 2. 将每个 Warp 的结果写入 Shared Memory
    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        smem_max[warp_id] = val_max;
        smem_sum[warp_id] = val_sum;
    }
    __syncthreads();

    // 3. 由第 0 个 Warp 汇总所有 Warp 的结果
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val_max = (lane_id < num_warps) ? smem_max[lane_id] : -FLT_MAX;
        val_sum = (lane_id < num_warps) ? smem_sum[lane_id] : 0.0f;
        
        warpReduceOnlineSoftmax(val_max, val_sum);
    }
}

// =====================================================================
// Kernel 逻辑 (高度精简，一目了然)
// =====================================================================

// Kernel 1: 第一次遍历显存，计算每个 Block 的局部 max 和 sum
__global__ void pass1_block_reduce_kernel(const float* x, float* block_max, float* block_sum, int N) {
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // Grid-Stride Loop: 处理任意大小的 N
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float val = x[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // 调用高度封装的 Block 归约
    blockReduceOnlineSoftmax(local_max, local_sum);

    // Block 内的 0 号线程负责写回 Global Memory
    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = local_max;
        block_sum[blockIdx.x] = local_sum;
    }
}

// Kernel 2: 汇总所有 Block 的结果，得到全局唯一的 max 和 sum (单 Block 启动)
__global__ void pass2_global_reduce_kernel(const float* block_max, const float* block_sum, float* global_max, float* global_sum, int num_blocks) {
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // 这个 Kernel 只用 1 个 Block 跑，所以直接遍历前面的局部结果
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float b_max = block_max[i];
        float b_sum = block_sum[i];
        float new_max = fmaxf(local_max, b_max);
        local_sum = local_sum * expf(local_max - new_max) + b_sum * expf(b_max - new_max);
        local_max = new_max;
    }

    blockReduceOnlineSoftmax(local_max, local_sum);

    if (threadIdx.x == 0) {
        *global_max = local_max;
        *global_sum = local_sum;
    }
}

// Kernel 3: 第二次遍历显存，计算最终概率并写回 output
__global__ void pass3_write_output_kernel(const float* x, float* output, const float* global_max, const float* global_sum, int N) {
    float g_max = *global_max;
    float g_sum = *global_sum;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        output[i] = expf(x[i] - g_max) / g_sum;
    }
}

// =====================================================================
// Host 端启动逻辑
// =====================================================================

extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 256;
    // 限制最大 Block 数，防止海量资源消耗，剩下的数据交给 Kernel 里的 Grid-Stride 处理
    const int maxBlocks = 1024; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxBlocks) blocksPerGrid = maxBlocks;

    // 申请中间变量的显存
    float *d_block_max, *d_block_sum;
    float *d_global_max, *d_global_sum;
    cudaMalloc(&d_block_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_block_sum, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMalloc(&d_global_sum, sizeof(float));

    // Phase 1: 每个 Block 算出自己的 max 和 sum
    pass1_block_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_block_max, d_block_sum, N);

    // Phase 2: 用 1 个 Block 汇总出全局的 max 和 sum
    // 因为 blocksPerGrid 最大 1024，所以 1 个满载的 Block (1024 线程) 刚好能处理完
    pass2_global_reduce_kernel<<<1, 1024>>>(d_block_max, d_block_sum, d_global_max, d_global_sum, blocksPerGrid);

    // Phase 3: 映射输出概率分布
    pass3_write_output_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_global_max, d_global_sum, N);

    cudaDeviceSynchronize();

    cudaFree(d_block_max);
    cudaFree(d_block_sum);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
}