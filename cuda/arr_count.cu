#include <cuda_runtime.h>

// Warp 级归约辅助函数
__device__ __forceinline__ int warpReduceSum(int val) {
    unsigned int active_mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(active_mask, val, offset);
    }
    return val;
}

__global__ void countKernel(const int* input, int* output, int N, int K) {
    // 1. 寄存器局部累加 (Grid-Stride Loop)
    int local_count = 0;
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_id; i < N; i += stride) {
        if (input[i] == K) {
            local_count++;
        }
    }

    // 2. Warp 级归约
    int warp_sum = warpReduceSum(local_count);

    // 3. Block 级归约
    // 假设 blockDim.x 最大为 1024，最多有 32 个 Warp
    __shared__ int shared_warp_sums[32];
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // 每个 Warp 的第一个线程将结果写入 Shared Memory
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads(); // 确保所有 Warp 都写入完毕

    // 4. 由第一个 Warp 汇总整个 Block 的结果
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        // 读取该 Block 内有效 Warp 的和，超出的部分补 0
        int block_sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0;
        
        // 第一个 Warp 再做一次归约
        block_sum = warpReduceSum(block_sum);

        // 5. 仅由 Block 的 0 号线程执行全局原子加
        if (tid == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

extern "C" void solve(const int* input, int* output, int N, int K) {
    // 务必记得初始化输出指针指向的内存
    cudaMemset(output, 0, sizeof(int));

    // 如果 N 为 0，直接返回
    if (N <= 0) return;

    // 配置 Kernel 执行参数
    int threadsPerBlock = 256; 
    
    // 计算需要的 Block 数量，并设置一个合理的上限 (例如 4096) 
    // 这样配合 Grid-Stride Loop 足以跑满现代 GPU 的 SM 和访存带宽
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) {
        blocksPerGrid = 4096;
    }

    // 启动 Kernel
    countKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    
    // 确保执行完毕（平台可能在外部做同步，但自己加上更稳妥）
    cudaDeviceSynchronize();
}