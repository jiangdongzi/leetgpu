#include <cuda_runtime.h>

// CUDA Kernel
__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    // 动态分配共享内存，用于存储当前 Block 的局部直方图
    extern __shared__ int shared_hist[];

    // 1. 并行初始化共享内存为 0
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    
    // 确保所有线程都完成了初始化
    __syncthreads();

    // 2. 利用 Grid-Stride Loop 读取全局内存，并在共享内存中统计
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        int bin = input[i];
        // 局部原子加，竞争仅发生在当前 SM 的 L1/Shared Memory 内部
        atomicAdd(&shared_hist[bin], 1);
    }
    
    // 确保当前 Block 内的所有线程都完成了局部统计
    __syncthreads();

    // 3. 将局部直方图的结果汇总到最终的全局直方图中
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        // 只有当局部计数大于 0 时才进行全局原子加，减少不必要的全局访存
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// Host 端调用接口
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // 1. 将 Global Memory 中的最终输出数组清零
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    // 2. 配置执行参数
    int threads_per_block = 256; 
    
    // 限制最大 Block 数量，依靠 Kernel 内的 Grid-Stride Loop 处理大数据量
    // 4096 是一个经验值，足以占满现代 GPU 的所有 SM 队列以实现高 Occupancy
    int max_blocks = 4096;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    // 共享内存大小（字节数）
    size_t shared_mem_bytes = num_bins * sizeof(int);

    // 3. 启动 Kernel
    histogram_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        input, histogram, N, num_bins
    );
}