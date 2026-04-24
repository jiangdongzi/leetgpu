#include <cuda_runtime.h>
#include <limits.h>

// 1. 初始化 Kernel：避免使用带来 Host-Device 同步开销的 cudaMemcpy
__global__ void init_output(int* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = INT_MIN;
    }
}

// 2. 核心计算 Kernel：并行计算 + 块内规约 (Block Reduction)
__global__ void max_subarray_kernel(const int* input, int* output, int N, int window_size, int total_windows) {
    // 全局线程索引与局部线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // 默认赋极小值，这一步非常关键：可以自动处理最后一个没有被填满的 Block 的边界问题
    int my_sum = INT_MIN;

    // 每个线程负责一个起始位置的 Window Sum
    if (tid < total_windows) {
        int sum = 0;
        // 朴素求和，依赖 GPU L1/L2 Cache 扛住重复读取
        for (int i = 0; i < window_size; ++i) {
            sum += input[tid + i];
        }
        my_sum = sum;
    }

    // 分配共享内存用于 Block 级别的 Tree Reduction
    extern __shared__ int shared_max[];
    shared_max[local_tid] = my_sum;
    __syncthreads(); // 等待 Block 内所有线程将数据写入 Shared Memory

    // 并行归约：求当前 Block 的局部最大值
    // 要求 blockDim.x 是 2 的幂（此处我们设置了 256）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            if (shared_max[local_tid + stride] > shared_max[local_tid]) {
                shared_max[local_tid] = shared_max[local_tid + stride];
            }
        }
        __syncthreads(); // 确保每一层规约完成
    }

    // 最终：每个 Block 只有 thread 0 去写全局内存的 output
    // 极大减少了 atomicMax 对全局同一个内存地址的锁竞争冲突
    if (local_tid == 0 && shared_max[0] != INT_MIN) {
        atomicMax(output, shared_max[0]);
    }
}

// 3. 主入口函数
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int total_windows = N - window_size + 1;
    if (total_windows <= 0) return;

    // 启动异步初始化
    init_output<<<1, 1>>>(output);

    // 设置 Kernel 参数
    int threads_per_block = 256;
    int blocks = (total_windows + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(int);

    // 启动核心 Kernel
    max_subarray_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input, output, N, window_size, total_windows
    );

    // 面试时建议加上同步，确保异步 Kernel 执行完毕后交回控制权
    cudaDeviceSynchronize();
}