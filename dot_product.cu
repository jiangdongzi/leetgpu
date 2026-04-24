#include <cuda_runtime.h>

// 1. 线程束级别归约 (Warp-Level Reduction)
// 使用 __shfl_down_sync 替代共享内存，极大地减少延迟和同步开销
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2. 线程块级别归约 (Block-Level Reduction)
__inline__ __device__ float blockReduceSum(float val) {
    // 假设最大 blockDim.x 为 1024，最多 32 个 Warp
    static __shared__ float shared[32]; 
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 先在各个 Warp 内部求和
    val = warpReduceSum(val);

    // 每个 Warp 的第 0 个线程将结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); // 等待所有 Warp 写入完毕

    // 取出共享内存中的值，交由第一个 Warp 进行最终求和
    // 如果当前线程所在的 lane 超出了实际的 Warp 数量，则补 0
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// 3. 核心 Kernel 函数
__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 网格跨步循环：处理 N 远大于线程总数的情况，保证内存合并访问
    for (int i = idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    // 块内规约求和
    sum = blockReduceSum(sum);

    // 每个 Block 仅由 0 号线程执行一次全局内存的原子加
    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

// 4. 接口实现
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // 必须先在 Device 端将 result 初始化为 0
    cudaMemset(result, 0, sizeof(float));

    // 启动配置优化
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 限制最大 Block 数量，避免给 L2 Cache 和原子操作带来过大压力
    // 现代 GPU SM 数量有限，适当的 Block 数量配合 Grid-Stride Loop 效率最高
    int maxBlocks = 1024; 
    if (blocksPerGrid > maxBlocks) {
        blocksPerGrid = maxBlocks;
    }

    dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
}