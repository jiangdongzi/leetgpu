#include <cuda_runtime.h>

// 1. Warp 级别归约（利用寄存器通信，极速且无 shared memory 冲突）
__inline__ __device__ int warpReduceSum(int val) {
    // 每次将后半个 warp 的值累加到前半个 warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2. Block 级别归约
__inline__ __device__ int blockReduceSum(int val) {
    // 最多支持 32 个 Warp（即 1024 个 Thread/Block）
    static __shared__ int shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 第一步：Warp 内部归约
    val = warpReduceSum(val);

    // 第二步：每个 Warp 的第 0 个线程将结果写入 Shared Memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); // 确保所有 Warp 的部分和都已写入

    // 第三步：用第一个 Warp 读取 Shared Memory 并完成最终归约
    // 注意：只有当该线程对应的 Warp 实际存在时，才读取数据
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0;

    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

// 3. 核心 Kernel：网格跨步循环读取数据
__global__ void sum_kernel(const int* input, int* output, int count) {
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop：处理任意长度的数组，同时保证内存合并访问 (Coalesced Memory Access)
    for (int i = tid; i < count; i += stride) {
        sum += input[i];
    }

    // 在 Block 内求和
    sum = blockReduceSum(sum);

    // 只有每个 Block 的 0 号线程负责将该 Block 的最终结果原子加到全局内存
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// 4. 接口实现
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int count = E - S + 1;
    if (count <= 0) return;

    // 防御性编程：题目没有保证 output 初始化为 0，所以我们需要自己清零
    cudaMemset(output, 0, sizeof(int));

    // Block 大小设为 256 是个不错的经验值
    int blockSize = 256;
    
    // Grid 大小：1024 个 Block 足以打满现代 GPU 的 SM (Streaming Multiprocessors)
    // 配合 kernel 里的 grid-stride loop，可以自动处理远大于 1024 * 256 个元素的数据
    int numBlocks = 1024;

    // 指针偏移，直接从 input + S 开始读取，避免在 kernel 里做复杂的索引判断
    sum_kernel<<<numBlocks, blockSize>>>(input + S, output, count);
}