#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------------
// Kernel 1: 计算平方和并使用 atomicAdd 归约到全局变量
// -------------------------------------------------------------------------
__global__ void rmsnorm_sum_kernel(const float* input, float* d_sum, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    // Grid-stride loop: 确保无论 N 多大，不管分配多少线程，都能正确覆盖所有元素，
    // 同时保证了全局内存访问的合并（Memory Coalescing）
    for (int i = tid; i < N; i += stride) {
        float val = input[i];
        local_sum += val * val;
    }

    // Warp 内规约 (Warp reduction): 属于同一 warp 的 32 个线程利用寄存器直接交换数据
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Block 内规约 (Block reduction)
    __shared__ float warp_sums[32]; // 假设最多分配 1024 个线程 = 32 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 每个 warp 的第一个线程将计算好的 warp_sum 写入共享内存
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // 唤醒第一个 warp 来归约所有的 warp_sums
    if (warp_id == 0) {
        local_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        // Block 中的 0 号线程将当前 block 的最终平方和原子加 (atomicAdd) 到全局内存
        if (lane_id == 0) {
            atomicAdd(d_sum, local_sum);
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 2: 计算真正的 RMS 倒数，并逐元素应用缩放和偏置
// -------------------------------------------------------------------------
__global__ void rmsnorm_apply_kernel(const float* input, float gamma, float beta, float* output, const float* d_sum, int N, float eps) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // 使用共享内存缓存该 block 需要用到的 rsqrt(RMS)，避免每个线程去读一遍全局内存
    __shared__ float rms_rsqrt;
    
    if (threadIdx.x == 0) {
        // 计算均方
        float mean_sq = (*d_sum) / N;
        // rsqrtf 对应底层的近似平方根倒数指令 (MUFU.RSQ)，比 1.0f/sqrtf(...) 快得多
        rms_rsqrt = rsqrtf(mean_sq + eps);
    }
    __syncthreads(); // 确保所有线程都看到了 rms_rsqrt

    float r = rms_rsqrt; // 存入寄存器
    for (int i = tid; i < N; i += stride) {
        // 如果 input 和 output 的指针相同 (In-place 计算)，这里也是完全安全的
        output[i] = input[i] * r * gamma + beta;
    }
}

// -------------------------------------------------------------------------
// 主机端接口函数
// -------------------------------------------------------------------------
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    if (N <= 0) return;

    // 为规约结果分配 1 个 float 大小的显存，并初始化为 0
    float* d_sum = nullptr;
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    // 线程块配置策略
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // 限制最大 block 数量，防止对极大的 N 创建过多 block 导致 atomicAdd 严重冲突。
    // 256 个 block (共 65536 个线程) 通常足够吃满现代 GPU 的访存带宽。
    if (blocks > 256) {
        blocks = 256;
    }

    // Step 1: 启动第一个 Kernel 算局部和并规约
    rmsnorm_sum_kernel<<<blocks, threads>>>(input, d_sum, N);

    // Step 2: 启动第二个 Kernel 计算 RMS 并写回 output
    rmsnorm_apply_kernel<<<blocks, threads>>>(input, gamma, beta, output, d_sum, N, eps);

    // 回收显存 (如果是严苛的延迟敏感场景，生产环境中 d_sum 通常会预先分配好)
    cudaFree(d_sum);
}