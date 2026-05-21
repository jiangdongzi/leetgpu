// #include <cuda_runtime.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <float.h>
#include <math.h>

// 在一个 warp 内归约一组 online softmax 的状态二元组：
//   now_m   : 当前分片里的最大值
//   now_sum : 以 now_m 为基准的 exp 和，也就是 sum(exp(x_i - now_m))
//
// 合并两个分片时，新的最大值可能会变大。为了保持数值稳定，旧的 sum 需要
// 重新缩放到新的最大值基准下：
//   sum_new = sum_a * exp(max_a - max_new) + sum_b * exp(max_b - max_new)
//
// __shfl_down_sync 每轮让线程拿到同一个 warp 内 offset 距离处线程的状态，
// offset 从 16 到 1，最终 lane 0 会得到整个 warp 的归约结果。
__device__ __forceinline__ void solveAll(float& now_m,float& now_sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(0xffffffff, now_m, offset);
        float other_sum = __shfl_down_sync(0xffffffff, now_sum, offset);
        float max_temp = fmaxf(now_m, other_m);
        now_sum = now_sum * expf(now_m - max_temp) + other_sum * expf(other_m - max_temp);
        now_m = max_temp;
    }
}

// 存储每个 block 计算出来的局部 max 和局部 sum。
// 后面会通过 grid 级同步，确保所有 block 都写完后再读取这些数组做全局归约。
__device__ float g_block_max[1024]; // 需要能容纳本 kernel 启动的所有 block。
__device__ float g_block_sum[1024];


template<int threadsPerBlock>
__global__ void online_softmax_kernel(const float* x, float* output,  int N) {
    // cooperative_groups::this_grid() 只有在 cooperative launch 下才能用于
    // grid.sync()。它提供跨整个 grid 的同步能力，不只是 block 内同步。
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // 当前线程负责的第一个全局元素下标。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个 warp 最终只把 lane 0 的归约结果写到共享内存。
    // 前半段存每个 warp 的 sum，后半段存每个 warp 的 max。
    __shared__ float mem[(threadsPerBlock/32) * 2];
    int tid = threadIdx.x;

    

    // grid-stride loop 的跨度。这样即使 N 大于总线程数，每个线程也能处理多个元素。
    int stride = gridDim.x * blockDim.x;

    // 当前线程自己的 online softmax 状态。
    // max_self 是该线程遍历到的元素最大值；
    // sum_self 是以 max_self 为基准的 exp 和。
    float max_self = -FLT_MAX;
    float sum_self = 0.0f;

    // 第一级：每个线程用 grid-stride loop 扫自己的元素子集。
    // 这里在线更新 max 和 sum，避免 exp(x) 溢出，也不需要先单独扫一遍求 max。
    for (int i = idx; i < N; i += stride) {
        float val = x[i];
        float max_prev = max_self;
        max_self = fmaxf(max_prev, val);
        sum_self = sum_self * exp(max_prev - max_self) + exp(val - max_self);
    }

    // 第二级：warp 内归约。归约后每个 warp 的 lane 0 拿到该 warp 的局部结果。
    solveAll(max_self, sum_self);
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = threadsPerBlock / 32;

    // 每个 warp 只由 lane 0 写一次，减少共享内存写入量。
    if (lane_id == 0) {
        mem[warp_id] = sum_self;
        mem[warp_id + num_warps] = max_self;
    }
    // 确保所有 warp 的 lane 0 都已经把结果写入 mem，后面 warp 0 才能读取。
    __syncthreads();


    // 第三级：用 block 内的第一个 warp，把所有 warp 的结果继续归约成 block 结果。
    if (warp_id == 0) {
        float b_sum = (tid < num_warps) ? mem[tid] : 0.0f;
        float b_max = (tid < num_warps) ? mem[tid + num_warps] : -FLT_MAX;

        solveAll(b_max, b_sum);

        // 每个 block 的 tid 0 写出该 block 的局部 max/sum，供全局归约使用。
        if (tid == 0) {
            g_block_max[blockIdx.x] = b_max;
            g_block_sum[blockIdx.x] = b_sum;
        }
    }

    // grid 级同步：等待所有 block 都写完 g_block_max/g_block_sum。
    // 这是本实现能在单个 kernel 内完成跨 block 归约的关键。
    // 普通 kernel launch 不能安全使用这个同步，必须用 cudaLaunchCooperativeKernel。
    grid.sync();



    // 每个 block 都会在自己的共享内存里保存一份最终全局 max/sum。
    // 后续该 block 的所有线程都用这两个值计算自己的 output。
    __shared__ float final_G_max, final_G_sum;

    // 第四级：每个 block 的前 32 个线程读取所有 block 的结果，
    // 再做一次 warp 内归约，得到整个输入向量的全局 max/sum。
    // 这一步在每个 block 中重复执行一次，换来后面每个 block 都有本地 shared 结果可用。
    if (threadIdx.x < 32) {
        float b_max = -FLT_MAX;
        float b_sum = 0.0f;

        // 如果 block 数超过 32，单个线程会以步长 32 处理多个 block 的结果。
        for(int i = threadIdx.x; i < gridDim.x; i += 32) {
            float m = g_block_max[i];
            float s = g_block_sum[i];

            float m_prev = b_max;
            b_max = fmaxf(m_prev, m);
            b_sum = b_sum * exp(m_prev - b_max) + s * exp(m - b_max);
        }

        solveAll(b_max,b_sum);

        // lane 0 得到全局归约结果，写入本 block 的共享内存。
        if (threadIdx.x == 0) {
            final_G_max = b_max;
            final_G_sum = b_sum;
        }
    }
    // 确保本 block 的所有线程都能看到 final_G_max/final_G_sum。
    __syncthreads();


    // 最后一级：用全局 max/sum 计算 softmax。
    // softmax(x_i) = exp(x_i - global_max) / sum_j exp(x_j - global_max)
    for (int i = idx; i < N; i += stride) {
        output[i] = exp(x[i] - final_G_max) / final_G_sum;
    }

}



// x, output are device 
extern "C" void solve(const float* x, float* output, int N) {
    // 每个 block 512 个线程，也就是 16 个 warp。
    // threadsPerBlock 是模板参数，用来让 kernel 内共享内存大小在编译期确定。
    const int threadsPerBlock = 512;

    // 获取设备信息
    int device = 0;
    (cudaGetDevice(&device));

    int sm_count = 0;
    (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    // 计算在当前 kernel 配置下，每个 SM 最多能同时驻留多少个 block。
    // cooperative kernel 要求所有 block 能同时驻留，否则 grid.sync() 可能死锁。
    int blocks_per_sm = 0;
    (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        online_softmax_kernel<threadsPerBlock>,
        threadsPerBlock,
        0
    ));
    

    // Cooperative launch 的 block 数不能超过设备能同时驻留的总 block 数。
    // 同时也不需要超过覆盖 N 所需的 block 数。
    int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, sm_count * blocks_per_sm);

    // 准备 kernel 参数数组
    void* args[] = { (void*)&x, (void*)&output, (void*)&N };

    // 协作式内核启动。只有这种启动方式下，kernel 内的 grid.sync() 才合法。
    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    (cudaLaunchCooperativeKernel(
        (void*)online_softmax_kernel<threadsPerBlock>,
        gridDim,
        blockDim,
        args,
        0,       // dynamic shared memory，本 kernel 没有额外动态共享内存。
        0        // stream 0
    ));


}