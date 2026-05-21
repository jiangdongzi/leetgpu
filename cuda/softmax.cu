// #include <cuda_runtime.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <float.h>
#include <math.h>

__device__ __forceinline__ void solveAll(float& now_m,float& now_sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(0xffffffff, now_m, offset);
        float other_sum = __shfl_down_sync(0xffffffff, now_sum, offset);
        float max_temp = fmaxf(now_m, other_m);
        now_sum = now_sum * expf(now_m - max_temp) + other_sum * expf(other_m - max_temp);
        now_m = max_temp;
    }
}

// 存储所有线程块的max和sum
__device__ float g_block_max[1024]; // 足够容纳所有 SM 的输出
__device__ float g_block_sum[1024];


template<int threadsPerBlock>
__global__ void online_softmax_kernel(const float* x, float* output,  int N) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float mem[(threadsPerBlock/32) * 2];
    int tid = threadIdx.x;

    

    int stride = gridDim.x * blockDim.x;

    float max_self = -FLT_MAX;
    float sum_self = 0.0f;

    for (int i = idx; i < N; i += stride) {
        float val = x[i];
        float max_prev = max_self;
        max_self = fmaxf(max_prev, val);
        sum_self = sum_self * exp(max_prev - max_self) + exp(val - max_self);
    }
    solveAll(max_self, sum_self);
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = threadsPerBlock / 32;

    if (lane_id == 0) {
        mem[warp_id] = sum_self;
        mem[warp_id + num_warps] = max_self;
    }
    __syncthreads();


    if (warp_id == 0) {
        float b_sum = (tid < num_warps) ? mem[tid] : 0.0f;
        float b_max = (tid < num_warps) ? mem[tid + num_warps] : -FLT_MAX;

        solveAll(b_max, b_sum);

        if (tid == 0) {
            g_block_max[blockIdx.x] = b_max;
            g_block_sum[blockIdx.x] = b_sum;
        }
    }
    grid.sync();



    __shared__ float final_G_max, final_G_sum;

    if (threadIdx.x < 32) {
        float b_max = -FLT_MAX;
        float b_sum = 0.0f;

        for(int i = threadIdx.x; i < gridDim.x; i += 32) {
            float m = g_block_max[i];
            float s = g_block_sum[i];

            float m_prev = b_max;
            b_max = fmaxf(m_prev, m);
            b_sum = b_sum * exp(m_prev - b_max) + s * exp(m - b_max);
        }

        solveAll(b_max,b_sum);

        if (threadIdx.x == 0) {
            final_G_max = b_max;
            final_G_sum = b_sum;
        }
    }
    __syncthreads();


    for (int i = idx; i < N; i += stride) {
        output[i] = exp(x[i] - final_G_max) / final_G_sum;
    }

}



// x, output are device 
extern "C" void solve(const float* x, float* output, int N) {
    const int threadsPerBlock = 512;

    // 获取设备信息
    int device = 0;
    (cudaGetDevice(&device));

    int sm_count = 0;
    (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    int blocks_per_sm = 0;
    (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        online_softmax_kernel<threadsPerBlock>,
        threadsPerBlock,
        0
    ));
    

    // Cooperative launch 的 block 数不能超过设备能同时驻留的总 block 数。
    int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, sm_count * blocks_per_sm);

    // 准备 kernel 参数数组
    void* args[] = { (void*)&x, (void*)&output, (void*)&N };

    // 协作式内核启动
    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    (cudaLaunchCooperativeKernel(
        (void*)online_softmax_kernel<threadsPerBlock>,
        gridDim,
        blockDim,
        args,
        0,       // shared solveAllory
        0        // stream 0
    ));


}