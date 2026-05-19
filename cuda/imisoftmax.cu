#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <functional>

__device__ __forceinline__ void warpReduceOnlineSoftmax(float& max_val, float& sum_val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        const float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
        const float new_max = fmaxf(max_val, other_max);
        sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

__device__ __forceinline__ void blockReduceOnlineSoftmax(float& val_max, float& val_sum) {
    warpReduceOnlineSoftmax(val_max, val_sum);

    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        smem_max[warp_id] = val_max;
        smem_sum[warp_id] = val_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        const int num_warps = blockDim.x / 32;
        val_max = (lane_id < num_warps) ? smem_max[lane_id] : -FLT_MAX;
        val_sum = (lane_id < num_warps) ? smem_sum[lane_id] : 0.0f;
        warpReduceOnlineSoftmax(val_max, val_sum);
    }
}

__global__ void pass1_block_reduce_kernel(const float* x, float* block_max, float* block_sum, int N) {
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        const float val = x[i];
        const float new_max = fmaxf(val, local_max);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    blockReduceOnlineSoftmax(local_max, local_sum);

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = local_max;
        block_sum[blockIdx.x] = local_sum;
    }

    blockReduceOnlineSoftmax(local_max, local_sum);
}

__global__ void pass2_global_reduce_kernel(const float* block_max, const float* block_sum, float* global_max, float* global_sum, const int num_blocks) {
    const int idx = threadIdx.x;
    float local_max = block_max[idx];
    float local_sum = block_sum[idx];

    blockReduceOnlineSoftmax(local_max, local_sum);

    if (threadIdx.x == 0) {
        *global_max = local_max;
        *global_sum = local_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = std::min(1024, (N + threadsPerBlock - 1) / threadsPerBlock);
    float* d_block_max, *d_block_sum;
    float *d_global_max, *d_global_sum;
    cudaMalloc(&d_block_max, sizeof(float) * blocksPerGrid);
    cudaMalloc(&d_block_sum, sizeof(float) * blocksPerGrid);
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMalloc(&d_global_sum, sizeof(float));
    pass1_block_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_block_max, d_block_sum, N);
}