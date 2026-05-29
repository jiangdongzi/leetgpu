#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>


__inline__ __device__ float warp_reduce_online(float& max_val, float& sum_val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
        const float tmp_max = fmaxf(other_max, max_val);
        sum_val = sum_val * expf(max_val - tmp_max) + other_sum * expf(other_max - max_val);
    }
}

__global__ void block_reduce_online(const float * const input, const int N, float* output_max, float* output_sum) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int gtid = 1024 * blockIdx.x + warp_id * 32 + lane_id;
    float max_val = -2e38f;
    float sum_val = 0.f;
    if (gtid < N) {
        max_val = input[gtid];
        sum_val = 1.f;
    }
    warp_reduce_online(max_val, sum_val);
    static __shared__ float smem_max[32];
    static __shared__ float smem_sum[32];
    if (lane_id == 0) {
        smem_max[lane_id] = max_val; 
        smem_sum[lane_id] = sum_val; 
    }
    __syncthreads();
    if (warp_id == 0) {
        max_val = smem_max[lane_id];
        sum_val = smem_sum[lane_id];
        warp_reduce_online(max_val, sum_val);
        if (lane_id == 0) {
            output_max[blockIdx.x] = max_val;
            output_sum[blockIdx.x] = sum_val;
        }
    }
}

__global__ void block_reduce_online(const float * const input_max, const float* const input_sum, const int N, float* output_max, float* output_sum) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int gtid = 1024 * blockIdx.x;
    float max_val = -2e38f;
    float sum_val = 0.f;
    if (gtid < N) {
        max_val = input_max[gtid];
        sum_val = input_sum[gtid];
    }
    warp_reduce_online(max_val, sum_val);
    static __shared__ float smem_max[32];
    static __shared__ float smem_sum[32];
    if (lane_id == 0) {
        smem_max[lane_id] = max_val; 
        smem_sum[lane_id] = sum_val; 
    }
    __syncthreads();
    if (warp_id == 0) {
        max_val = smem_max[lane_id];
        sum_val = smem_sum[lane_id];
        warp_reduce_online(max_val, sum_val);
        if (lane_id == 0) {
            output_max[blockIdx.x] = max_val;
            output_sum[blockIdx.x] = sum_val;
        }
    }
}

float global_max, global_sum;

void scan_block(const float* input_max, const float* input_sum, const int N) {
    dim3 threads(32, 32);
    const int blocks = (N + 1023) / 1024;
    float* output_max;
    cudaMalloc(&output_max, 4 * blocks);
    float* output_sum;
    cudaMalloc(&output_sum, 4 * blocks);
    block_reduce_online<<<blocks, threads>>>(input_max, input_sum, N, output_max, output_sum);
    if (blocks == 1) {
        global_max = output_max[0];
        global_sum = output_sum[0];
    } else {
        scan_block(output_max, output_sum, blocks);
    }
    //free mem
}

__global__ void normalize_kernel(const float* x, float* output, int N) {
    const int tid = threadIdx.x;
    const int bId = blockIdx.x;
    const int gtid = bId * 1024 + tid;
    if (gtid < N) {
        output[gtid] = expf(x[gtid] - global_max) / global_sum;
    }
}

extern "C" void solve(const float* x, float* output, int N) {
    dim3 threads(32, 32);
    const int blocks = (N + 1023) / 1024;
    float* output_max;
    cudaMalloc(&output_max, 4 * blocks);
    float* output_sum;
    cudaMalloc(&output_sum, 4 * blocks);
    block_reduce_online<<<blocks, threads>>>(x, N, output_max, output_sum);
    scan_block(output_max, output_sum, blocks);
    //free
}

int main() {
    return 0;
}