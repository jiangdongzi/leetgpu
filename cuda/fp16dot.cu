#include <__clang_cuda_builtin_vars.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void dot_kernal(const half* A, const half* B, float* result, int N) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    float sum = 0.f;
    for (int i = gtid; i < N; i += stride) {
        sum += __half2float(A[i] * B[i]);
    }
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float sm[32];
    if (lane_id == 0) {
        sm[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = sm[lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(result, sum);
        }
    }

}

__global__ void convert_half(const float* fp32_result, half* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *result = __float2half_rn(*fp32_result);
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    float* fp32_result = nullptr;
    cudaMalloc(&fp32_result, sizeof(float));
    cudaMemset(fp32_result, 0, sizeof(float));
    const int blocks = std::min(4096, (N + 1023) / 1024);
    dot_kernal<<<blocks, 1024>>>(A, B, fp32_result, N);
    convert_half<<<1, 1>>>(fp32_result, result);
    cudaFree(fp32_result);
}