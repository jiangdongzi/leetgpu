#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


__global__ void dot_kernal(const half* A, const half* B, half* result, int N) {
    const int tid = threadIdx.x;
    const int gitd = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    float local_sum = 0.f;
    for (int i = gitd; i < N; i += stride) {
        local_sum += __half2float(A[i]) * __half2float(B[i]);
    }
    __shared__ float sm[1024];
    sm[tid] = local_sum;
    __syncthreads();
    for (int offset = 512; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] += sm[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(result, __float2half(sm[0]));
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    const int blocks = std::min(1024, (N + 1023) / 1024);
    dot_kernal<<<blocks, 1024>>>(A, B, result, N);
}