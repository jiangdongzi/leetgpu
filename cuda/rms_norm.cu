#include <algorithm>
#include <cuda_runtime.h>

__device__ float sum = 0.f;
__global__ void sum_kernal(const float* input, int N) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    float local_sum = 0.f;
    for (int i = gtid; i < N; i += stride) {
        const float val = input[i];
        local_sum += val * val;
    }
    __shared__ float sm[256];
    sm[tid] = local_sum;
    __syncthreads();
    for (int offset = 256 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] += sm[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&sum, sm[0] / N);
    }
}

__global__ void rms_kernal(const float* input, int N, const float gamma, const float beta, const float eps, float* output) {
    const float rms = rsqrt(sum + eps);
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        output[i] = input[i] * rms * gamma + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    const int blocks = std::min(1024, (N + 255) / 256);
    const float zero = 0.f;
    cudaMemcpyToSymbol(sum, &zero, sizeof(float));
    sum_kernal<<<blocks, 256>>>(input, N);
    rms_kernal<<<blocks, 256>>>(input, N, gamma, beta, eps, output);
}