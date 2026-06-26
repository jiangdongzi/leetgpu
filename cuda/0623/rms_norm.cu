#include <cuda_runtime.h>

__global__ void rms_kernal(const float* input, const int N, const float gamma, const float beta, const float eps, float* output) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    float local_sm = 0.f;
    for (int i = tid; i < N; i += stride) {
        local_sm += input[i] * input[i];
    }
    __shared__ float sm[1024];
    sm[tid] = local_sm;
    __syncthreads();
    for (int offset = 1024 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] += sm[tid + offset];
        }
        __syncthreads();
    }
    const float tmp = sm[0] / N + eps;
    const float b = rsqrtf(tmp);
    for (int i = tid; i < N; i += stride) {
        output[i] = b * input[i] * gamma + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    rms_kernal<<<1, 1024>>>(input, N, gamma, beta, eps, output);
}