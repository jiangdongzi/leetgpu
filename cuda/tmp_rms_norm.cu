#include <cuda_runtime.h>

__device__ __inline__ void warp_reduce(float& partioal_sum) {
    for (int offset = 16; offset > 0; offset /= 2) {
        partioal_sum += __shfl_down_sync(0xffffffff, partioal_sum, offset);
    }
}
__global__ void sum_kernal(const float* input, float* output_rms, int N) {

    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    float partial_sum = 0.f;
    for (int i = gtid; i < N; i += blockDim.x * gridDim.x) {
        const float tmp = input[i];
        partial_sum += tmp * tmp;
    }
    warp_reduce(partial_sum);
    __shared__ float sm[32];
    if (tid % 32 == 0) {
        sm[tid / 32] = partial_sum;
    }
    __syncthreads();
    if (tid < 32) {
        partial_sum = sm[tid];
    }
    if (tid < 32) {
        warp_reduce(partial_sum);
    }
    if (tid == 0) {
        atomicAdd(output_rms, partial_sum);
    }
}

// input, output are device pointers
__global__ void rms_kernal(const float* input, float gamma, float beta, float* output, int N,
                      float eps, const float* partial_sum) {
    const float rms = rsqrtf(*partial_sum / N + eps);
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    for (int i = gtid; i < N; i += blockDim.x * gridDim.x) {
        output[i] = gamma * input[i] * rms + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    const int block_size = 1024;
    int blocks = 128;
    const int need_block = (N + 1023) / 1024;
    if (need_block < 128) {
        blocks = need_block;
    }
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, 4);
    sum_kernal<<<blocks, block_size>>>(input, d_sum, N);
    rms_kernal<<<blocks, block_size>>>(input, gamma, beta, output, N, eps, d_sum);
    cudaFree(d_sum);
}