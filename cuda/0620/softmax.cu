#include <cfloat>
#include <cmath>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

__device__ float g_max_val;
__device__ float g_sum_val;
__device__ void merge(float& a_max, float& a_sum, const float b_max, const float b_sum) {
    const float max_val = fmaxf(a_max, b_max);
    a_sum = expf(a_max - max_val) * a_sum + expf(b_max - max_val) * b_sum;
    a_max = max_val;
}

__device__ void block_reduce(float& max_val, float& sum_val) {
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    const int tid = threadIdx.x;
    s_max[tid] = max_val;
    s_sum[tid] = sum_val;
    __syncthreads();
    for (int offset = 256; offset > 0; offset /= 2) {
        if (tid < offset) {
            const float other_max = s_max[tid + offset];
            const float other_sum = s_sum[tid + offset];
            float local_max_s = s_max[tid];
            float local_sum_s = s_sum[tid];
            merge(local_max_s, local_sum_s, other_max, other_sum);
            s_max[tid] = local_max_s;
            s_sum[tid] = local_sum_s;
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_val = s_max[0];
        sum_val = s_sum[0];
    }
}

__global__ void block_softmax_kernel(const float* input, const int N, float* block_max, float* block_sum) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float local_max = -FLT_MAX;
    float local_sum = 0.f;
    for (int i = gtid; i < N; i += stride) {
        merge(local_max, local_sum, input[i], 1);
    }
    block_reduce(local_max, local_sum);
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid == 0) {
        block_max[b] = local_max;
        block_sum[b] = local_sum;
    }
}

__global__ void global_softmax_kernal(const float* block_max, const float* block_sum, const int N) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    float local_max = -FLT_MAX;
    float local_sum = 0.f;
    for (int i = gtid; i < N; i += stride) {
        merge(local_max, local_sum, block_max[i], block_sum[i]);
    }
    block_reduce(local_max, local_sum);
    if (tid == 0) {
        g_max_val = local_max;
        g_sum_val = local_sum;
    }
}

__global__ void final(const float* input, float* output, const int N) {
    const float max_val = g_max_val;
    const float sum_val = g_sum_val;
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    float* block_max, *block_sum;
    cudaMalloc(&block_max, sizeof(float) * blocksPerGrid);
    cudaMalloc(&block_sum, sizeof(float) * blocksPerGrid);

    block_softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N, block_max, block_sum);
    global_softmax_kernal<<<std::min(1024, (blocksPerGrid + 255) / 256), threadsPerBlock>>>(block_max, block_sum, blocksPerGrid);
    final<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaFree(block_max);
    cudaFree(block_sum);
    cudaDeviceSynchronize();
}