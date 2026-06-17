#include <algorithm>
#include <cuda_runtime.h>

__global__ void sum_kernal(const int* input, int* output, const int N) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    int val = 0;
    for (int i = gtid; i < N; i += stride) {
        val += input[i];
    }
    extern __shared__ int sm[];
    const int tid = threadIdx.x;
    sm[tid] = val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] += sm[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sm[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    const int blocks = std::min(1024, (E - S + 1 + 255) / 256);
    sum_kernal<<<blocks, 256, 1024>>>(input + S, output, E - S + 1);
}
