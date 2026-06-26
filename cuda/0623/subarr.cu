#include <algorithm>
#include <cuda_runtime.h>

__global__ void sum_arr(const int* input, const int N, int* output) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    int local_sum = 0;
    for (int i = gtid; i < N; i += stride) {
        local_sum += input[i];
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
        atomicAdd(output, sm[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    const int blocks = std::min(1024, (E - S + 1 + 255) / 256);
    sum_arr<<<blocks, 256>>>(input + S, E - S + 1, output);
}