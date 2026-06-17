#include <algorithm>
#include <cuda_runtime.h>

__global__ void count_ele_kernal(const int* input, int* output, int N, int K) {
    const int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int cnt = 0;
    for (int i = gtid; i < N; i += stride) {
        if (input[i] == K) cnt++;
    }
    extern __shared__ int sm[];
    const int tid = threadIdx.x;
    sm[tid] = cnt;
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

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int blocks = std::min(1024, (N + 255) / 256);
    count_ele_kernal<<<blocks, 256, 1024>>>(input, output, N, K);
}