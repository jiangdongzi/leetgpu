#include <algorithm>
#include <cuda_runtime.h>

__global__ void cnt_arr_kernal(const int* input, const int N, const int K, int* output) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    int local_cnt = 0;
    for (int i = gtid; i < N; i += stride) {
        if (input[i] == K) {
            local_cnt++;
        }
    }
    __shared__ float sm[256];
    sm[tid] = local_cnt;
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

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int blocks = std::min(1024, (N + 255) / 256);
    cnt_arr_kernal<<<blocks, 256>>>(input, N, K, output);
}