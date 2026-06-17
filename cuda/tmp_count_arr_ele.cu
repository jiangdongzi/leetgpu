#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void warp_reduce(int& val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

void __global__ count_ele_kernal(const int* input, int* output, int N, int K) {
    int cnt = 0;
    const int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = gtid; i < N; i += blockDim.x * gridDim.x) {
        if (input[i] == K) {
            cnt++;
        }
    }
    const int tid = threadIdx.x;
    __shared__ int sm[1024];
    sm[tid] = cnt;
    __syncthreads();
    for (int i = 1024 / 2; i > 0; i /= 2) {
        if (tid < i) {
            sm[tid] += sm[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sm[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int block_size = 1024;
    int blocks = 1024;
    if ((N + 1023) / 1024 < blocks) {
        blocks = (N + 1023) / 1024;
    }
    count_ele_kernal<<<blocks, block_size>>>(input, output, N, K);
}