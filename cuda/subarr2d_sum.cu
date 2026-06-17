#include <algorithm>
#include <cuda_runtime.h>

__global__ void sum_kernal(const int* input, int* output, int N, int S_COL,
                      int E_COL) {
    const int row = blockIdx.y;
    const int* arr = input + row * N + S_COL;
    const int total = E_COL - S_COL + 1;
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    int val = 0;
    for (int i = gtid; i < total; i += stride) {
        val += arr[i];
    }
    extern __shared__ int sm[];
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
extern "C" void solve(const int* input, int* output, int M, int N, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    dim3 gridDim(std::min(1024, (E_COL - S_COL + 1 + 255) / 256), E_ROW - S_ROW + 1);
    sum_kernal<<<gridDim, 256, 1024>>>(input + S_ROW * N, output, N, S_COL, E_COL);
}
