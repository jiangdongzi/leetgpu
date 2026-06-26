#include <cuda_runtime.h>

__device__ void arr_sum(const int* input, const int N, int* outpu) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    int local_sum = 0;
    for (int i = tid; i < N; i += stride) {
        local_sum += input[i];
    }
    __shared__ int sm[256];
    sm[tid] = local_sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) {
            sm[tid] += sm[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(outpu, sm[0]);
    }
}

__global__ void arr_sum_kernal(const int* input, const int M, const int S, const int E, int* output) {
    const int b = blockIdx.x;
    const int* myInput = input + b * M + S;
    arr_sum(myInput, E - S + 1, output);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    arr_sum_kernal<<<E_ROW - S_ROW + 1, 256>>>(input + S_ROW * M, M, S_COL, E_COL, output);
}
