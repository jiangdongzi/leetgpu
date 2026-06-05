#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrix_transpose_kernel(const float* input, float* output, const int M, const int N) {
    __shared__ float sm[TILE_DIM][TILE_DIM + 1];
    const int gtx = threadIdx.x + blockIdx.x * blockDim.x;
    const int gty = threadIdx.y + blockIdx.y * blockDim.y;
    if (gtx < N && gty < M) {
        const int idx = gty * N + gtx;
        sm[threadIdx.y][threadIdx.x] = input[idx];
    }

    __syncthreads();

    const int output_x = blockIdx.y * blockDim.y + threadIdx.x;
    const int output_y = blockDim.x * blockIdx.x + threadIdx.y;

    if (output_y < M && output_x < N) {
        output[output_y * M + output_x] = sm[threadIdx.x][threadIdx.y];
    }
}