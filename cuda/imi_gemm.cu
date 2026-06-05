#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__ void gemm_kernal(const float* A, const float* B, float* C, const int M, const int N, const int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    const int aRow = blockDim.y * blockIdx.y + threadIdx.y;
    const int bCol = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.f;
    for (int i = 0; i < N; i += TILE_SIZE) {
        const int aCol = i + threadIdx.x;
        if (aRow < M && aCol < N) {
            sA[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();
        const int bRow = i + threadIdx.y;
        if (bRow < N && bCol < K) {
            sB[threadIdx.y][threadIdx.x] = B[bRow * K + bCol];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++) {
            val += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (aRow < M && bCol < K) {
        C[aRow * K + bCol] = val;
    }
}