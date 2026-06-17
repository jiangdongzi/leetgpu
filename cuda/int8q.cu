#include <cstdint>
#include <cuda_runtime.h>

constexpr int TILE = 16;
__global__ void gemm_kernal(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];
    int acc = 0;
    const int rowA = blockIdx.y * blockDim.y + threadIdx.y;
    const int colB = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < K; i += TILE) {
        const int colA = i + threadIdx.x;
        if (rowA < M && colA < K) {
            sA[threadIdx.y][threadIdx.x] = A[rowA * K + colA];
        } else {
            sA[threadIdx.y][threadIdx.x] = zero_point_A;
        }
        const int rowB = i + threadIdx.y;
        if (rowB < K && colB < N) {
            sB[threadIdx.y][threadIdx.x] = B[rowB * N + colB];
        } else {
            sB[threadIdx.y][threadIdx.x] = zero_point_B;
        }
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += (sA[threadIdx.y][i] - zero_point_A) * (sB[i][threadIdx.x] - zero_point_B);
        }
        __syncthreads();
    }
    if (rowA < M && colB < N) {
        const float tmp = (float)acc * scale_A * scale_B / scale_C + (float)zero_point_C;
        const int r = (int)rintf(tmp);
        C[rowA * N + colB] = r > 127 ? 127 : (r < -128 ? -128 : r);
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + 15) / TILE, (M + 15) / TILE);
    gemm_kernal<<<gridDim, blockDim>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
}