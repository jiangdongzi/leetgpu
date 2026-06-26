#include <cuda_runtime.h>

constexpr int TILE = 16;
__global__ void gemm_kernal(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];
    int acc = 0;
    const int row = threadIdx.y + blockIdx.y * TILE;
    const int col = threadIdx.x + blockIdx.x * TILE;
    const float scale = scale_A * scale_B / scale_C;
    for (int k = 0; k < K; k += TILE) {
        const int colA = k + threadIdx.x;
        if (row < M && colA < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + colA];
        } else {
            sA[threadIdx.y][threadIdx.x] = zero_point_A;
        }
        const int rowB = k + threadIdx.y;
        if (rowB < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[rowB * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = zero_point_B;
        }
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += (sA[threadIdx.y][i] - zero_point_A) * (sB[i][threadIdx.x] - zero_point_B);
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        const float tmp = acc * scale + zero_point_C;
        const int r = (int)rintf(tmp);
        if (r > 127) {
            C[row * N + col] = 127;
        } else if (r < -128) {
            C[row * N + col] = -128;
        } else {
            C[row * N + col] = r;
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blockdsPerGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernal<<<blockdsPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
}