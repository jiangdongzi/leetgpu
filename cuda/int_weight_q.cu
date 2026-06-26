#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int TILE = 16;

__global__ void gemm_kernal(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, const int M, const int N, const int K, const int group_size) {
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    __shared__ float sX[TILE][TILE];
    __shared__ float sW[TILE][TILE];
    const int scale_cols = K / group_size;
    float acc = 0.f;
    for (int k = 0; k < K; k += TILE) {
        const int colA = k + threadIdx.x;
        if (row < M && colA < K) {
            sX[threadIdx.y][threadIdx.x] = __half2float(x[row * K + colA]);
        } else {
            sX[threadIdx.y][threadIdx.x] = 0.f;
        }
        const int rowW = k + threadIdx.y;
        if (rowW < K && col < N) {
            const uint8_t byte = w_q[col * (K / 2) + rowW / 2];
            const int nibble = (rowW & 1) ? (byte & 0xf) : (byte >> 4);
            const float s = __half2float(scales[col * scale_cols + rowW / group_size]);
            sW[threadIdx.y][threadIdx.x] = (nibble - 8) * s;
        } else {
            sW[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += sX[threadIdx.y][i] * sW[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        y[row * N + col] = __float2half(acc);
    }
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernal<<<blocks, threads>>>(x, w_q, scales, y, M, N, K, group_size);
}