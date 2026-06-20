#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int TILE = 16;

__global__ void matmul_kernal(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    const int m = blockIdx.y * TILE + threadIdx.y;
    const int n = blockIdx.x * TILE + threadIdx.x;
    const int kPacked = K / 2;
    const int kScales = K / group_size;
    __shared__ float sx[TILE][TILE];
    __shared__ float sw[TILE][TILE];
    float acc = 0.f;
    for (int k = 0; k < K; k += TILE) {
        const int xk = k + threadIdx.x;
        if (m < M && xk < K) {
            sx[threadIdx.y][threadIdx.x] = __half2float(x[m * K + xk]);
        } else {
            sx[threadIdx.y][threadIdx.x] = 0.f;
        }
        const int wk = k + threadIdx.y;
        if (wk < K && n < N) {
            const uint8_t byte = w_q[n * kPacked + wk / 2];
            const int nib = (wk & 1) ? (byte & 0xf) : (byte >> 4);
            const float s = __half2float(scales[n * kScales + wk / group_size]);
            sw[threadIdx.y][threadIdx.x] = (nib - 8) * s;
        } else {
            sw[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += sx[threadIdx.y][i] * sw[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (m < M && n < N) {
        y[m * N + n] = __float2half(acc);
    }
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernal<<<blocks, threads>>>(x, w_q, scales, y, M, N, K, group_size);
}