#include <__clang_cuda_builtin_vars.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BM 16
#define BN 16
#define BK 64
#define PAD 1

__device__ __forceinline__ int decode_int4(uint8_t v) {
    return (int)v - 8;
}

__global__ void w4a16_matmul_kernel(
    const __half* __restrict__ x,        // [M, K]
    const uint8_t* __restrict__ w_q,     // [N, K / 2]
    const __half* __restrict__ scales,   // [N, K / group_size]
    __half* __restrict__ y,              // [M, N]
    int M, int N, int K,
    int group_size
) {
    int tx = threadIdx.x;   // 0 ~ 15，对应 N 方向
    int ty = threadIdx.y;   // 0 ~ 15，对应 M 方向

    int row = blockIdx.y * BM + ty;  // y 的 m
    int col = blockIdx.x * BN + tx;  // y 的 n

    __shared__ __half sx[BM][BK + PAD];
    __shared__ __half sw[BN][BK + PAD];

    float acc = 0.0f;

    int tid = ty * blockDim.x + tx;       // 0 ~ 255
    int num_threads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // -------------------------------
        // 1. load x tile: [BM, BK]
        // -------------------------------
        int total_x = BM * BK;

        for (int idx = tid; idx < total_x; idx += num_threads) {
            int r = idx / BK;
            int kk = idx % BK;

            int global_m = blockIdx.y * BM + r;
            int global_k = k0 + kk;

            if (global_m < M && global_k < K) {
                sx[r][kk] = x[global_m * K + global_k];
            } else {
                sx[r][kk] = __float2half(0.0f);
            }
        }

        // -------------------------------
        // 2. load and dequantize w tile: [BN, BK]
        //    注意 W 逻辑形状是 [N, K]
        // -------------------------------
        int total_w = BN * BK;

        for (int idx = tid; idx < total_w; idx += num_threads) {
            int n_inner = idx / BK;
            int kk = idx % BK;

            int global_n = blockIdx.x * BN + n_inner;
            int global_k = k0 + kk;

            __half val = __float2half(0.0f);

            if (global_n < N && global_k < K) {
                // w_q 每个 byte 存两个 int4
                int byte_idx = global_n * (K / 2) + global_k / 2;
                uint8_t packed = w_q[byte_idx];

                uint8_t nibble;
                if ((global_k & 1) == 0) {
                    // even k: high nibble
                    nibble = packed >> 4;
                } else {
                    // odd k: low nibble
                    nibble = packed & 0x0F;
                }

                int w_int = decode_int4(nibble);

                int scale_idx = global_n * (K / group_size)
                              + global_k / group_size;

                float s = __half2float(scales[scale_idx]);
                val = __float2half((float)w_int * s);
            }

            sw[n_inner][kk] = val;
        }

        __syncthreads();

        // -------------------------------
        // 3. 当前线程计算一个 y[row, col]
        // -------------------------------
        if (row < M && col < N) {
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float xv = __half2float(sx[ty][kk]);
                float wv = __half2float(sw[tx][kk]);
                acc += xv * wv;
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        y[row * N + col] = __float2half(acc);
    }
}

extern "C" void solve(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    __half* y,
    int M, int N, int K,
    int group_size
) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    w4a16_matmul_kernel<<<grid, block>>>(
        x, w_q, scales, y, M, N, K, group_size
    );
}