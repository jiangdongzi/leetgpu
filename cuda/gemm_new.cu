#include <cuda_runtime.h>

constexpr int BLOCKS_M = 128;
constexpr int BLOCKS_N = 128;

constexpr int BLOCKS = 256;
constexpr int BLOCKS_K = 8;

constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCKS / THREAD_A_LAYOUT_X;

constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCKS / THREAD_B_LAYOUT_X;

constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCKS / THREAD_C_LAYOUT_X;

constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

constexpr int THREAD_C_TILE_SIZE_X = 8;
constexpr int THREAD_C_TILE_SIZE_Y = 8;

#define FLOAT4(x) *(reinterpret_cast<float4*>(&x))

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {

    const int row0 = blockIdx.x * BLOCKS_M;
    const int col0 = blockIdx.y * BLOCKS_N;
    const int tidx = threadIdx.x;

    const int tAx = tidx % THREAD_A_LAYOUT_X;
    const int tAy = tidx / THREAD_A_LAYOUT_X;

    const int tBx = tidx % THREAD_B_LAYOUT_X;
    const int tBy = tidx / THREAD_B_LAYOUT_X;

    const int warp_id = tidx / 32;
    const int lane_id = tidx % 32;

    const int warp_x = warp_id % THREAD_C_WARP_DIM_X;
    const int warp_y = warp_id / THREAD_C_WARP_DIM_X;

    const int lane_x = (lane_id & 15) >> 1;
    const int lane_y = ((lane_id >> 4) << 1) + (lane_id & 1);
    const int tCy = warp_y * THREAD_C_WARP_Y + lane_y;
    const int tCx = warp_x * THREAD_C_WARP_X + lane_x;
    float tCsA[THREAD_C_TILE_SIZE_X];
    float tCsB[THREAD_C_TILE_SIZE_Y];
    __shared__ float sA[BLOCKS_K][BLOCKS_M];
    __shared__ float sB[BLOCKS_K][BLOCKS_N];
    float acc[THREAD_C_TILE_SIZE_Y][THREAD_C_TILE_SIZE_X] = {0.f};

    for (int k = 0; k < K; k += BLOCKS_K) {
        for (int i = 0; i < BLOCKS_M; i += THREAD_A_LAYOUT_Y) {
            const int r = row0 + i + tAy;
            const int c = k + tAx;
            if (r < M && c < K) {
                sA[tAx][(i + tAy) ^ (tAx << 2)] = A[r * K + c];
            } else {
                sA[tAx][(i + tAy) ^ (tAx << 2)] = 0.f;
            }
        }
        for (int i = 0; i < BLOCKS_K; i += THREAD_B_LAYOUT_Y) {
            const int r = k + i + tBy;
            for (int j = 0; j < BLOCKS_N; j += THREAD_B_LAYOUT_X) {
                const int c = col0 + j + tBx;
                if (r < K && c < N) {
                    sB[i + tBy][j + tBx] = B[r * N + c];
                } else {
                    sB[i + tBy][j + tBx] = 0.f;
                }
            }
        }
        __syncthreads();
        for (int tk = 0; tk < BLOCKS_K; tk++) {
            for (int tm = 0; tm < (THREAD_C_TILE_SIZE_Y >> 2); tm++) {
                const int c = tk;
                const int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                FLOAT4(tCsA[tm << 2]) = FLOAT4(sA[c][r ^ (c << 2)]);
            }
            for (int tn = 0; tn < (THREAD_C_TILE_SIZE_X >> 2); tn++) {
                const int r = tk;
                const int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                FLOAT4(tCsB[tn << 2]) = FLOAT4(sB[r][c]);
            }
            for (int tm = 0; tm < THREAD_C_TILE_SIZE_Y; tm++) {
                for (int tn = 0; tn < THREAD_C_TILE_SIZE_X; tn++) {
                    acc[tm][tn] += tCsA[tm] * tCsB[tn];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < THREAD_C_TILE_SIZE_Y; i++) {
        const int r = row0 + (tCy << 2) + (i & ~3) * THREAD_C_LAYOUT_Y + (i & 3);
        for (int j = 0; j < THREAD_C_TILE_SIZE_X; j++) {
            const int c = col0 + (tCx << 2) + (j & ~3) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCKS);
    dim3 blocksPerGrid((M + BLOCKS_M - 1) / BLOCKS_M,
                       (N + BLOCKS_N - 1) / BLOCKS_N);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}