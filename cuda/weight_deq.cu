#include <cuda_runtime.h>

__global__ void dec_kernal(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total = M * N;
    if (gtid >= total) return;
    const int col = gtid % N;
    const int row = gtid / N;
    const int s_r = row / TILE_SIZE, s_c = col / TILE_SIZE;
    const int s_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    Y[gtid] = X[gtid] * S[s_r * s_n + s_c];
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int total = M * N;
    dec_kernal<<<(total + 255) / 256, 256>>>(X, S, Y, M, N, TILE_SIZE);
}