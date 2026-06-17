#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

__global__ void weight_dec_kernal(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (gtid >= total) return;
    const int row = gtid / N;
    const int col = gtid % N;
    const int scale_r = row / TILE_SIZE;
    const int scale_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
    const float scale = S[scale_r * scale_cols + col / TILE_SIZE];
    Y[gtid] = X[gtid] * scale;
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int total = M * N;
    weight_dec_kernal<<<(total + 255) / 256, 256>>>(X, S, Y, M, N, TILE_SIZE);
}