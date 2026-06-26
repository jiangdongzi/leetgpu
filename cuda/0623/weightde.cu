#include <cuda_runtime.h>

__global__ void weight_de_kernal(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid >= M * N) return;
    const int col = gtid % N;
    const int row = gtid / N;
    const int SN = (N + TILE_SIZE - 1) / TILE_SIZE;
    Y[gtid] = X[gtid] * S[row / TILE_SIZE * SN + col / TILE_SIZE];
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int blocks = ((M * N) + 255) / 256;
    weight_de_kernal<<<blocks, 256>>>(X, S, Y, M, N, TILE_SIZE);
}