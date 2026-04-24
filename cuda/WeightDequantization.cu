#include <cuda_runtime.h>

// CUDA Kernel：执行实际的反量化计算
__global__ void dequantize_kernel(
    const float* __restrict__ X,
    const float* __restrict__ S,
    float* __restrict__ Y,
    int M, int N, int TILE_SIZE, int S_cols)
{
    // 获取全局线程的 2D 坐标 (对应矩阵的 col 和 row)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，防止越界访存
    if (row < M && col < N) {
        // 计算当前元素对应的 Scale 矩阵的行号和列号
        int s_row = row / TILE_SIZE;
        int s_col = col / TILE_SIZE;

        // 计算物理内存中的一维扁平化索引
        int x_idx = row * N + col;
        int s_idx = s_row * S_cols + s_col;

        // 执行反量化：读取 X 和 S，相乘并写回 Y
        Y[x_idx] = X[x_idx] * S[s_idx];
    }
}

// Host 端入口函数，需保持函数签名不变
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    // 1. 定义线程块维度 (Block Size)
    // 32x32 = 1024 线程/Block，能够很好的隐藏访存延迟并保持高 Occupancy
    dim3 block(32, 32);

    // 2. 定义网格维度 (Grid Size)
    // 使用 ceiling division 确保 Grid 能够完全覆盖 M x N 的矩阵
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 3. 计算 Scale 矩阵的列宽，用于在 Kernel 中进行二维到一维的索引转换
    int S_cols = (N + TILE_SIZE - 1) / TILE_SIZE;

    // 4. 启动 Kernel
    dequantize_kernel<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE, S_cols);
}