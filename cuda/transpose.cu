#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // 申请 Shared Memory，列维度 +1 (TILE_DIM + 1) 用于避免 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 1. 计算当前线程在输入矩阵中的全局坐标
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 将数据从 Global Memory 读入 Shared Memory (连续读取，合并访存)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // 同步，确保整个 Block 的数据都已加载到 Shared Memory
    __syncthreads();

    // 2. 计算当前线程在输出矩阵中的全局坐标
    // 注意：转置后，输出矩阵的 Block 坐标 (blockIdx.x, blockIdx.y) 发生了对调
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 将数据从 Shared Memory 写入 Global Memory
    // 从 tile 中按列读出 tile[tx][ty]，由于加了 Padding，不会发生 Bank Conflict
    // 写入 global_mem 时是连续写入，保证了写操作的合并访存
    if (out_x < rows && out_y < cols) {
        // 注意 output 的一维索引计算：out_y * (输出矩阵的列数 rows) + out_x
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // 设定 Block 大小
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    
    // 设定 Grid 大小，向上取整以覆盖边缘
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    // 启动 Kernel
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}