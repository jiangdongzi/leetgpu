#include <cuda_runtime.h>

// 定义 Tile 的大小，需要与 host 代码中的 threadsPerBlock 匹配
#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. 计算当前线程负责的全局行号和列号
    // blockIdx.y 对应行 (M)，blockIdx.x 对应列 (K)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 2. 声明共享内存（__shared__ 变量存储在每个 Block 的片上内存中，速度非常快）
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // 用于累加当前线程负责的 C[row][col] 的点积结果
    float value = 0.0f;

    // 3. 遍历所有需要的 Tile 分块来计算完整的点积
    // 矩阵 A 宽度为 N，矩阵 B 高度为 N，所以共需要向上取整 (N / TILE_SIZE) 个阶段
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int ph = 0; ph < numTiles; ++ph) {
        
        // 4. 协作加载矩阵 A 的当前 Tile 到共享内存
        // 边界检查：如果行索引超出 M 或列索引（在此阶段中）超出 N，则补 0
        if (row < M && (ph * TILE_SIZE + threadIdx.x) < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + (ph * TILE_SIZE + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 5. 协作加载矩阵 B 的当前 Tile 到共享内存
        // 边界检查：如果行索引（在此阶段中）超出 N 或列索引超出 K，则补 0
        if ((ph * TILE_SIZE + threadIdx.y) < N && col < K) {
            s_B[threadIdx.y][threadIdx.x] = B[(ph * TILE_SIZE + threadIdx.y) * K + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 6. 同步线程，确保当前 Block 内所有线程都已经完成了共享内存的加载
        __syncthreads();

        // 7. 对当前加载到共享内存中的 Tile 进行子矩阵乘法，并累加到 value 中
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        // 8. 再次同步线程，确保在进入下一个阶段覆盖共享内存之前，所有线程都已经完成了计算
        __syncthreads();
    }

    // 9. 将最终结果写回全局内存的 C 矩阵中，并进行边界检查
    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // 启动配置，16x16 线程块
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // 计算 Grid 大小，注意 x 对应 K (列)，y 对应 M (行)
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}