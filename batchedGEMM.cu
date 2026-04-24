#include <cuda_runtime.h>

// 定义分块大小，32x32 是一个能很好平衡寄存器和共享内存占用的经典配置
// 确保每个 Block 有 1024 个线程，达到较高的 Occupancy
#define TILE_SIZE 32

__global__ void batched_sgemm_tiled_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int BATCH, int M, int N, int K) 
{
    // 利用 Grid 的 Z 维度来处理 Batch，天然隔离不同矩阵的计算
    int b = blockIdx.z;

    // 计算当前 batch 的起始指针偏移
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C = C + b * M * N;

    // 当前线程负责计算 C 矩阵中的确切行列坐标
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 申请共享内存，用于缓存 A 和 B 的分块
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 计算 K 维度上需要滑动多少个 Tile
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // 1. 协同加载 A 的分块到共享内存 (包含边界检查)
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = batch_A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f; // 越界部分补零，防止非法访存
        }

        // 2. 协同加载 B 的分块到共享内存 (包含边界检查)
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = batch_B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 必须同步，确保整个 Block 的线程都把数据写完
        __syncthreads();

        // 3. 利用共享内存中的数据进行局部点积计算
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // 必须同步，确保进入下一个 Tile 之前，当前 Tile 的共享内存数据都被使用完毕
        __syncthreads();
    }

    // 4. 将最终累加结果写回全局内存
    if (row < M && col < N) {
        batch_C[row * N + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    // 定义 Block 尺寸
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    
    // 定义 Grid 尺寸，X 对应 N 的划分，Y 对应 M 的划分，Z 对应 Batch
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        BATCH
    );

    // 启动内核
    batched_sgemm_tiled_kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
    
    // 同步设备（在独立跑 Kernel 测试时是个好习惯）
    cudaDeviceSynchronize();
}