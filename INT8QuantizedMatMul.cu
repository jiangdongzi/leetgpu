#include <cuda_runtime.h>
#include <math.h>

// 定义分块大小，32x32 是绝大多数架构下平衡寄存器和共享内存的黄金尺寸
#define TILE_SIZE 32

__global__ void quantizedMatMulKernel(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int8_t* __restrict__ C,
    int M, int N, int K,
    float scale_A, float scale_B, float scale_C,
    int zero_point_A, int zero_point_B, int zero_point_C)
{
    // 计算当前线程负责计算的 C 矩阵的全局行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 申请共享内存
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];

    int32_t sum = 0;
    
    // 计算需要遍历的 Tile 数量
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // 1. 将 A 矩阵的块加载到共享内存
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            // 核心细节：越界必须补 zero_point，保证 (A - zA) 为 0
            As[threadIdx.y][threadIdx.x] = zero_point_A; 
        }

        // 2. 将 B 矩阵的块加载到共享内存
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            // 同理，越界补 zero_point_B
            Bs[threadIdx.y][threadIdx.x] = zero_point_B;
        }

        // 等待 block 内所有线程将数据加载到 Shared Memory 完毕
        __syncthreads();

        // 3. 计算当前块的内积并累加到局部寄存器 sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += (int32_t)(As[threadIdx.y][k] - zero_point_A) * (int32_t)(Bs[k][threadIdx.x] - zero_point_B);
        }

        // 确保所有线程用完当前的 Shared Memory 后再进入下一个 Tile
        __syncthreads();
    }

    // 4. 计算最终结果并写回 Global Memory
    if (row < M && col < N) {
        // 浮点缩放系数提出来计算，避免在循环内重复开销
        float scale_multiplier = (scale_A * scale_B) / scale_C;
        
        // 转换为 float 计算缩放
        float scaled_val = (float)sum * scale_multiplier;
        
        // 四舍五入，并加上 C 的零点
        int32_t rounded_val = (int32_t)roundf(scaled_val) + zero_point_C;

        // Clamp 截断处理
        if (rounded_val > 127) rounded_val = 127;
        else if (rounded_val < -128) rounded_val = -128;

        C[row * N + col] = (int8_t)rounded_val;
    }
}

// 保持题目要求的 C 接口签名不变
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C)
{
    // 配置 Kernel 启动参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // 启动 Kernel
    quantizedMatMulKernel<<<gridDim, blockDim>>>(
        A, B, C, M, N, K,
        scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C);
        
    // 同步设备（如果测试平台有计时模块，这一步能保证计时准确）
    cudaDeviceSynchronize(); 
}