#include <cuda_runtime.h>

// 块大小设为 32，每个 Block 有 32x32=1024 个线程，达到硬件限制的最大效率
#define TILE_SIZE 32

// ---------------------------------------------------------
// 1. 核心 Kernel：利用共享内存的分块矩阵乘法
// ---------------------------------------------------------
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // 计算当前线程负责的全局行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 声明共享内存用于缓存当前 Tile
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 遍历所有需要的 Tile
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // 协同加载矩阵 A 的 Tile 到共享内存 (带边界越界检查)
        if (row < N && m * TILE_SIZE + threadIdx.x < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 协同加载矩阵 B 的 Tile 到共享内存
        if (m * TILE_SIZE + threadIdx.y < N && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // 等待当前 Tile 的所有数据加载完毕

        // 计算当前 Tile 的点积 (使用 #pragma unroll 提示编译器展开循环以提升性能)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads(); // 等待所有线程计算完毕，准备加载下一个 Tile
    }

    // 将最终计算结果写回 Global Memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------
// 2. 辅助 Kernel：初始化单位矩阵
// ---------------------------------------------------------
__global__ void init_identity_kernel(float* mat, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        mat[row * N + col] = (row == col) ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------
// 3. 主调函数：矩阵快速幂逻辑
// ---------------------------------------------------------
extern "C" void solve(const float* input, float* output, int N, int P) {
    // 计算显存大小
    size_t bytes = N * N * sizeof(float);
    
    // 我们需要 3 个缓冲区来完成原位的快速幂计算，避免频繁 cudaMalloc
    float *d_res, *d_base, *d_temp;
    cudaMalloc(&d_res, bytes);
    cudaMalloc(&d_base, bytes);
    cudaMalloc(&d_temp, bytes);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // 初始化：res 为单位矩阵，base 为输入矩阵 A
    init_identity_kernel<<<grid, block>>>(d_res, N);
    cudaMemcpy(d_base, input, bytes, cudaMemcpyDeviceToDevice);

    int current_P = P;

    // 矩阵快速幂核心循环
    while (current_P > 0) {
        if (current_P % 2 == 1) {
            // res = res * base
            matmul_kernel<<<grid, block>>>(d_res, d_base, d_temp, N);
            
            // 【面试加分项】通过交换指针代替耗时的 cudaMemcpy(DeviceToDevice)
            float* tmp = d_res; d_res = d_temp; d_temp = tmp;
        }
        current_P /= 2;
        
        if (current_P > 0) {
            // base = base * base
            // 注意：A 和 B 传入相同的指针 d_base 是安全的，因为 kernel 内部只读不写
            matmul_kernel<<<grid, block>>>(d_base, d_base, d_temp, N);
            
            // 同样交换指针
            float* tmp = d_base; d_base = d_temp; d_temp = tmp;
        }
    }

    // 将最终计算出的结果拷贝回输出参数
    cudaMemcpy(output, d_res, bytes, cudaMemcpyDeviceToDevice);

    // 释放临时显存
    cudaFree(d_res);
    cudaFree(d_base);
    cudaFree(d_temp);
}