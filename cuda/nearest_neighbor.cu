#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

// CUDA Kernel
__global__ void nearestNeighborKernel(const float* __restrict__ points, int* __restrict__ indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // 1. 将当前线程负责的目标点加载到寄存器中 (Register)
    float target_x, target_y, target_z;
    if (idx < N) {
        target_x = points[idx * 3];
        target_y = points[idx * 3 + 1];
        target_z = points[idx * 3 + 2];
    }

    float min_dist = FLT_MAX;
    int best_idx = -1;

    // 2. 声明共享内存 (Shared Memory)
    // 技巧：这里在共享内存中将全局内存的 AoS (Array of Structures) 转为 SoA (Structure of Arrays)
    // 这样做可以避免后续计算时的 Shared Memory Bank Conflicts
    __shared__ float shared_x[BLOCK_SIZE];
    __shared__ float shared_y[BLOCK_SIZE];
    __shared__ float shared_z[BLOCK_SIZE];

    // 计算需要分多少个 Tile
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 3. Tiling 循环，每次加载 BLOCK_SIZE 个点到共享内存
    for (int t = 0; t < num_tiles; ++t) {
        int load_idx = t * BLOCK_SIZE + tx;

        // 协作加载点数据到共享内存
        if (load_idx < N) {
            shared_x[tx] = points[load_idx * 3];
            shared_y[tx] = points[load_idx * 3 + 1];
            shared_z[tx] = points[load_idx * 3 + 2];
        }
        
        // 屏障同步：确保当前 Tile 的所有点都已加载完毕
        __syncthreads();

        // 4. 计算当前目标点与当前 Tile 中所有点的距离
        if (idx < N) {
            // 处理最后一个可能不满的 Tile
            int limit = min(BLOCK_SIZE, N - t * BLOCK_SIZE);
            
            #pragma unroll
            for (int j = 0; j < limit; ++j) {
                int global_j = t * BLOCK_SIZE + j;
                
                // 排除自己与自己的比较
                if (global_j != idx) {
                    float dx = target_x - shared_x[j];
                    float dy = target_y - shared_y[j];
                    float dz = target_z - shared_z[j];
                    float dist = dx * dx + dy * dy + dz * dz;

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = global_j;
                    }
                }
            }
        }
        
        // 屏障同步：确保所有线程都计算完毕，然后再进入下一个循环覆盖共享内存
        __syncthreads();
    }

    // 5. 将结果写回全局内存
    if (idx < N) {
        indices[idx] = best_idx;
    }
}

// 供外部调用的接口
extern "C" void solve(const float* points, int* indices, int N) {
    if (N <= 1) return; // 边界情况保护

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    // 启动 Kernel
    nearestNeighborKernel<<<blocks, threads>>>(points, indices, N);
    
    // 注意：LeetGPU 平台通常会在底层隐式调用 cudaDeviceSynchronize() 并测时，
    // 在这里调用 Kernel 即可。
}