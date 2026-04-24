#include <cuda_runtime.h>

// CUDA Kernel：每个线程计算输出矩阵中的一个元素
__global__ void rope_kernel(const float* Q, const float* cos, const float* sin, float* output, int M, int D) {
    // 获取当前线程的全局唯一索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * D;

    // 确保线程没有越界
    if (idx < total_elements) {
        // 计算当前元素属于哪一列 (0 到 D-1)
        int d = idx % D;
        int half_D = D / 2;

        // 获取当前位置的基础值
        float q_val = Q[idx];
        float cos_val = cos[idx];
        float sin_val = sin[idx];

        // 计算 rotate_half 对应位置的值
        float q_rot;
        if (d < half_D) {
            // 如果在左半边，取右半边的对应值并取负
            q_rot = -Q[idx + half_D];
        } else {
            // 如果在右半边，取左半边的对应值（不取负）
            q_rot = Q[idx - half_D];
        }

        // 核心公式：x * cos + rotate_half(x) * sin
        output[idx] = q_val * cos_val + q_rot * sin_val;
    }
}

// 供外部调用的宿主函数
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int total_elements = M * D;
    
    // 定义线程块大小，256 往往是一个性能不错的默认值
    int threads_per_block = 256;
    
    // 计算需要的网格大小 (向上取整)
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // 启动内核
    rope_kernel<<<blocks_per_grid, threads_per_block>>>(Q, cos, sin, output, M, D);
}