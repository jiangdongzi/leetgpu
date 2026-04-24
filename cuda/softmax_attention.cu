#include <cuda_runtime.h>
#include <math.h>

// 定义负无穷大，用于 Online Softmax 初始化
#define NEG_INF __int_as_float(0xff800000)

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V, float* output, 
                                         int M, int N, int d, float scale) {
    // 线程块配置：blockDim.x = 32 (Warp 大小), blockDim.y = 16 (每 Block 包含 16 个 Warp)
    // 每个 Warp 独立处理一个 Query
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int tid = warp_id * 32 + lane_id;
    
    // 全局 Query 索引
    int i = blockIdx.x * 16 + warp_id;
    bool valid_query = (i < M);
    
    // 使用 Register 缓存当前 Query 向量和 Output 累加器 (d <= 128，每个线程最多处理 4 个元素)
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (valid_query) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int col = lane_id + k * 32;
            if (col < d) {
                q[k] = Q[i * d + col];
            }
        }
    }
    
    // Online Softmax 的状态变量
    float m_prev = NEG_INF; 
    float l_prev = 0.0f;
    
    // 申请 Shared Memory 缓存 K 和 V 的 Tile
    // Tile 大小设为 32，最大 d 为 128，占用 32 * 128 * 4 bytes = 16KB per SMEM array
    __shared__ float K_shared[32 * 128];
    __shared__ float V_shared[32 * 128];
    
    // 沿 N 维度进行 Tiling 遍历
    for (int j_tile = 0; j_tile < N; j_tile += 32) {
        int num_keys = min(32, N - j_tile);
        int total_elements = num_keys * d;
        
        // 1. 协同加载 K 和 V 到 Shared Memory (完全内存合并)
        const float* K_ptr = K + j_tile * d;
        const float* V_ptr = V + j_tile * d;
        
        for (int idx = tid; idx < total_elements; idx += 512) {
            K_shared[idx] = K_ptr[idx];
            V_shared[idx] = V_ptr[idx];
        }
        
        __syncthreads(); // 等待当前 Tile 加载完成
        
        // 2. 在当前 Tile 上计算 Attention
        if (valid_query) {
            for (int jj = 0; jj < num_keys; ++jj) {
                // 计算当前 Query 和 K_shared[jj] 的点积
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int col = lane_id + k * 32;
                    if (col < d) {
                        sum += q[k] * K_shared[jj * d + col];
                    }
                }
                
                // Warp 内部规约求和 (无分支，无需 Shared Memory 同步)
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }
                float S = __shfl_sync(0xffffffff, sum, 0); // 广播点积结果给 Warp 内所有线程
                S *= scale; // 除以 sqrt(d)
                
                // 更新 Online Softmax
                float m_curr = fmaxf(m_prev, S);
                float w1 = expf(m_prev - m_curr);
                float w2 = expf(S - m_curr);
                
                l_prev = l_prev * w1 + w2;
                
                // 累加 V 到寄存器
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int col = lane_id + k * 32;
                    if (col < d) {
                        o_acc[k] = o_acc[k] * w1 + w2 * V_shared[jj * d + col];
                    }
                }
                
                m_prev = m_curr;
            }
        }
        
        __syncthreads(); // 确保所有 Warp 在加载下一个 Tile 前计算完毕
    }
    
    // 3. 写入最终结果到 Global Memory
    if (valid_query) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int col = lane_id + k * 32;
            if (col < d) {
                output[i * d + col] = o_acc[k] / l_prev;
            }
        }
    }
}

// 主函数接口 (题目规定的 Signature)
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // Block 配置为 (32, 16)：包含 16 个 Warp，共计 512 线程
    int warps_per_block = 16;
    dim3 block(32, warps_per_block);
    // Grid 维度根据 Query 数量 M 和每个 Block 处理的 Query 数进行计算
    dim3 grid((M + warps_per_block - 1) / warps_per_block);
    
    // 提前在 CPU 计算 scale 因子传入 Kernel，避免 GPU 内重复计算
    float scale = 1.0f / sqrtf((float)d);
    
    softmax_attention_kernel<<<grid, block>>>(Q, K, V, output, M, N, d, scale);
    
    // 等待 Kernel 执行完毕
    cudaDeviceSynchronize();
}