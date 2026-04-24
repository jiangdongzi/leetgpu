#include <cuda_runtime.h>
#include <math.h>

// 核心的 Kernel 函数
__global__ void decode_attention_int8_kernel(
    const float* Q,
    const int8_t* K_int8,
    const int8_t* V_int8,
    const float* k_scale,
    const float* v_scale,
    float* output,
    int num_heads,
    int seq_len,
    int head_dim)
{
    int h = blockIdx.x;       // 当前处理的 Attention Head
    int tid = threadIdx.x;    // 当前处理的特征维度

    // 将 Query 的当前维度值加载到寄存器中
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = Q[h * head_dim + tid];
    }

    // Online Softmax 的状态变量
    float m_prev = -1e20f; // 相当于负无穷大
    float l_prev = 0.0f;
    float o_acc = 0.0f;    // 最终输出向量的局部累加器

    // 缩放因子：1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    // 申请 32 个 float 的共享内存，用于 32 个 Warp 的归约求和
    __shared__ float shared_mem[32];

    int num_warps = blockDim.x / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 沿着序列维度迭代 (处理每一个历史 Token)
    for (int s = 0; s < seq_len; s++) {
        
        // 1. 读取并反量化 Key，计算 Q * K^T 的局部乘积
        float qk = 0.0f;
        if (tid < head_dim) {
            int8_t k_val = K_int8[h * seq_len * head_dim + s * head_dim + tid];
            float k_s = k_scale[h * seq_len + s];
            qk = q_val * ((float)k_val * k_s);
        }

        // --- 开始 Block 级别的规约求和 (Block Reduce Sum) ---
        float val = qk;
        // Warp 内规约
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // 每个 Warp 的第 0 个线程将结果写入 Shared Memory
        if (lane_id == 0) {
            shared_mem[warp_id] = val;
        }
        __syncthreads(); 

        // Warp 0 对所有 Warp 的结果进行最终规约
        if (warp_id == 0) {
            val = (lane_id < num_warps) ? shared_mem[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                shared_mem[0] = val;
            }
        }
        __syncthreads(); 
        // --- 规约结束，现在 shared_mem[0] 里存着当前 token 的未缩放 Score ---

        float score = shared_mem[0] * scale;

        // 2. Online Softmax 核心计算
        float m_curr = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_curr);
        float exp_curr = expf(score - m_curr);

        l_prev = l_prev * exp_prev + exp_curr;

        // 3. 读取并反量化 Value，累加到输出寄存器中
        if (tid < head_dim) {
            int8_t v_val = V_int8[h * seq_len * head_dim + s * head_dim + tid];
            float v_s = v_scale[h * seq_len + s];
            float v_f = (float)v_val * v_s;
            o_acc = o_acc * exp_prev + exp_curr * v_f;
        }

        // 更新状态，进入下一个 token
        m_prev = m_curr;
    }

    // 4. 将 Softmax 的分母除掉，写入最终结果到 Global Memory
    if (tid < head_dim) {
        output[h * head_dim + tid] = o_acc / l_prev;
    }
}

extern "C" void solve(
    const float* Q, 
    const int8_t* K_int8, 
    const int8_t* V_int8, 
    const float* k_scale, 
    const float* v_scale, 
    float* output, 
    int num_heads, 
    int seq_len, 
    int head_dim) 
{
    // 约束说明 head_dim <= 256，所以固定 256 的 block 大小刚好能覆盖一个特征维度，
    // 并且能整除 32 构成完整的 warps，简化规约逻辑。
    int block_size = 256; 
    
    dim3 grid(num_heads);
    dim3 block(block_size);

    decode_attention_int8_kernel<<<grid, block>>>(
        Q, K_int8, V_int8, k_scale, v_scale, output, 
        num_heads, seq_len, head_dim
    );
}