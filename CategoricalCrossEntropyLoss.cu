#include <cuda_runtime.h>
#include <math.h>

// Warp 级别的最大值规约
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp 级别的求和规约
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;

    // 定位到当前样本的 logits 起始地址
    const float* row_logits = logits + row * C;
    int label = true_labels[row];

    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // ==========================================
    // 第一步：寻找当前行的最大值 (Log-Sum-Exp 的 m)
    // ==========================================
    float thread_max = -1e20f; 
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[i]);
    }

    // Warp 级规约
    thread_max = warpReduceMax(thread_max);
    if (lane == 0) shared_max[wid] = thread_max;
    __syncthreads();

    // Block 级规约
    float max_val = (threadIdx.x < (blockDim.x / warpSize)) ? shared_max[lane] : -1e20f;
    if (wid == 0) max_val = warpReduceMax(max_val);

    // 广播最大值给所有线程
    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = max_val;
    __syncthreads();
    max_val = block_max;

    // ==========================================
    // 第二步：计算 exp(x - max) 的和
    // ==========================================
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        thread_sum += expf(row_logits[i] - max_val);
    }

    thread_sum = warpReduceSum(thread_sum);
    if (lane == 0) shared_sum[wid] = thread_sum;
    __syncthreads();

    float sum_val = (threadIdx.x < (blockDim.x / warpSize)) ? shared_sum[lane] : 0.0f;
    if (wid == 0) sum_val = warpReduceSum(sum_val);

    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = sum_val;
    __syncthreads();
    sum_val = block_sum;

    // ==========================================
    // 第三步：计算最终 Loss 并累加
    // ==========================================
    if (threadIdx.x == 0) {
        // LSE = max + log(sum)
        float log_sum_exp = max_val + logf(sum_val);
        // Loss_j = LSE - z_{j, y_j}
        float row_loss = log_sum_exp - row_logits[label];
        
        // 累加平均值到全局 loss
        atomicAdd(loss, row_loss / N);
    }
}

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    // 初始化设备端 loss 变量为 0
    cudaMemset(loss, 0, sizeof(float));

    // 一个 Block 对应一行样本，256 个线程足以覆盖 C <= 1000 的高效计算
    int threads = 256;
    int blocks = N;

    cross_entropy_kernel<<<blocks, threads>>>(logits, true_labels, loss, N, C);
}