#include <cuda_runtime.h>
#include <cstdio>

// Warp 级别的规约求和
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block 级别的规约求和
__inline__ __device__ float blockReduceSum(float val) {
    // 只需要 32 个 float 的 shared memory 来存储每个 Warp 的结果
    static __shared__ float shared[32]; 
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 1. 每个 Warp 内部先进行规约
    val = warpReduceSum(val);

    // 2. 将每个 Warp 的结果写入 Shared Memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); // 确保所有 Warp 都写完了

    // 3. 由第一个 Warp 读取 Shared Memory 并进行最终规约
    // 只有 blockDim.x / warpSize 个有效数据
    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Kernel 函数
__global__ void reduce_kernel(const float* input, float* output, int N) {
    float sum = 0.0f;
    
    // Grid-Stride Loop: 每个线程处理多个元素，最大化吞吐量并自适应 N 的大小
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Block 内规约
    sum = blockReduceSum(sum);

    // 4. Block 的 0 号线程将结果原子累加到全局 output
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// 题目要求的接口
extern "C" void solve(const float* input, float* output, int N) {
    // 确保 output 初始化为 0，因为我们要使用 atomicAdd
    cudaMemset(output, 0, sizeof(float));

    int threads = 256;
    // 限制 block 数量，确保每个线程能处理多个元素，隐藏访存延迟
    // 对于 T4/A100 等 GPU，1024 到 4096 都是不错的 Grid 尺寸
    int blocks = 1024; 
    
    if (N < threads * blocks) {
        blocks = (N + threads - 1) / threads;
    }

    reduce_kernel<<<blocks, threads>>>(input, output, N);
}

// 验证 solve 函数是否正确
int main() {
    const int N = 1 << 20; // 1048576
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f; // 这样总和应该是 N
    }
    float* d_input;
    float* d_output;
    float h_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_input, d_output, N);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum: %f (Expected: %f)\n", h_output, static_cast<float>(N));
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    return 0;
}