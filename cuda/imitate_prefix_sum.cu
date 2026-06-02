#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

constexpr int block_size = 256;

__global__ void scan_block(const float* input, float* output, const int N, float* block_sums) {
    const int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    const int t = threadIdx.x;
    __shared__ float sh[block_size];           // static 可去掉
    bool valid = gtid < N;
    sh[t] = valid ? input[gtid] : 0.f;
    __syncthreads();                            // 写完共享内存先同步一次

    for (int step = 1; step < block_size; step *= 2) {
        float tmp = 0.f;
        if (t >= step) {                        // bug1: 原来是 t > step
            tmp = sh[t - step];
        }
        __syncthreads();
        sh[t] += tmp;
        __syncthreads();
    }

    if (valid) {
        output[gtid] = sh[t];
        const int my_block_size = min(block_size, N - blockIdx.x * block_size); // 用块起点算
        if (t == my_block_size - 1) {
            block_sums[blockIdx.x] = sh[t];     // bug2: 原来是 blockDim.x
        }
    }
}

__global__ void merge(float* output, const float* block_sums, const int N) {
    // 给第 (blockIdx.x + 1) 个块加上前缀块和 block_sums[blockIdx.x]
    const int b = blockIdx.x;
    const int gtid = (b + 1) * block_size + threadIdx.x;
    if (gtid < N) {                             // bug3: 加边界判断
        output[gtid] += block_sums[b];
    }
}

void scan(const float* input, float* output, const int N) {
    const int numBlocks = (N + block_size - 1) / block_size;
    float* block_sums;
    cudaMalloc(&block_sums, numBlocks * sizeof(float));
    scan_block<<<numBlocks, block_size>>>(input, output, N, block_sums);

    if (numBlocks == 1) {
        cudaFree(block_sums);
        return;
    }
    scan(block_sums, block_sums, numBlocks);
    merge<<<numBlocks - 1, block_size>>>(output, block_sums, N);  // bug4: grid 为 numBlocks-1
    cudaFree(block_sums);                       // 别忘了释放
}

extern "C" void solve(const float* input, float* output, const int N) {
    scan(input, output, N);
}

int main() {
    float *input, *output;
    cudaMalloc(&input, 4 * sizeof(float));
    std::vector<float> host_data{1, 2, 3, 4};
    cudaMemcpy(input, host_data.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&output, 4 * sizeof(float));
    solve(input, output, 4);
    std::vector<float> h_out(4);
    cudaMemcpy(h_out.data(), output, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    for (const float e : h_out) printf("%f\n", e);  // 期望 1 3 6 10
}