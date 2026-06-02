#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

constexpr int block_size = 256;
__global__ void scan_block(const float* input, float* output, const int N, float* block_sums) {
    const int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    const int t = threadIdx.x;
    static __shared__ float sh[block_size];
    bool valid = gtid < N;
    sh[t] = valid ? input[gtid] : 0.f;
    for (int step = 1; step < block_size; step *= 2) {
        float tmp = 0.f;
        if (t > step) {
            tmp = sh[t - step];
        }
        __syncthreads();
        sh[t] += tmp;
        __syncthreads();
    }
    if (valid) {
        output[gtid] = sh[t];
        const int my_block_size = min(block_size, N - gtid);
        if (t == my_block_size - 1) {
            block_sums[blockDim.x] = sh[t];
        }
    }
}

__global__ void merge(float* output, const float* block_sums) {
    const int b = blockIdx.x;
    const int gtid = b * 256 + threadIdx.x;
    output[gtid + 256] += block_sums[b];
}

void scan(const float* input, float* output, const int N) {
    float* block_sums;
    cudaMalloc(&block_sums, (N + 255) / 256 * sizeof(float));
    scan_block<<<(N + 255) / 256, block_size>>>(input, output, N, block_sums);
    if ((N + 255) / 256 == 1) {
        cudaFree(block_sums);
        return;
    }
    scan(block_sums, block_sums, (N + 255) / 256);
    merge<<<(N + 255) / 256, block_size>>>(output, block_sums);
}

extern "C" void solve(const float* input, float* output, const int N) {
    scan(input, output, N);
}