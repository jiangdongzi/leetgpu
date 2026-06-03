#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void hist_kernal(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int sm[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) sm[i] = 0;
    __syncthreads();
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gtid; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&sm[input[i]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (sm[i] > 0) {
            atomicAdd(&histogram[i], sm[i]);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int block_size = 256;
    size_t sm_size = num_bins * sizeof(int);
    hist_kernal<<<(N + block_size - 1) / block_size, block_size, sm_size>>>(input, histogram, N, num_bins);
}

int main() {
    std::vector<int> vec(1025, 1);
    int* input, *output;
    cudaMalloc(&input, 4 * 1025);
    cudaMemcpy(input, vec.data(), 4 * 1025, cudaMemcpyHostToDevice);
    std::vector<int> out{0, 0, 0};
    cudaMalloc(&output, 12);
    cudaMemset(output, 0, 12);
    solve(input, output, 1025, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), output, 12, cudaMemcpyDeviceToHost);
}