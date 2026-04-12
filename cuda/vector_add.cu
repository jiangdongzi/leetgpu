#include <cstddef>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void solve(const float* a, const float* b, float* c, size_t n) {
    const int threads_per_block = 256;
    const int blocks_per_grid = static_cast<int>((n + threads_per_block - 1) / threads_per_block);
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
    cudaDeviceSynchronize();
}
