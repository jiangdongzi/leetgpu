#include <cstddef>
#include <cuda_runtime.h>
#include <cstdio>

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

//验证solve函数是否正确
int main() {
    const size_t n = 1000;
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
    solve(a, b, c, n);
    for (size_t i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            printf("Error at index %zu: expected %f, got %f\n", i, a[i] + b[i], c[i]);
            return 1;
        }
    }
    return 0;
}
