#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

extern "C" void solve(const float *a, const float *b, float *c, size_t n) {
  const int threads_per_block = 256;
  const int blocks_per_grid =
      static_cast<int>((n + threads_per_block - 1) / threads_per_block);
  vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
  cudaDeviceSynchronize();
}

int main() {
  // Example usage
  const size_t n = 10000000;
  float *a, *b, *c, *compare_c;

  // Allocate memory on the host
  a = (float *)malloc(n * sizeof(float));
  b = (float *)malloc(n * sizeof(float));
  c = (float *)malloc(n * sizeof(float));
  compare_c = (float *)malloc(n * sizeof(float));

  // Initialize input vectors
  for (size_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
    compare_c[i] = a[i] + b[i];
  }

  // Allocate memory on the device
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

  // Call the solve function
  solve(d_a, d_b, d_c, n);

  // Copy result back to host
  cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  // Verify results
  for (size_t i = 0; i < n; ++i) {
    if (c[i] != compare_c[i]) {
      printf("Error at index %zu: expected %f, got %f\n", i, compare_c[i],
             c[i]);
      return -1;
    }
  }
  printf("All results are correct!\n");
  // Clean up
  free(a);
  free(b);
  free(c);
  free(compare_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
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
