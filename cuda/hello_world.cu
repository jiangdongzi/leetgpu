#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(error));                           \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

__global__ void hello_from_gpu() {
    std::printf("Hello World from CUDA kernel! block=%d thread=%d\n",
                blockIdx.x, threadIdx.x);
}

int main() {
    std::printf("Hello World from CPU!\n");

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::printf("CUDA runtime is present, but no usable GPU was found: %s\n",
                    cudaGetErrorString(error));
        std::printf("Hello World CUDA program finished successfully.\n");
        return EXIT_SUCCESS;
    }

    if (device_count == 0) {
        std::printf("No CUDA-capable GPU was found.\n");
        std::printf("Hello World CUDA program finished successfully.\n");
        return EXIT_SUCCESS;
    }

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::printf("Using CUDA device 0: %s\n", props.name);

    hello_from_gpu<<<1, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Hello World CUDA program finished successfully.\n");
    return EXIT_SUCCESS;
}
