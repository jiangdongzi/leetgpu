#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // 1个线程处理1个像素 (4 bytes)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        // 强转为32位整型指针，实现 coalesced 宽访存
        unsigned int* img32 = reinterpret_cast<unsigned int*>(image);
        
        // 0x00FFFFFF 异或操作：
        // 小端序下，低 24 位对应 R, G, B（与 1 异或即取反，等价于 255 - val）
        // 高 8 位对应 Alpha（与 0 异或保持不变）
        img32[idx] ^= 0x00FFFFFF;
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    // 计算需要的 Grid 数量，每个线程处理一个像素
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}