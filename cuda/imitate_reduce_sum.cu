#include <cuda_runtime.h>
#include <cstdio>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    val = warpReduceSum(val);
    static __shared__ float smem[32];
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = smem[lane_id];
        return warpReduceSum(val);
    } else {
        return 0;
    }
}

__global__ void reduce_kernal(const float* input, float* output, const int N) {
    float val = 0.f;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int gtid = blockIdx.x * 1024 + warp_id * 32 + lane_id;
    for (int i = gtid; i < N; i += 1024 * gridDim.x) {
        val += input[i];
    }
    val = blockReduceSum(val);
    if (warp_id == 0 && lane_id == 0) {
        atomicAdd(output, val);
    }
}

extern "C" void solve(const float* input, float* output, const int N) {
    dim3 threads(32, 32);
    int blocks = 1024;
    if (N < 1024 * 1024) {
        blocks = (N + 1023) / 1024;
    }
    reduce_kernal<<<blocks, threads>>>(input, output, N);
}

// 验证 solve 函数是否正确
int main() {
    const int N = 1 << 25; // 1048576
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