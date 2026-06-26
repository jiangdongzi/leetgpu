#include <algorithm>
#include <cuda_runtime.h>

__device__ void warp_reduce(float& local_sum) {
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
}
__global__ void mse_kernal(const float* predictions, const float* targets, float* mse, int N) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    float tmp_sum = 0.f;
    for (int i = gtid; i < N; i += stride) {
        const float a = predictions[i] - targets[i];
        tmp_sum += a * a;
    }
    warp_reduce(tmp_sum);
    __shared__ float sm[32];
    const int lane_id = tid % 32, warp_id = tid / 32;
    if (lane_id == 0) {
        sm[warp_id] = tmp_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        tmp_sum = sm[tid];
        warp_reduce(tmp_sum);
        if (tid == 0) {
            atomicAdd(mse, tmp_sum / N);
        }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    const int blocks = std::min(1024, (N + 1023) / 1024);
    mse_kernal<<<blocks, 1024>>>(predictions, targets, mse, N);
}