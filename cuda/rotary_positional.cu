#include <cuda_runtime.h>

__global__ void rp_kernal(float* Q, float* cos, float* sin, float* output, int M, int D) {
    const int tid = threadIdx.x;
    const int gtid = tid + blockDim.x * blockIdx.x;
    if (gtid >= M * D) return;
    const int col = gtid % D;
    const int half = D / 2;
    float half_val;
    if (col < half) {
        half_val = -Q[gtid + half];
    } else {
        half_val = Q[gtid - half];
    }
    output[gtid] = Q[gtid] * cos[gtid] + half_val * sin[gtid];
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    const int total = M * D;
    rp_kernal<<<(total + 255) / 256, 256>>>(Q, cos, sin, output, M, D);
}
