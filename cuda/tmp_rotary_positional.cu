#include <cuda_runtime.h>

__global__ void rotary_kernal(float* Q, float* cos, float* sin, float* output, int M, int D) {
    const int half = D / 2;
    const int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    const int total = M * D;
    if (gtid >= total) return;
    const int col = gtid % D;
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
    rotary_kernal<<<(total + 127) / 128, 128>>>(Q, cos, sin, output, M, D);
}