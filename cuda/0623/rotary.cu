#include <cuda_runtime.h>

__global__ void rotay_kernal(float* Q, float* cos, float* sin, float* output, int M, int D) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid >= M * D) return;
    const int col = gtid % D;
    const int half_d = D / 2;
    float half;
    if (col < half_d) {
        half = -Q[gtid + half_d];
    } else {
        half = Q[gtid - half_d];
    }
    output[gtid] = Q[gtid] * cos[gtid] + half * sin[gtid];
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    const int total = M * D;
    const int blocks = (total + 1023) / 1024;
    rotay_kernal<<<blocks, 1024>>>(Q, cos, sin, output, M, D);
}