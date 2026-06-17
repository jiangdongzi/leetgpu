#include <cuda_runtime.h>
#include <float.h>

__global__ void nearest(const float* points, int* indices, int N) {
    const int i = blockIdx.x;
    const int nt = blockDim.x;
    const float x = points[3 * i + 0];
    const float y = points[3 * i + 1];
    const float z = points[3 * i + 2];
    const int tid = threadIdx.x;
    float dist = FLT_MAX;
    int idx = -1;
    for (int j = tid; j < N; j += nt) {
        if (i == j) continue;
        const float xj = points[3 * j + 0];
        const float yj = points[3 * j + 1];
        const float zj = points[3 * j + 2];
        const float xd = x - xj;
        const float yd = x - yj;
        const float zd = x - zj;
        const float d = xd * xd + yd * yd + zd * zd;
        if (d < dist) {
            dist = d;
            idx = j;
        }
    }

    //warp归约
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other_d = __shfl_down_sync(0xffffffff, dist, offset);
        if (other_d < dist) {
            dist = other_d;
            idx = __shfl_down_sync(0xffffffff, idx, offset);
        }
    }
    __shared__ float sm[8];
    __shared__ int sIdx[8];
    if (tid % 32 == 0) {
        sm[tid / 32] = dist;
        sIdx[tid / 32] = idx;
    }
    __syncthreads();
    if (tid < 8) {
        dist = sm[tid];
        idx = sIdx[tid];
    }
    __syncthreads();
    //warp归约
    for (int offset = 4; offset > 0; offset /= 2) {
        const float other_d = __shfl_down_sync(0xffffffff, dist, offset);
        if (other_d < dist) {
            dist = other_d;
            idx = __shfl_down_sync(0xffffffff, idx, offset);
        }
    }
    if (tid == 0) {
        indices[i] = idx;
    }
}

extern "C" void solve(const float* points, int* indices, int N) {
    int threads = 256;
    int blocks = N;              // 一个 block 负责一个点
    nearest<<<blocks, threads>>>(points, indices, N);
}