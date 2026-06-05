#include <cuda_runtime.h>
#include <float.h>

// 每个 block 算一个 query 点 i 的最近邻
// blockDim.x 个线程协作扫描所有 N 个候选点，做 block 内归约
__global__ void nearest(const float* points, int* indices, int N) {
    int i = blockIdx.x;          // 当前 query 点
    int tid = threadIdx.x;
    int nt = blockDim.x;

    // 把 query 点坐标读进寄存器（一次，复用 N 次）
    float xi = points[3 * i + 0];
    float yi = points[3 * i + 1];
    float zi = points[3 * i + 2];

    float bestDist = FLT_MAX;
    int   bestIdx  = -1;

    // grid-stride 扫描所有候选点 j
    for (int j = tid; j < N; j += nt) {
        if (j == i) continue;    // 排除自己
        float dx = points[3 * j + 0] - xi;
        float dy = points[3 * j + 1] - yi;
        float dz = points[3 * j + 2] - zi;
        float d = dx * dx + dy * dy + dz * dz;   // 平方距离，无需开方
        if (d < bestDist) { bestDist = d; bestIdx = j; }
    }

    // block 内归约：找全局最小，平局取更小的 index
    __shared__ float sDist[256];
    __shared__ int   sIdx[256];
    sDist[tid] = bestDist;
    sIdx[tid]  = bestIdx;
    __syncthreads();

    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float od = sDist[tid + s];
            int   oi = sIdx[tid + s];
            // 距离更小，或距离相等但下标更小 → 更新
            if (od < sDist[tid] || (od == sDist[tid] && oi < sIdx[tid])) {
                sDist[tid] = od;
                sIdx[tid]  = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) indices[i] = sIdx[0];
}

extern "C" void solve(const float* points, int* indices, int N) {
    int threads = 256;
    int blocks = N;              // 一个 block 负责一个点
    nearest<<<blocks, threads>>>(points, indices, N);
}