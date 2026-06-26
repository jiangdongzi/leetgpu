#include <cfloat>
#include <cuda_runtime.h>

__global__ void nei_kernal(const float* points, const int N, int* indices) {
    const int idx = blockIdx.x;
    const float a = points[3 * idx + 0];
    const float b = points[3 * idx + 1];
    const float c = points[3 * idx + 2];
    float min_dist = FLT_MAX, min_idx = -1;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    for (int i = tid; i < N; i += stride) {
        if (idx == i) continue;
        const float x = a - points[3 * i + 0];
        const float y = b - points[3 * i + 1];
        const float z = c - points[3 * i + 2];
        const float cur_dist = x * x + y * y + z * z;
        if (cur_dist < min_dist) {
            min_dist = cur_dist;
            min_idx = i;
        }
    }
    __shared__ float s_dist[256];
    __shared__ float s_idx[256];
    s_dist[tid] = min_dist;
    s_idx[tid] = min_idx;
    __syncthreads();
    for (int offset = 256 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            const float other_dist = s_dist[tid + offset];
            const float other_idx = s_idx[tid + offset];
            const float cur_dist = s_dist[tid];
            const float cur_idx = s_idx[tid];
            if (other_dist < cur_dist || (other_dist == cur_dist && other_idx < cur_idx)) {
                s_dist[tid] = other_dist;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        indices[idx] = s_idx[0];
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    nei_kernal<<<N, 256>>>(points, N, indices);
}