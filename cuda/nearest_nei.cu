#include <cfloat>
#include <cuda_runtime.h>

__global__ void nearest_neighbor_kernal(const float* points, int* indices, int N) {
    const int myIdx = blockIdx.x;
    const float i = points[3 * myIdx + 0];
    const float j = points[3 * myIdx + 1];
    const float k = points[3 * myIdx + 2];
    float dist = FLT_MAX;
    int nearest_idx = -1;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    for (int idx = tid; idx < N; idx += stride) {
        if (idx == myIdx) {
            continue;
        }
        const float x = points[3 * idx + 0] - i;
        const float y = points[3 * idx + 1] - j;
        const float z = points[3 * idx + 2] - k;
        const float tmp = x * x + y * y + z * z;
        if (tmp < dist) {
            dist = tmp;
            nearest_idx = idx;
        }
    }
    __shared__ float sDist[256];
    __shared__ int sIdx[256];
    sDist[tid] = dist;
    sIdx[tid] = nearest_idx;
    __syncthreads();
    for (int offset = 256 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            const float other_val = sDist[tid + offset];
            const int other_idx = sIdx[tid + offset];
            const float my_val = sDist[tid];
            const int my_idx = sIdx[tid];
            if (other_val < my_val || (other_val == my_val && other_idx < my_idx)) {
                sDist[tid] = other_val;
                sIdx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        indices[myIdx] = sIdx[0];
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    nearest_neighbor_kernal<<<N, 256>>>(points, indices, N);
}