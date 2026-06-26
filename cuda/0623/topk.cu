#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

__device__ void swap(float& a, float& b) {
    float tmp = a;
    a = b;
    b = tmp;
}
__device__ void heapify(float* h, int idx, const int N) {
    while (true) {
        int smallest = idx;
        const int l = 2 * idx + 1, r = 2 * idx + 2;
        if (l < N && h[smallest] > h[l]) {
            smallest = l;
        }
        if (r < N && h[smallest] > h[r]) {
            smallest = r;
        }
        if (smallest == idx) break;
        swap(h[smallest], h[idx]);
        idx = smallest;
    }
}

__device__ void try_insert_h(float* h, const float val, const int N) {
    if (val < h[0]) return;
    h[0] = val;
    heapify(h, 0, N);
}

__device__ void heap_sort_desc(float* h, const int N) {
    for (int end = N - 1; end > 0; end--) {
        swap(h[0], h[end]);
        heapify(h, 0, end);
    }
}

__device__ void gen_block_topk(float* arr, const int k, float* candidate) {
    const int tid = threadIdx.x;
    int idx = 0;
    __shared__ float s_max[1024];
    __shared__ int s_idx[1024];
    for (int i = 0; i < k; i++) {
        s_max[tid] = arr[idx];
        s_idx[tid] = tid;
        __syncthreads();
        for (int offset = 1024 / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const float other_max_val = s_max[tid + offset];
                const float max_val = s_max[tid];
                if (max_val < other_max_val) {
                    s_max[tid] = other_max_val;
                    s_idx[tid] = s_idx[tid + offset];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            candidate[i] = s_max[0];
        }
        if (s_idx[0] == tid) {
            idx++;
        }
        __syncthreads();
    }
}

__global__ void scan_block(const float* input, const int N, const int k, float* cand) {
    float h[1024];
    for (int i = 0; i < k; i++) {
        h[i] = -FLT_MAX;
    }
    const int tid = threadIdx.x;
    const int gtid = tid + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        try_insert_h(h, input[i], k);
    }
    heap_sort_desc(h, k);
    gen_block_topk(h, k, cand + blockIdx.x * k);
}

__global__ void global_topk(const float* cand, const int N, const int k, float* output) {
    float h[1024];
    for (int i = 0; i < k; i++) {
        h[i] = -FLT_MAX;
    }
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    for (int i = tid; i < N; i += stride) {
        try_insert_h(h, cand[i], k);
    }
    heap_sort_desc(h, k);
    gen_block_topk(h, k, output);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    const int blocks = std::min(1024, (N + 1023) / 1024);
    float* cand;
    cudaMalloc(&cand, sizeof(float) * k * blocks);
    scan_block<<<blocks, 1024>>>(input, N, k, cand);
    global_topk<<<1, 1024>>>(cand, k * blocks, k, output);
    cudaFree(cand);
}