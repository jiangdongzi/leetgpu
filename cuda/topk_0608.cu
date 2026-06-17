#include <cuda_runtime.h>
#include <float.h>

__device__ void heapify(float* heap, int i, const int n) {
    while (true) {
        int largest = i;
        const int l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && heap[largest] < l) largest = l;
        if (r < n && heap[largest] < r) largest  = r;
        if (i == largest) break;
        float tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp;
        i = largest;
    }
}

//heap是小顶堆
__device__ void heap_sort_desc(float* heap, const int n) {
    for (int end = n - 1; end > 0; end--) {
        float tmp = heap[end]; heap[end] = heap[0]; heap[0] = tmp;
        heapify(heap, 0, end);
    }
}

__device__ void block_merge_write(const float* arr, const int k, float* out) {
    __shared__ float sval[256];
    __shared__ float sidx[256];
    int taken = 0;
    const int tid = threadIdx.x;
    for (int i = 0; i < k; i++) {
        float head = arr[taken];
        sval[tid] = head;
        sidx[tid] = tid;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s /= 2) {
            if (tid < s && sval[tid] < sval[tid + s]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
                __syncthreads();
            }
        }
        if (tid == 0) {
            out[i] = sval[0];
        }
        if (tid == sidx[0]) {
            taken++;
        }
        __syncthreads();
    }
}

__device__ void heap_try_insert(float* h, const int k, const int val) {
    if (val < h[0]) return;
    h[0] = val;
    heapify(h, 0, k);
}

__global__ void topk_partial(const float* input, const int N, const int k, float* cand) {
    float h[1024];
    for (int i = 0; i < 1024; i++) {
        h[i] = -FLT_MAX;
    }
    const int stride = blockDim.x * gridDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gtid; i < N; i += stride) {
        heap_try_insert(h, k, input[i]);
    }
    heap_sort_desc(h, k);
    block_merge_write(h, k, cand + blockIdx.x * k);
}

__global__ void topk_final(const float* in, const int N, const int k, float* output) {
    float h[1024];
    for (int i = 0; i < 1024; i++) {
        h[i] = -FLT_MAX;
    }
    const int stride = blockDim.x;
    const int gtid = threadIdx.x;
    for (int i = gtid; i < N; i++) {
        heap_try_insert(h, k, in[i]);
    }
    heap_sort_desc(h, k);
    block_merge_write(h, k, output);
}

extern "C" void solve(const float* input, float* output, int N, int k) {
    int blocks = 320;
    constexpr int BLK = 256;
    int maxBlocks = (N + BLK - 1) / BLK;
    if (blocks > maxBlocks) blocks = maxBlocks;
    if (blocks < 1) blocks = 1;

    float* cand = nullptr;
    cudaMalloc(&cand, (size_t)blocks * k * sizeof(float));

    topk_partial<<<blocks, BLK>>>(input, N, k, cand);
    topk_final  <<<1,      BLK>>>(cand, blocks * k, k, output);

    cudaFree(cand);
}