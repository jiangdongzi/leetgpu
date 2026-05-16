#include <algorithm>
#include <cuda_runtime.h>

// 迭代式最小堆调整，避免 GPU 调用栈开销
__device__ void heapify(float* heap, int i, int n) {
    while (true) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && heap[left] < heap[smallest]) smallest = left;
        if (right < n && heap[right] < heap[smallest]) smallest = right;

        if (smallest != i) {
            float temp = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = temp;
            i = smallest;
        } else {
            break;
        }
    }
}

// 阶段一：每个 Block 维护一个 Shared Memory 最小堆，处理属于自己的数据块
__global__ void block_topk_kernel(const float* input, float* temp_output, int N, int k) {
    extern __shared__ float sdata[];
    float* heap = sdata;
    int* lock = (int*)&sdata[k]; // 动态共享内存布局：k个float + 1个int锁

    if (threadIdx.x == 0) *lock = 0;

    // 计算当前 Block 需要处理的数据范围
    int chunk_size = (N + gridDim.x - 1) / gridDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, N);
    int elements_in_chunk = chunk_end - chunk_start;

    // 初始化堆空间为极小值
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        heap[i] = -1e38f; 
    }
    __syncthreads();

    if (elements_in_chunk > 0) {
        // 【核心优化】：由 Thread 0 预先消费前 k 个元素并建堆。
        // 这可以在 O(k) 时间内建立一个极高的过滤阈值 (heap[0])，
        // 从而完美避开后续数百个线程同时争抢原子锁的“冷启动风暴”。
        if (threadIdx.x == 0) {
            int init_count = min(k, elements_in_chunk);
            for (int i = 0; i < init_count; ++i) {
                heap[i] = input[chunk_start + i];
            }
            for (int j = k / 2 - 1; j >= 0; j--) {
                heapify(heap, j, k);
            }
        }
        __syncthreads();

        // 剩余元素由 Block 内所有线程并行消费
        for (int i = chunk_start + k + threadIdx.x; i < chunk_end; i += blockDim.x) {
            float val = input[i];
            // 绝大多数元素在此处被无锁过滤（Warp 内部零冲突）
            if (val > heap[0]) {
                // 自旋锁保护临界区
                while (atomicCAS(lock, 0, 1) != 0) {}
                if (val > heap[0]) { // Double-check locking
                    heap[0] = val;
                    heapify(heap, 0, k);
                }
                atomicExch(lock, 0);
            }
        }
    }
    __syncthreads();

    // 将各 Block 的局部 Top-K 写入全局中间内存
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        temp_output[blockIdx.x * k + i] = heap[i];
    }
}

// 阶段二：全局归约与排序
__global__ void global_topk_kernel(const float* temp_input, float* output, int total_elements, int k) {
    extern __shared__ float sdata[];
    float* heap = sdata;
    int* lock = (int*)&sdata[k];

    if (threadIdx.x == 0) *lock = 0;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        heap[i] = -1e38f;
    }
    __syncthreads();

    if (total_elements > 0) {
        if (threadIdx.x == 0) {
            int init_count = min(k, total_elements);
            for (int i = 0; i < init_count; ++i) {
                heap[i] = temp_input[i];
            }
            for (int j = k / 2 - 1; j >= 0; j--) {
                heapify(heap, j, k);
            }
        }
        __syncthreads();

        for (int i = k + threadIdx.x; i < total_elements; i += blockDim.x) {
            float val = temp_input[i];
            if (val > heap[0]) {
                while (atomicCAS(lock, 0, 1) != 0) {}
                if (val > heap[0]) {
                    heap[0] = val;
                    heapify(heap, 0, k);
                }
                atomicExch(lock, 0);
            }
        }
    }
    __syncthreads();

    // 最终排序：利用堆排序机制，连续抽取最小值交换至尾部，
    // 原地实现降序排列 (Descending Order)
    if (threadIdx.x == 0) {
        for (int i = k - 1; i >= 0; i--) {
            float min_val = heap[0];
            heap[0] = heap[i];
            heap[i] = min_val;
            heapify(heap, 0, i); 
        }
        // 回写至结果数组
        for (int i = 0; i < k; ++i) {
            output[i] = heap[i];
        }
    }
}

extern "C" void solve(const float* input, float* output, int N, int k) {
    if (k <= 0 || N <= 0) return;
    if (k > N) k = N;

    int threads = 256;
    int blocks = 1024; // 1024 能够打满 T4 这类 GPU 的 SM 资源
    if (N / threads < blocks) {
        blocks = std::max(1, N / threads);
    }

    float* d_temp;
    cudaMalloc(&d_temp, blocks * k * sizeof(float));

    // Shared Memory 大小：k 个 float 存堆数据，外加 1 个 int 作为锁
    int shared_mem_size = (k + 1) * sizeof(float);

    block_topk_kernel<<<blocks, threads, shared_mem_size>>>(input, d_temp, N, k);
    
    // 单个 Block 收尾，汇聚 1024*k 个局部最值
    global_topk_kernel<<<1, 256, shared_mem_size>>>(d_temp, output, blocks * k, k);

    cudaFree(d_temp);
}