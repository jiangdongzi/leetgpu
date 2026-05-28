#include <cuda_runtime.h>

// 每个 block 负责一个 (head, query_row) 组合
// blockDim.x 个线程协作处理该行
__global__ void mha_kernel(const float* Q, const float* K, const float* V,
                           float* output, int N, int d_model, int h, int d_k) {
    int head = blockIdx.y;          // 第几个 head
    int qi   = blockIdx.x;          // 第几行 query
    int tid  = threadIdx.x;
    int nt   = blockDim.x;

    int col_off = head * d_k;       // 该 head 在 d_model 中的列偏移
    float scale = rsqrtf((float)d_k);

    extern __shared__ float smem[];
    float* scores = smem;           // 长度 N，存放该 query 对所有 key 的分数

    // 1. 计算 scores[j] = (Q_qi · K_j) * scale
    for (int j = tid; j < N; j += nt) {
        const float* qrow = Q + qi * d_model + col_off;
        const float* krow = K + j  * d_model + col_off;
        float dot = 0.f;
        for (int t = 0; t < d_k; ++t) dot += qrow[t] * krow[t];
        scores[j] = dot * scale;
    }
    __syncthreads();

    // 2. softmax over scores（block 内归约求 max 与 sum）
    __shared__ float reduce[1024];

    // 求 max
    float local_max = -1e30f;
    for (int j = tid; j < N; j += nt) local_max = fmaxf(local_max, scores[j]);
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();

    // exp 并求 sum
    float local_sum = 0.f;
    for (int j = tid; j < N; j += nt) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.f / reduce[0];
    __syncthreads();

    // 3. output_qi = sum_j (weight_j * V_j)，每个线程负责若干输出列
    for (int t = tid; t < d_k; t += nt) {
        float acc = 0.f;
        for (int j = 0; j < N; ++j)
            acc += scores[j] * V[j * d_model + col_off + t];
        output[qi * d_model + col_off + t] = acc * inv_sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    int d_k = d_model / h;
    int threads = 256;

    dim3 grid(N, h);                // 每行每 head 一个 block
    size_t shmem = N * sizeof(float);

    mha_kernel<<<grid, threads, shmem>>>(Q, K, V, output, N, d_model, h, d_k);
    cudaDeviceSynchronize();
}