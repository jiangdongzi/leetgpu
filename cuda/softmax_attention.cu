#include <cuda_runtime.h>

// 每个 block 负责一个 query 行（共 M 行），blockDim.x 个线程协作
__global__ void softmax_attn_kernel(const float* Q, const float* K, const float* V,
                                    float* output, int M, int N, int d) {
    int qi  = blockIdx.x;
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    float scale = rsqrtf((float)d);

    extern __shared__ float smem[];
    float* scores = smem;            // 长度 N

    // 1. scores[j] = (Q_qi · K_j) * scale
    for (int j = tid; j < N; j += nt) {
        const float* qrow = Q + qi * d;
        const float* krow = K + j  * d;
        float dot = 0.f;
        for (int t = 0; t < d; ++t) dot += qrow[t] * krow[t];
        scores[j] = dot * scale;
    }
    __syncthreads();

    // 2. softmax（block 内归约求 max 与 sum）
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

    // 3. output_qi = sum_j weight_j * V_j
    for (int t = tid; t < d; t += nt) {
        float acc = 0.f;
        for (int j = 0; j < N; ++j)
            acc += scores[j] * V[j * d + t];
        output[qi * d + t] = acc * inv_sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
    int threads = 256;
    dim3 grid(M);
    size_t shmem = N * sizeof(float);

    softmax_attn_kernel<<<grid, threads, shmem>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}