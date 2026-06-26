#include <cfloat>
#include <cuda_runtime.h>

__device__ void merge(float& max_a, float& sum_a, const float max_b, const float sum_b) {
    const float tmp_max = fmaxf(max_a, max_b);
    const float w1 = expf(max_a - tmp_max);
    const float w2 = expf(max_b - tmp_max);
    max_a = tmp_max;
    sum_a = w1 * sum_a + w2 * sum_b;
}

__device__ void block_reduce(float& local_max, float& local_sum) {
    __shared__ float sm_max[256];
    __shared__ float sm_sum[256];
    const int tid = threadIdx.x;
    sm_max[tid] = local_max;
    sm_sum[tid] = local_sum;
    __syncthreads();
    for (int offset = 256 / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            const float other_max = sm_max[tid + offset];
            const float other_sum = sm_sum[tid + offset];
            float local_max = sm_max[tid];
            float local_sum = sm_sum[tid];
            merge(local_max, local_sum, other_max, other_sum);
            sm_max[tid] = local_max;
            sm_sum[tid] = local_sum;
        }
        __syncthreads();
    }
    if (tid == 0) {
        local_max = sm_max[0];
        local_sum = sm_sum[0];
    }
}

__global__ void ccel_kernal(const float* logits, const int C, const int* true_labels, float* loss, const int N) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int b = blockIdx.x;
    const float* myL = logits + C * b;
    float local_max = -FLT_MAX, local_sum = 0.f;
    for (int i = tid; i < C; i += stride) {
        merge(local_max, local_sum, myL[i], 1);
    }
    block_reduce(local_max, local_sum);
    if (tid == 0) {
        const float cur_loss = log(local_sum) + local_max - myL[true_labels[b]];
        atomicAdd(loss, cur_loss / N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    ccel_kernal<<<N, 256>>>(logits, C, true_labels, loss, N);
}