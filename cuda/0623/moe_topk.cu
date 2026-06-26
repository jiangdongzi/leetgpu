#include <cfloat>
#include <cuda_runtime.h>

__global__ void moe_topk_kernal(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    const int b = blockIdx.x;
    const float* myL = logits + b * E;
    int* my_topk_indices = topk_indices + k * b;
    float* my_topk_weights = topk_weights + b * k;
    const int tid = threadIdx.x;
    bool choosed = false;
    __shared__ float s_max[256];
    __shared__ int s_idx[256];
    for (int i = 0; i < k; i++) {
        s_idx[tid] = tid;
        if (choosed || tid >= E) {
            s_max[tid] = -FLT_MAX;
        } else {
            s_max[tid] = myL[tid];
        }
        __syncthreads();
        for (int offset = 256 / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const float other_max = s_max[tid + offset];
                const float my_max = s_max[tid];
                if (other_max > my_max) {
                    s_max[tid] = other_max;
                    s_idx[tid] = s_idx[tid + offset];
                }
            }
            __syncthreads();
        }
        if (tid == s_idx[0]) {
            choosed = true;
            my_topk_indices[i] = tid;
        }
        __syncthreads();
    }
    const float max_val = myL[my_topk_indices[0]];
    float local_sum = 0.f;
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            local_sum += expf(myL[my_topk_indices[i]] - max_val);
        }
        s_max[0] = local_sum;
    }
    __syncthreads();
    local_sum = s_max[0];
    if (tid >= k) return;
    my_topk_weights[tid] = expf(myL[my_topk_indices[tid]] - max_val) / local_sum;
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    moe_topk_kernal<<<M, 256>>>(logits, topk_weights, topk_indices, M, E, k);
}
