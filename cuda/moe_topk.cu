#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

__global__ void moe_topk_kernal(const float* logits, float* topk_weights, int* topk_indices, const int M, const int E, const int k) {
    const int b = blockIdx.x;
    const float* myLogits = logits + b * E;
    int* my_topk_indices = topk_indices + b * k;
    float* my_tok_wights = topk_weights + b * k;
    const int tid = threadIdx.x;
    __shared__ float sVal[256];
    __shared__ int sIdx[256];
    bool in_indices = false;
    for (int i = 0; i < k; i++) {
        if (in_indices || tid >= E) {
            sVal[tid] = -FLT_MAX;
        } else {
            sVal[tid] = myLogits[tid];
            sIdx[tid] = tid;
        }
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const float my_val = sVal[tid];
                const float other_val = sVal[tid + offset];
                if (my_val < other_val) {
                    sVal[tid] = other_val;
                    sIdx[tid] = sIdx[tid + offset];
                }
            }
            __syncthreads();
        }
        if (sIdx[0] == tid) {
            in_indices = true;
            my_topk_indices[i] = tid;
        }
        __syncthreads();
    }
    __shared__ float max_logits[1];
    if (tid == 0) max_logits[0] = sVal[0];
    __syncthreads();
    float numerator;
    if (tid < k) {
        numerator = expf(myLogits[my_topk_indices[tid]] - max_logits[0]);
        sVal[tid] = numerator;
    } else {
        sVal[tid] = 0.f;
    }
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sVal[tid] += sVal[tid + offset];
        }
        __syncthreads();
    }
    if (tid < k) {
        my_tok_wights[tid] = numerator / sVal[0];
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    moe_topk_kernal<<<M, 256>>>(logits, topk_weights, topk_indices, M, E, k);
}
