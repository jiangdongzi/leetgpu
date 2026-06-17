#include <cuda_runtime.h>
#include <float.h>

// logits, topk_weights, topk_indices are device pointers

__device__ __inline__ bool inIdxes(const int * const idxes, const int cnt, const int idx) {
    for (int j = 0; j < cnt; j++) {
        if (idx == idxes[j]) return true;
    }
    return false;
}

__device__ __inline__ void find_max(const float* logits, const int E, int& idx, const int * const idxes, const int cnt) {
    const int tid = threadIdx.x;
    const int startIdx = E * blockIdx.x;
    int maxIdx = -1;
    for (int i = tid; i < E; i += blockDim.x) {
        if (inIdxes(idxes, cnt, i)) continue;
        if (maxIdx == -1 || logits[startIdx + i] > logits[startIdx + maxIdx]) {
            maxIdx = i;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        const int otherIdx = __shfl_down_sync(0xffffffff, maxIdx, offset);
        if (maxIdx == -1 || otherIdx != -1 && logits[startIdx + maxIdx] < logits[startIdx + otherIdx]) {
            maxIdx = otherIdx;
        }
    }
    if (tid == 0) {
        idx = maxIdx;
    }
}

void __global__ moe_kernal(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    const int startIdx = E * blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ int idxes[];
    for (int i = 0; i < k; i++) {
        int idx;
        find_max(logits, E, idx, idxes, i);
        if (tid == 0) {
            idxes[i] = idx;
            topk_indices[blockIdx.x * k + i] = idx;
        }
        __syncthreads();
    }
    const float max_weight = logits[startIdx + idxes[0]];
    float local_sum = 0.f;
    for (int i = tid; i < k; i += blockDim.x) {
        local_sum += expf(logits[startIdx + idxes[i]] - max_weight);
    }
    //归约求和
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    for (int i = tid; i < k; i += blockDim.x) {
        topk_weights[blockIdx.x * k + i] = expf(logits[startIdx + idxes[i]] - max_weight) / local_sum;
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    moe_kernal<<<M, 32, k * sizeof(int)>>>(logits, topk_weights, topk_indices, M, E, k);
}