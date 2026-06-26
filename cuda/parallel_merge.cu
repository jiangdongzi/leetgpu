#include <algorithm>
#include <cuda_runtime.h>

__device__ int lower_bound(const float* A, const int M, const float val) {
    int l = 0, r = M;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (A[mid] < val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__device__ int upper_bound(const float* B, const int N, const float val) {
    int l = 0, r = N;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (B[mid] <= val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__global__ void merge_kernal(const float* A, const float* B, float* C, int M, int N) {
    const int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < M) {
        const float val = A[gtid];
        const int less_or_eq_val_cnt = upper_bound(B, N, val);
        C[gtid + less_or_eq_val_cnt] = val;
    }
    if (gtid < N) {
        const float val = B[gtid];
        const int less_val_cnt = lower_bound(A, M, val);
        C[gtid + less_val_cnt] = val;
    }
} 

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const int max_cnt = std::max(M, N);
    const int blocks = (max_cnt + 255) / 256;
    merge_kernal<<<blocks, 256>>>(A, B, C, M, N);
}