#include <__clang_cuda_builtin_vars.h>
#include <algorithm>
#include <cuda_runtime.h>

__device__ int lower_bound(const float* arr, const int N, const float val) {
    int l = 0, r = N;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (arr[mid] < val) {
            l = mid + 1;
        } else {
            l = mid;
        }
    }
    return l;
}

__device__ int upper_bound(const float* arr, const int N, const float val) {
    int l = 0, r = N;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (arr[mid] <= val) {
            l = mid + 1;
        } else {
            l = mid;
        }
    }
    return l;
}

__global__ void merge(const float* A, const float* B, float* C, int M, int N) {
    const int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < M) {
        const float aVal = A[gtid];
        const int uB = upper_bound(B, N, aVal);
        C[gtid + uB] = aVal;
    }
    if (gtid < N) {
        const float bVal = B[gtid];
        const int lA = lower_bound(A, M, bVal);
        C[gtid + lA] = bVal;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const int max_l = std::max(M, N);
    merge<<<(max_l + 255) / 256, 256>>>(A, B, C, M, N);
}