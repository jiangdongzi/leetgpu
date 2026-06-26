#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kvattn_kernal(const float* Q, const int8_t* K_int8, const int8_t* V_int8, const float* k_scale,
                                const float* v_scale, float* output, const int num_heads, const int seq_len, const int head_dim) {
    const int h = blockIdx.x;
    const float* myQ = Q + h * head_dim;
    const int8_t* myK = K_int8 + h * seq_len * head_dim;
    const int8_t* myV = V_int8 + h * seq_len * head_dim;
    const float* myKS = k_scale + h * seq_len;
    const float* myVS = v_scale + h * seq_len;
}

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {}
