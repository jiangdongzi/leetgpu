#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <math.h>


#define NEG_INF __int_as_float(0xff800000)

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V, float* output, 
    int M, int N, int d, float scale) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    const int tid = 32 * warp_id + lane_id;
    const int i = blockIdx.x * 16 + warp_id;
    const bool valid_query = i < M;

    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (valid_query) {
        for (int k = 0; k < 4; k++) {
            const int col = k * 32 + lane_id;
            if (col < d) {
                q[k] = Q[i * d + col];
            }
        }
    }

    float m_prev = NEG_INF;
    float l_prev = 0;

    __shared__ float K_shared[32 * 128];
    __shared__ float V_shared[32 * 128];

    for (int j_tile = 0; j_tile < N; j_tile += 32) {
        const int num_keys = min(32, N - j_tile);
        const int total_elements = num_keys * d;

        const float* const K_ptr = K + j_tile * d;
        const float* const V_ptr = V + j_tile * d;

        for (int idx = tid; idx < total_elements; idx += 512) {
            K_shared[idx] =  K_ptr[idx];
            V_shared[idx] =  V_ptr[idx];
        }
        __syncthreads();

        if (valid_query) {
            for (int jj = 0; jj < num_keys; jj++) {
                float sum = 0;

                for (int k = 0; k < 4; k++) {
                    const int col = lane_id + k * 32;
                    if (col < d) {
                        sum += q[k] * K_shared[jj * d + col];
                    }
                }

                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }
                float S = __shfl_sync(0xffffffff, sum, 0);
                S *= scale;
                const float m_curr = fmaxf(S, m_prev);
                const float w1 = expf(m_prev - m_curr);
                const float w2 = expf(S - m_curr);

                l_prev = l_prev * w1 + w2;

                for (int k = 0; k < 4; k++) {
                    const int col = lane_id + 32 * k;
                    if (col < d) {
                        o_acc[k] = o_acc[k] * w1 + V_ptr[jj * d + col];
                    }
                }
                m_prev = m_curr;
            }
        }
        __syncthreads();
    }
    if (valid_query) {
        for (int k = 0; k < 4; k++) {
            const int col = lane_id + 32 * k;
            if (col < d) {
                output[i * d + col] = o_acc[k] / l_prev;
            }
        }
    }
}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const int warps_per_block = 16;
    dim3 block(32, warps_per_block);
    dim3 grid((M + warps_per_block - 1) / warps_per_block);
    const float scale = 1.0f / sqrtf(float(d));
}