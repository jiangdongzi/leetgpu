#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

constexpr int WARPS_PER_BLOCK = 8;
constexpr int TILE = 8;
__global__ void softmax_attn_kernal(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = lane_id + warp_id * 32;
    const int qi = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int valid = qi < M;
    const float scale = rsqrtf((float)d);
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = sK + TILE * d;
    float regs[4];
    const int nk_regs = (d - lane_id + 31) / 32;
    float acc[4]{0.f};
    float l_prev = 0.f, m_prev = -FLT_MAX;
    if (valid) {
        for (int i = 0; i < nk_regs; i++) {
            regs[i] = Q[qi * d + lane_id + 32 * i];
        }
    }
    for (int j_tile = 0; j_tile < N; j_tile += TILE) {
        const int n_keys = min(TILE, N - j_tile);
        for (int idx = tid; idx < n_keys * d; idx += WARPS_PER_BLOCK * 32) {
            sK[idx] = K[j_tile * d + idx];
            sV[idx] = V[j_tile * d + idx];
        }
        __syncthreads();
        if (valid) {
            for (int k = 0; k < n_keys; k++) {
                float S = 0.f;
                for (int j = 0; j < nk_regs; j++) {
                    S += regs[j] * sK[k * d + lane_id + j * 32];
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    S += __shfl_xor_sync(0xffffffff, S, offset);
                }
                S *= scale;
                const float tmp_max = fmaxf(S, m_prev);
                const float w1 = expf(m_prev - tmp_max);
                const float w2 = expf(S - tmp_max);
                m_prev = tmp_max;
                l_prev = l_prev * w1 + w2;
                for (int i = 0; i < nk_regs; i++) {
                    acc[i] = w1 * acc[i] + w2 * sV[k * d + lane_id + i * 32];
                }
            }
        }
        __syncthreads();
    }
    if (valid) {
        for (int i = 0; i < nk_regs; i++) {
            output[qi * d + lane_id + i * 32] = acc[i] / l_prev;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {}