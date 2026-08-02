#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

constexpr int WARPS_PER_BLOCKS = 8;

__global__ void softmax_attn_kernal(const float* Q, const float* K, const float* V, float* output, int M, int N, const int d, const int tile) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = lane_id + warp_id * 32;
    float q_regs[4];
    const int nk_regs = (d - lane_id + 31) / 32;
    float l_max = -FLT_MAX, l_prev = 0.f;
    float acc[4] {0.f};
    extern __shared__  float sm[];
    float* sK = sm;
    float* sV = sm + tile * d;
    const int qi = blockIdx.x * WARPS_PER_BLOCKS + warp_id;
    const int valid = qi < M;
    if (valid) {
        for (int i = 0; i < nk_regs; i++) {
            q_regs[i] = Q[qi * d + lane_id + i * 32];
        }
    }
    const float scale = rsqrtf(d);
    for (int jj = 0; jj < N; jj += tile) {
        const int num_keys = std::min(tile, N - jj);
        const int total_eles = num_keys * d;
        for (int idx = tid; idx < total_eles; idx += 32 * WARPS_PER_BLOCKS) {
            sK[idx] = K[jj * d + idx];
            sV[idx] = V[jj * d + idx];
        }
        __syncthreads();
        if (valid) {
            for (int kk = 0; kk < num_keys; kk++) {
                float local_sum = 0.f;
                for (int i = 0; i < nk_regs; i++) {
                    local_sum += q_regs[i] * sK[kk * d + lane_id + i * 32];
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
                }
                local_sum *= scale;
                const float tmp_max = fmaxf(local_sum, l_max);
                const float w1 = expf(l_max - tmp_max);
                const float w2 = expf(local_sum - tmp_max);
                l_max = tmp_max;
                l_prev = w1 * l_prev + w2;
                for (int i = 0; i < nk_regs; i++) {
                    acc[i] = acc[i] * w1 + w2 * sV[kk * d + lane_id + i * 32];
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
                      int d) {
    dim3 threads(32, WARPS_PER_BLOCKS);
    const size_t smem_bytes = 2 * 8 * d * sizeof(float);
    softmax_attn_kernal<<<(M + WARPS_PER_BLOCKS - 1) / WARPS_PER_BLOCKS, threads, smem_bytes>>>(Q, K, V, output, M, N, d, 8);
}