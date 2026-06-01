#include <cstddef>
#include <cuda_runtime.h>
#include <iterator>

constexpr int WARPS_PER_BLOCK = 8;
constexpr int TILE = 8;

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V,
                                         float* output, int M, int N, int d, int tile) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * 32 + lane_id;
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = smem + d * N * tile;
    const int qi = blockIdx.x *WARPS_PER_BLOCK + warp_id;
    const bool valid = qi < M;
    const float* qStart = Q + qi * d;
    const int nk_reg = (d - lane_id + 31) / 32;
    float q_reg[nk_reg];
    float acc[nk_reg];
    if (valid) {
        for (int i = 0; i < nk_reg; i++) {
            q_reg[i] = qStart[lane_id + i * 32];
            acc[i] = 0.f;
        }
    }
    const float scale = rsqrtf(d);
    float m_prev = -2e38f;
    float l_prev = 0.f;
    for (int j_tile = 0; j_tile < N; j_tile += tile) {
        const int num_keys = min(tile, N - j_tile);
        const int total_ele = d * num_keys;
        for (int i = tid; i < total_ele;  i += 32 * WARPS_PER_BLOCK) {
            sK[i] = K[j_tile * d + i];
            sV[i] = V[j_tile * d + i];
        }
        __syncthreads();
        for (int jj = 0; jj < num_keys; jj++) {
            float sm = 0.f;
            for (int i = 0; i < nk_reg; i++) {
                sm += q_reg[i] * sK[jj * d + lane_id + i * 32];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                sm += __shfl_down_sync(0xffffffff, sm, offset);
            }
            float S = __shfl_sync(0xffffffff, sm, 0);

            const float m_curr = fmaxf(S, m_prev);
            const float w1 = expf(m_prev - m_curr);
            const float w2 = expf(S - m_curr);
            m_prev = m_curr;
            l_prev = l_prev * w1 + w2;
            if (valid) {
                for (int i = 0; i < nk_reg; i++) {
                    acc[i] = acc[i] * w1 + w2 * sV[jj * d + lane_id + 32 * i];
                }
            }
        }
    }
    if (valid) {
        for (int i = 0; i < nk_reg; i++) {
            output[qi * d + lane_id + i * 32] = acc[i] / l_prev;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
    const float tile = 16;
    dim3 threads(32, 8);
    const int blocks = (M + 7) / 8;
    size_t smem = 2 * d * tile * sizeof(float);
    softmax_attention_kernel<<<blocks, threads, smem>>>(Q, K, V, output, M, N, d, tile);
}