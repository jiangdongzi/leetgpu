#include <cfloat>
#include <cuda_runtime.h>

constexpr int WARPS_PER_BLOCK = 8;
constexpr int TILE = 8;
__global__ void mha_kernal(const float* Q, const float* K, const float* V, float* output, 
    const int N, const int d_model, const int h) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * 32 + lane_id;
    const int qi = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool valid = qi < N;
    const int head = blockIdx.y;
    const int d_k = d_model / h;
    const int col = head * d_k;
    const float scale = rsqrtf(d_k);
    float regs[32];
    const int nk_reg = (d_k - lane_id + 31) / 32;
    if (valid) {
        for (int i = 0; i < nk_reg; i++) {
            regs[i] = Q[qi * d_model + col + lane_id + i * 32];
        }
    }
    extern __shared__ float sm[];
    float* sK = sm;
    float* sV = sm + d_k * TILE;
    float acc[32] {0.f};
    float l_prev = 0.f, m_prev = -FLT_MAX;
    for (int k = 0; k < N; k += TILE) {
        const int num_keys = min(TILE, N - k);
        const int total_ele = num_keys * d_k;
        for (int idx = tid; idx < total_ele; idx += WARPS_PER_BLOCK *32) {
            const int r = idx / d_k, c = idx % d_k;
            sK[idx] = K[(k + r) * d_model + col + c];
            sV[idx] = V[(k + r) * d_model + col + c];
        }
        __syncthreads();
        if (valid) {
            for (int i = 0; i < num_keys; i++) {
                float local_sum = 0.f;
                for (int j = 0; j < nk_reg; j++) {
                    local_sum += regs[j] * sK[i * d_k + lane_id + j * 32];
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
                }
                local_sum *= scale;
                const float tmp_max = fmaxf(m_prev, local_sum);
                const float w1 = expf(m_prev - tmp_max);
                const float w2 = expf(local_sum - tmp_max);
                m_prev = tmp_max;
                l_prev = w1 * l_prev + w2;
                for (int j = 0; j < nk_reg; j++) {
                    acc[j] = w1 * acc[j] + w2 * sV[i * d_k + lane_id + j * 32];
                }
            }
        }
        __syncthreads();
    }
    if (valid) {
        for (int i = 0; i < nk_reg; i++) {
            output[qi * d_model + col + lane_id + i * 32] = acc[i] / l_prev;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    dim3 threadsPerBlock(32, WARPS_PER_BLOCK);
    dim3 blocksPerGrid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, h);
    const int sm_size = 2 * sizeof(float) * (d_model / h) * TILE;
    mha_kernal<<<blocksPerGrid, threadsPerBlock, sm_size>>>(Q, K, V, output, N, d_model, h);
}