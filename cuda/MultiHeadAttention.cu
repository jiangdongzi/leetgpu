#include <cuda_runtime.h>

#define NEG_INF __int_as_float(0xff800000)
#define WARPS_PER_BLOCK 8
#define MAXK 32                     // ceil(d_k_max / 32) = 1024/32

// warp-per-query：一个 warp 协作处理一个 (head, query 行)，d_k 任意大
__global__ void mha_kernel(const float* Q, const float* K, const float* V,
                           float* output, int N, int d_model, int h, int d_k, int tile) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int tid     = warp_id * 32 + lane_id;

    int head = blockIdx.y;
    int qi   = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = (qi < N);

    int col_off = head * d_k;
    float scale = rsqrtf((float)d_k);
    int nk_reg = (d_k - lane_id + 31) / 32;

    float q_reg[MAXK];
    float acc[MAXK];
    #pragma unroll
    for (int k = 0; k < MAXK; ++k) { q_reg[k] = 0.f; acc[k] = 0.f; }

    if (valid)
        for (int k = 0; k < nk_reg; ++k)
            q_reg[k] = Q[qi * d_model + col_off + (lane_id + k * 32)];

    float m_prev = NEG_INF, l_prev = 0.f;

    extern __shared__ float smem[];
    float* sK = smem;                       // tile * d_k
    float* sV = sK + tile * d_k;            // tile * d_k

    for (int j_tile = 0; j_tile < N; j_tile += tile) {
        int num_keys = min(tile, N - j_tile);

        // 载入该 head 的 K/V tile（按 d_model 跨步取 d_k 列）
        for (int idx = tid; idx < num_keys * d_k; idx += WARPS_PER_BLOCK * 32) {
            int r = idx / d_k, c = idx % d_k;
            sK[idx] = K[(j_tile + r) * d_model + col_off + c];
            sV[idx] = V[(j_tile + r) * d_model + col_off + c];
        }
        __syncthreads();

        if (valid) {
            for (int jj = 0; jj < num_keys; ++jj) {
                float s = 0.f;
                for (int k = 0; k < nk_reg; ++k)
                    s += q_reg[k] * sK[jj * d_k + (lane_id + k * 32)];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    s += __shfl_down_sync(0xffffffff, s, off);
                float S = __shfl_sync(0xffffffff, s, 0) * scale;

                float m_curr = fmaxf(m_prev, S);
                float w1 = expf(m_prev - m_curr);
                float w2 = expf(S - m_curr);
                l_prev = l_prev * w1 + w2;

                for (int k = 0; k < nk_reg; ++k)
                    acc[k] = acc[k] * w1 + w2 * sV[jj * d_k + (lane_id + k * 32)];

                m_prev = m_curr;
            }
        }
        __syncthreads();
    }

    if (valid) {
        float inv_l = 1.f / l_prev;
        for (int k = 0; k < nk_reg; ++k)
            output[qi * d_model + col_off + (lane_id + k * 32)] = acc[k] * inv_l;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    int d_k = d_model / h;
    int tile = min(64, 6144 / d_k);
    if (tile < 1) tile = 1;

    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, h);
    size_t shmem = 2 * tile * d_k * sizeof(float);
    mha_kernel<<<grid, block, shmem>>>(Q, K, V, output, N, d_model, h, d_k, tile);
    cudaDeviceSynchronize();
}