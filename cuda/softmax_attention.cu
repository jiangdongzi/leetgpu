#include <cuda_runtime.h>

#define NEG_INF __int_as_float(0xff800000)
#define WARPS_PER_BLOCK 8
#define MAXK 32                     // 每个 lane 最多负责的维度数 = ceil(d_max/32) = 1024/32

// warp-per-query：一个 warp（32 lane）协作处理一个 query 行
// Q/acc 用定长寄存器数组（按 32 跨步映射维度），只 K/V tile 进 shared → d 任意大
__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V,
                                         float* output, int M, int N, int d, int tile) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int tid     = warp_id * 32 + lane_id;

    int qi = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = (qi < M);

    float scale = rsqrtf((float)d);
    int nk_reg = (d - lane_id + 31) / 32;   // 本 lane 实际负责的维度个数

    // 维度映射：lane_id 负责列 lane_id, lane_id+32, ...；q_reg[k]/acc[k] 是第 k 个分块
    float q_reg[MAXK];
    float acc[MAXK];
    #pragma unroll
    for (int k = 0; k < MAXK; ++k) { q_reg[k] = 0.f; acc[k] = 0.f; }

    if (valid)
        for (int k = 0; k < nk_reg; ++k)
            q_reg[k] = Q[qi * d + (lane_id + k * 32)];

    float m_prev = NEG_INF, l_prev = 0.f;

    extern __shared__ float smem[];
    float* sK = smem;                       // tile * d
    float* sV = sK + tile * d;              // tile * d

    for (int j_tile = 0; j_tile < N; j_tile += tile) {
        int num_keys = min(tile, N - j_tile);

        // 协同载入 K/V tile（合并访存）
        for (int idx = tid; idx < num_keys * d; idx += WARPS_PER_BLOCK * 32) {
            sK[idx] = K[j_tile * d + idx];
            sV[idx] = V[j_tile * d + idx];
        }
        __syncthreads();

        if (valid) {
            for (int jj = 0; jj < num_keys; ++jj) {
                // warp 协作点积
                float s = 0.f;
                for (int k = 0; k < nk_reg; ++k)
                    s += q_reg[k] * sK[jj * d + (lane_id + k * 32)];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    s += __shfl_down_sync(0xffffffff, s, off);
                float S = __shfl_sync(0xffffffff, s, 0) * scale;

                // online softmax 增量更新
                float m_curr = fmaxf(m_prev, S);
                float w1 = expf(m_prev - m_curr);
                float w2 = expf(S - m_curr);
                l_prev = l_prev * w1 + w2;

                for (int k = 0; k < nk_reg; ++k)
                    acc[k] = acc[k] * w1 + w2 * sV[jj * d + (lane_id + k * 32)];

                m_prev = m_curr;
            }
        }
        __syncthreads();
    }

    if (valid) {
        float inv_l = 1.f / l_prev;
        for (int k = 0; k < nk_reg; ++k)
            output[qi * d + (lane_id + k * 32)] = acc[k] * inv_l;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
    int tile = min(64, 6144 / d);           // 控制 shared = 2*tile*d*4 <= 48KB
    if (tile < 1) tile = 1;

    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t shmem = 2 * tile * d * sizeof(float);
    softmax_attention_kernel<<<grid, block, shmem>>>(Q, K, V, output, M, N, d, tile);
    cudaDeviceSynchronize();
}