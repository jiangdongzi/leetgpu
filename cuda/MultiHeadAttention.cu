#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK_SIZE 256
#define TILE_N 32

__global__ void mha_tiled_online_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N,
    int d_model,
    int h
) {
    int q_row = blockIdx.x;   // 当前 query row
    int head  = blockIdx.y;   // 当前 head
    int tid   = threadIdx.x;

    int dk = d_model / h;
    int head_base = head * dk;

    extern __shared__ float smem[];

    float* q_cache = smem;                         // [dk]
    float* acc     = q_cache + dk;                 // [dk]
    float* k_tile  = acc + dk;                     // [TILE_N * dk]
    float* v_tile  = k_tile + TILE_N * dk;         // [TILE_N * dk]
    float* scores  = v_tile + TILE_N * dk;         // [TILE_N]
    float* reduce  = scores + TILE_N;              // [BLOCK_SIZE]

    // ------------------------------------------------------------
    // 1. 缓存当前 Q[q_row, head, :]，初始化 acc
    // ------------------------------------------------------------
    for (int d = tid; d < dk; d += BLOCK_SIZE) {
        q_cache[d] = Q[q_row * d_model + head_base + d];
        acc[d] = 0.0f;
    }

    __syncthreads();

    float scale = rsqrtf((float)dk);

    // online softmax 的全局状态
    float m = -FLT_MAX;
    float l = 0.0f;

    // ------------------------------------------------------------
    // 2. 按 TILE_N 扫描 K/V
    // ------------------------------------------------------------
    for (int tile_start = 0; tile_start < N; tile_start += TILE_N) {
        int tile_len = min(TILE_N, N - tile_start);

        // --------------------------------------------------------
        // 2.1 加载 K_tile / V_tile 到 shared memory
        //
        // 布局：
        //   k_tile[t * dk + d] = K[tile_start + t, head, d]
        //   v_tile[t * dk + d] = V[tile_start + t, head, d]
        // --------------------------------------------------------
        int total = tile_len * dk;

        for (int idx = tid; idx < total; idx += BLOCK_SIZE) {
            int t = idx / dk;
            int d = idx % dk;

            int global_row = tile_start + t;
            int global_off = global_row * d_model + head_base + d;

            k_tile[t * dk + d] = K[global_off];
            v_tile[t * dk + d] = V[global_off];
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.2 计算当前 query 对 tile 内每个 key 的 score
        //     scores[t] = dot(q, k_tile[t]) / sqrt(dk)
        //
        // 这里为了面试友好：
        //   对 tile 内每个 t，block 内并行算 dot
        // --------------------------------------------------------
        for (int t = 0; t < tile_len; ++t) {
            float local_dot = 0.0f;

            for (int d = tid; d < dk; d += BLOCK_SIZE) {
                local_dot += q_cache[d] * k_tile[t * dk + d];
            }

            reduce[tid] = local_dot;
            __syncthreads();

            for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                scores[t] = reduce[0] * scale;
            }

            __syncthreads();
        }

        // --------------------------------------------------------
        // 2.3 求 tile 内 max
        // --------------------------------------------------------
        float tile_max = -FLT_MAX;

        for (int t = tid; t < tile_len; t += BLOCK_SIZE) {
            tile_max = fmaxf(tile_max, scores[t]);
        }

        reduce[tid] = tile_max;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            }
            __syncthreads();
        }

        tile_max = reduce[0];

        // --------------------------------------------------------
        // 2.4 求 tile 内 exp sum
        // --------------------------------------------------------
        float tile_sum = 0.0f;

        for (int t = tid; t < tile_len; t += BLOCK_SIZE) {
            float e = expf(scores[t] - tile_max);
            scores[t] = e;
            tile_sum += e;
        }

        reduce[tid] = tile_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        tile_sum = reduce[0];

        // --------------------------------------------------------
        // 2.5 把当前 tile 的 softmax 状态合并到全局 online 状态
        //
        // 旧状态：
        //   m, l, acc
        //
        // 当前 tile 状态：
        //   tile_max, tile_sum, tile_acc
        //
        // 合并：
        //   m_new = max(m, tile_max)
        //   old_scale  = exp(m - m_new)
        //   tile_scale = exp(tile_max - m_new)
        //
        //   l_new = l * old_scale + tile_sum * tile_scale
        //
        //   acc_new =
        //       acc * old_scale
        //       +
        //       tile_acc * tile_scale
        // --------------------------------------------------------
        float m_new = fmaxf(m, tile_max);

        float old_scale  = expf(m - m_new);
        float tile_scale = expf(tile_max - m_new);

        // 每个线程负责若干个 d，计算 tile_acc[d]
        for (int d = tid; d < dk; d += BLOCK_SIZE) {
            float tile_acc = 0.0f;

            for (int t = 0; t < tile_len; ++t) {
                // scores[t] 当前存的是 exp(score[t] - tile_max)
                tile_acc += scores[t] * v_tile[t * dk + d];
            }

            acc[d] = acc[d] * old_scale + tile_acc * tile_scale;
        }

        l = l * old_scale + tile_sum * tile_scale;
        m = m_new;

        __syncthreads();
    }

    // ------------------------------------------------------------
    // 3. 写回 output = acc / l
    // ------------------------------------------------------------
    float inv_l = 1.0f / (l + 1e-20f);

    for (int d = tid; d < dk; d += BLOCK_SIZE) {
        output[q_row * d_model + head_base + d] = acc[d] * inv_l;
    }
}

extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int N,
    int d_model,
    int h
) {
    int dk = d_model / h;

    dim3 block(BLOCK_SIZE);
    dim3 grid(N, h);

    // q_cache[dk]
    // acc[dk]
    // k_tile[TILE_N * dk]
    // v_tile[TILE_N * dk]
    // scores[TILE_N]
    // reduce[BLOCK_SIZE]
    size_t shared_mem_size =
        ((size_t)dk * 2
        + (size_t)TILE_N * dk * 2
        + TILE_N
        + BLOCK_SIZE) * sizeof(float);

    mha_tiled_online_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, output, N, d_model, h
    );
}