#include <cfloat>
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
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * 32 + lane_id;
    __shared__ float sQ[256];
    if (tid < head_dim) {
        sQ[tid] = myQ[tid];
    }
    __syncthreads();
    __shared__ float l_prev, m_prev, new_max, w1;
    if (tid == 0) {
        m_prev = -FLT_MAX;
        l_prev = 0.f;
        new_max = m_prev;
    }
    float head_dim_tid_v_sum = 0.f;
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    __shared__ float raw_weight_k[32];
    __shared__ float w2[32];
    for (int s = 0; s < seq_len; s += 32) {
        if (s + warp_id < seq_len) {
            float local_sum = 0.f;
            const int8_t* local_k = myK + (s + warp_id) * head_dim;
            for (int i = lane_id; i < head_dim; i += 32) {
                local_sum += sQ[i] * (float)(local_k[i]);
            }
            local_sum *= myKS[s + warp_id];
            for (int offset = 16; offset > 0; offset /= 2) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            }
            if (lane_id == 0) {
                raw_weight_k[warp_id] = local_sum * inv_sqrt_d;
            }
        } else if (lane_id == 0) {
            raw_weight_k[warp_id] = -FLT_MAX;
        }
        __syncthreads();
        if (warp_id == 0) {
            float score = raw_weight_k[lane_id];

            float local_max = score;
            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
            }

            // 只有 lane 0 的 local_max 是 tile 内最大值
            float nm = fmaxf(local_max, m_prev);

            // 从 lane 0 广播 new max
            nm = __shfl_sync(0xffffffff, nm, 0);

            float old_m = m_prev;
            float alpha = expf(old_m - nm);

            float tmp;
            if (score == -FLT_MAX) {
                tmp = 0.f;
            } else {
                tmp = expf(score - nm);
            }

            w2[lane_id] = tmp;

            float sum_tmp = tmp;
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_tmp += __shfl_down_sync(0xffffffff, sum_tmp, offset);
            }

            if (lane_id == 0) {
                w1 = alpha;
                l_prev = l_prev * alpha + sum_tmp;
                m_prev = nm;
            }
        }
        __syncthreads();
        if (tid < head_dim) {
            float v_sum = 0.f;
            for (int i = s; i < min(s + 32, seq_len); i++) {
                const float vScale = myVS[i];
                v_sum += myV[i * head_dim + tid] * vScale * w2[i - s];
            }
            head_dim_tid_v_sum = head_dim_tid_v_sum * w1 + v_sum;
        }
    }
    if (tid < head_dim) {
        output[h * head_dim + tid] = head_dim_tid_v_sum / l_prev;
    }
}

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    dim3 threads(32, 32);
    kvattn_kernal<<<num_heads, threads>>>(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
}
