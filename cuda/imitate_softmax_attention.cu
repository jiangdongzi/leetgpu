#include <cstddef>
#include <cuda_runtime.h>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <float.h>

constexpr int WARPS_PER_BLOCK = 8;
constexpr int TILE = 8;

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V,
                                         float* output, int M, int N, int d, int tile) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * 32 + lane_id;
    extern __shared__ float smem[];
    float* sK = smem;
    // BUG 修复: 原为 smem + d * N * tile, 但 solve 只分配了 2*d*tile 个 float,
    // sV 应紧跟在 sK(d*tile 个)之后. 原偏移含 N 会严重越界 -> illegal memory access.
    float* sV = smem + d * tile;
    const int qi = blockIdx.x *WARPS_PER_BLOCK + warp_id;
    const bool valid = qi < M;
    const float* qStart = Q + qi * d;
    const int nk_reg = (d - lane_id + 31) / 32;
    // 原代码 float q_reg[nk_reg] 是 VLA(变长数组), CUDA 不支持, 无法编译.
    // 改为固定上界数组以便编译/测试, 算法不变(只用前 nk_reg 个元素). 支持 d <= 1024.
    constexpr int MAX_REG = 32;
    float q_reg[MAX_REG];
    float acc[MAX_REG];
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
            // BUG 修复: 标准 attention 为 softmax(QKᵀ/√d)V, 必须乘 scale(=rsqrtf(d)).
            // 原代码算了 scale 却没用, 漏掉了缩放.
            float S = __shfl_sync(0xffffffff, sm, 0) * scale;

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
        // BUG 修复: 循环末尾需要同步, 否则快的 warp 会在慢的 warp 还在读 sK/sV 时,
        // 就开始把下一个 tile 的数据写进来, 造成 shared memory 读写竞争.
        __syncthreads();
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
    cudaDeviceSynchronize();
}

// ---------------- 测试代码 ----------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            printf("  [CUDA error] %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(_e));                                     \
        }                                                                       \
    } while (0)

// CPU 参考: output = softmax(Q·Kᵀ * scale) · V
//   Q: M×d, K: N×d, V: N×d, output: M×d
//   use_scale=true 时 scale = 1/sqrt(d)（标准 attention）, 否则 scale = 1（无缩放）
static void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& out,
                          int M, int N, int d, bool use_scale) {
    out.assign((size_t)M * d, 0.f);
    const float scale = use_scale ? 1.f / sqrtf((float)d) : 1.f;
    std::vector<float> scores(N);
    for (int i = 0; i < M; ++i) {
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < d; ++k) s += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
            s *= scale;
            scores[j] = s;
            m = std::max(m, s);
        }
        float denom = 0.f;
        for (int j = 0; j < N; ++j) {
            scores[j] = expf(scores[j] - m);
            denom += scores[j];
        }
        for (int j = 0; j < N; ++j) {
            const float w = scores[j] / denom;
            for (int k = 0; k < d; ++k) out[(size_t)i * d + k] += w * V[(size_t)j * d + k];
        }
    }
}

static float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, fabsf(a[i] - b[i]));
    return e;
}

static void run_case(const char* name, int M, int N, int d, unsigned seed) {
    printf("==== case: %s (M=%d, N=%d, d=%d) ====\n", name, M, N, d);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Q((size_t)M * d), K((size_t)N * d), V((size_t)N * d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, sizeof(float) * Q.size()));
    CUDA_CHECK(cudaMalloc(&dK, sizeof(float) * K.size()));
    CUDA_CHECK(cudaMalloc(&dV, sizeof(float) * V.size()));
    CUDA_CHECK(cudaMalloc(&dO, sizeof(float) * Q.size()));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), sizeof(float) * Q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), sizeof(float) * K.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), sizeof(float) * V.size(), cudaMemcpyHostToDevice));

    solve(dQ, dK, dV, dO, M, N, d);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [kernel 执行失败] %s\n", cudaGetErrorString(err));
        printf("  -> 很可能是 shared memory 越界 (sV = smem + d*N*tile)\n\n");
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        cudaDeviceReset();  // 清掉 sticky error, 让后续用例还能继续
        return;
    }

    std::vector<float> gpu((size_t)M * d);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dO, sizeof(float) * gpu.size(), cudaMemcpyDeviceToHost));

    std::vector<float> ref_scaled, ref_noscale;
    cpu_attention(Q, K, V, ref_scaled, M, N, d, true);
    cpu_attention(Q, K, V, ref_noscale, M, N, d, false);

    const float e_scaled = max_abs_err(gpu, ref_scaled);
    const float e_noscale = max_abs_err(gpu, ref_noscale);

    const int show = std::min(d, 6);
    printf("  gpu  [row0] :");
    for (int k = 0; k < show; ++k) printf(" % .5f", gpu[k]);
    printf("\n  ref  [row0] :");
    for (int k = 0; k < show; ++k) printf(" % .5f", ref_scaled[k]);
    printf("\n");
    printf("  max_abs_err vs 标准(1/sqrt(d)) = %.3e -> %s\n",
           e_scaled, e_scaled < 1e-3f ? "PASS" : "FAIL");
    printf("  max_abs_err vs 无缩放(scale=1)  = %.3e -> %s\n\n",
           e_noscale, e_noscale < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

int main() {
    run_case("tiny",     2,   3,   4, 1);
    run_case("small",    4,   8,  16, 2);
    run_case("d32",      8,  10,  32, 3);
    run_case("odd_dims", 17, 20,  40, 4);  // 各维都不是 32 的倍数
    run_case("big_N",    8,  64,  64, 5);  // N、d 较大, 触发 shared memory 越界
    return 0;
}