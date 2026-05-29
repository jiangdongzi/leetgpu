#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// 面试友好的 online softmax 实现：
// 1. pass1: 每个 block 用 online 公式扫描一段输入，得到 block 级 (max, sum)。
// 2. pass2: 单个 block 继续归约所有 block 结果，得到全局 (max, sum)。
// 3. pass3: 再扫一遍输入，写出 exp(x - global_max) / global_sum。
//
// 这个版本避免 cooperative launch 和 grid.sync()，逻辑更容易讲清楚。
// 性能上虽然有 3 次 kernel launch、读输入两遍，但全程保持数值稳定，
// 且没有在 solve 中反复 cudaMalloc/cudaFree。

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

__device__ float g_block_max[kMaxBlocks];
__device__ float g_block_sum[kMaxBlocks];
__device__ float g_global_max;
__device__ float g_global_sum;

// 合并两个 online softmax 状态。
//
// 状态 (m, s) 表示：
//   m = 当前这部分数据的最大值
//   s = sum(exp(x_i - m))
//
// 如果要合并 A(m_a, s_a) 和 B(m_b, s_b)，新最大值是：
//   m = max(m_a, m_b)
//
// 两边的 sum 都要缩放到新的 m 作为基准：
//   s = s_a * exp(m_a - m) + s_b * exp(m_b - m)
__device__ __forceinline__ void merge_online(float& max_a,
                                             float& sum_a,
                                             float max_b,
                                             float sum_b) {
    const float new_max = fmaxf(max_a, max_b);
    sum_a = sum_a * expf(max_a - new_max) + sum_b * expf(max_b - new_max);
    max_a = new_max;
}

// 一个 warp 内做 online 状态归约。
// 归约完成后，lane 0 持有整个 warp 的结果。
__device__ __forceinline__ void warp_reduce_online(float& max_val,
                                                   float& sum_val) {
    unsigned mask = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_max = __shfl_down_sync(mask, max_val, offset);
        const float other_sum = __shfl_down_sync(mask, sum_val, offset);
        merge_online(max_val, sum_val, other_max, other_sum);
    }
}

// 一个 block 内做 online 状态归约。
// 先每个 warp 内归约，再用第一个 warp 归约所有 warp 的结果。
// 归约完成后，threadIdx.x == 0 持有整个 block 的结果。
__device__ __forceinline__ void block_reduce_online(float& max_val,
                                                    float& sum_val) {
    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    warp_reduce_online(max_val, sum_val);

    if (lane_id == 0) {
        warp_max[warp_id] = max_val;
        warp_sum[warp_id] = sum_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? warp_max[lane_id] : -FLT_MAX;
        sum_val = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
        warp_reduce_online(max_val, sum_val);
    }
}

// pass1: 每个 block 负责一部分输入，输出一个 block 级 online 状态。
__global__ void reduce_blocks_kernel(const float* x, int n) {
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // grid-stride loop 让 block 数可以固定上限，同时仍然覆盖任意长度输入。
    for (int i = global_tid; i < n; i += stride) {
        merge_online(local_max, local_sum, x[i], 1.0f);
    }

    block_reduce_online(local_max, local_sum);

    if (threadIdx.x == 0) {
        g_block_max[blockIdx.x] = local_max;
        g_block_sum[blockIdx.x] = local_sum;
    }
}

// pass2: 用一个 block 归约 pass1 的所有 block 结果，得到全局 max/sum。
__global__ void reduce_global_kernel(int num_blocks) {
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        merge_online(local_max, local_sum, g_block_max[i], g_block_sum[i]);
    }

    block_reduce_online(local_max, local_sum);

    if (threadIdx.x == 0) {
        g_global_max = local_max;
        g_global_sum = local_sum;
    }
}

// pass3: 根据全局 max/sum 写出 softmax 结果。
__global__ void normalize_kernel(const float* x, float* output, int n) {
    const float max_val = g_global_max;
    const float sum_val = g_global_sum;

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < n; i += stride) {
        output[i] = expf(x[i] - max_val) / sum_val;
    }
}

} // namespace

extern "C" void solve(const float* x, float* output, int N) {
    if (N <= 0) {
        return;
    }

    const int needed_blocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int blocks = std::min(needed_blocks, kMaxBlocks);

    reduce_blocks_kernel<<<blocks, kThreadsPerBlock>>>(x, N);
    reduce_global_kernel<<<1, kThreadsPerBlock>>>(blocks);
    normalize_kernel<<<blocks, kThreadsPerBlock>>>(x, output, N);
}

namespace {

// CPU 参考实现：数值稳定的 softmax。
void softmax_reference(const std::vector<float>& x, std::vector<float>& out) {
    const int n = static_cast<int>(x.size());
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(x[i] - max_val));
    }
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<float>(std::exp(static_cast<double>(x[i] - max_val)) / sum);
    }
}

// 跑一个测试用例：在 GPU 上调用 solve，并与 CPU 参考结果对比。
bool run_case(const char* name, const std::vector<float>& host_x) {
    const int n = static_cast<int>(host_x.size());

    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_x, d_out, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] %-18s CUDA error: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_x);
        cudaFree(d_out);
        return false;
    }

    std::vector<float> gpu_out(n);
    cudaMemcpy(gpu_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> ref_out(n);
    softmax_reference(host_x, ref_out);

    float max_abs_err = 0.0f;
    double gpu_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(gpu_out[i] - ref_out[i]));
        gpu_sum += gpu_out[i];
    }

    cudaFree(d_x);
    cudaFree(d_out);

    const float tol = 1e-5f;
    const bool ok = max_abs_err <= tol;
    printf("[%s] %-18s N=%-8d max_abs_err=%.3e  sum=%.6f\n",
           ok ? "PASS" : "FAIL", name, n, max_abs_err, gpu_sum);
    return ok;
}

} // namespace

int main() {
    srand(123);

    bool all_ok = true;

    // 1. 小规模、手工可推。
    all_ok &= run_case("small", {1.0f, 2.0f, 3.0f, 4.0f});

    // 2. 全相等输入：结果应为均匀分布 1/N。
    all_ok &= run_case("uniform", std::vector<float>(1000, 5.0f));

    // 3. 含很大值，检验数值稳定性（不会 inf/nan）。
    all_ok &= run_case("large-values",
                       {1000.0f, 1001.0f, 999.0f, 1000.5f, 998.5f});

    // 4. 含负数和正数混合。
    all_ok &= run_case("mixed", {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f, -10.0f});

    // 5. 单元素。
    all_ok &= run_case("single", {42.0f});

    // 6. 大规模随机，跨多个 block，验证 grid-stride + 多 pass 归约。
    {
        std::vector<float> big(1 << 20);
        for (auto& v : big) {
            v = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f;
        }
        all_ok &= run_case("large-random", big);
    }

    // 7. 超大规模，元素数远超 kMaxBlocks * kThreadsPerBlock，验证 grid-stride 覆盖。
    {
        std::vector<float> huge(5 * 1000 * 1000);
        for (auto& v : huge) {
            v = (static_cast<float>(rand()) / RAND_MAX) * 6.0f - 3.0f;
        }
        all_ok &= run_case("huge-random", huge);
    }

    printf("\n%s\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}