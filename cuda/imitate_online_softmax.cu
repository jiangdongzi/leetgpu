#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>


__inline__ __device__ void warp_reduce_online(float& max_val, float& sum_val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
        const float tmp_max = fmaxf(other_max, max_val);
        sum_val = sum_val * expf(max_val - tmp_max) + other_sum * expf(other_max - max_val);
        max_val = tmp_max;
    }
}

__global__ void block_reduce_online(const float * const input, const int N, float* output_max, float* output_sum) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int gtid = 1024 * blockIdx.x + warp_id * 32 + lane_id;
    float max_val = -2e38f;
    float sum_val = 0.f;
    if (gtid < N) {
        max_val = input[gtid];
        sum_val = 1.f;
    }
    warp_reduce_online(max_val, sum_val);
    static __shared__ float smem_max[32];
    static __shared__ float smem_sum[32];
    if (lane_id == 0) {
        smem_max[warp_id] = max_val; 
        smem_sum[warp_id] = sum_val; 
    }
    __syncthreads();
    if (warp_id == 0) {
        max_val = smem_max[lane_id];
        sum_val = smem_sum[lane_id];
        warp_reduce_online(max_val, sum_val);
        if (lane_id == 0) {
            output_max[blockIdx.x] = max_val;
            output_sum[blockIdx.x] = sum_val;
        }
    }
}

__global__ void block_reduce_online(const float * const input_max, const float* const input_sum, const int N, float* output_max, float* output_sum) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int gtid = 1024 * blockIdx.x + warp_id * 32 + lane_id;
    float max_val = -2e38f;
    float sum_val = 0.f;
    if (gtid < N) {
        max_val = input_max[gtid];
        sum_val = input_sum[gtid];
    }
    warp_reduce_online(max_val, sum_val);
    static __shared__ float smem_max[32];
    static __shared__ float smem_sum[32];
    if (lane_id == 0) {
        smem_max[lane_id] = max_val; 
        smem_sum[lane_id] = sum_val; 
    }
    __syncthreads();
    if (warp_id == 0) {
        max_val = smem_max[lane_id];
        sum_val = smem_sum[lane_id];
        warp_reduce_online(max_val, sum_val);
        if (lane_id == 0) {
            output_max[blockIdx.x] = max_val;
            output_sum[blockIdx.x] = sum_val;
        }
    }
}

__device__ float global_max, global_sum;

void scan_block(const float* input_max, const float* input_sum, const int N) {
    dim3 threads(32, 32);
    const int blocks = (N + 1023) / 1024;
    float* output_max;
    cudaMalloc(&output_max, 4 * blocks);
    float* output_sum;
    cudaMalloc(&output_sum, 4 * blocks);
    block_reduce_online<<<blocks, threads>>>(input_max, input_sum, N, output_max, output_sum);
    if (blocks == 1) {
        global_max = output_max[0];
        global_sum = output_sum[0];
    } else {
        scan_block(output_max, output_sum, blocks);
    }
    //free mem
}

__global__ void normalize_kernel(const float* x, float* output, int N) {
    const int tid = threadIdx.x;
    const int bId = blockIdx.x;
    const int gtid = bId * 1024 + tid;
    if (gtid < N) {
        output[gtid] = expf(x[gtid] - global_max) / global_sum;
    }
}

extern "C" void solve(const float* x, float* output, int N) {
    dim3 threads(32, 32);
    const int blocks = (N + 1023) / 1024;
    float* output_max;
    cudaMalloc(&output_max, 4 * blocks);
    float* output_sum;
    cudaMalloc(&output_sum, 4 * blocks);
    block_reduce_online<<<blocks, threads>>>(x, N, output_max, output_sum);
    scan_block(output_max, output_sum, blocks);
    normalize_kernel<<<1024, threads>>>(x, output, N);
    cudaFree(output_max);
    cudaFree(output_sum);
}

static void cpu_softmax(const std::vector<float>& x, std::vector<float>& ref) {
    ref.resize(x.size());
    float m = -FLT_MAX;
    for (float v : x) m = std::max(m, v);
    float s = 0.f;
    for (float v : x) s += expf(v - m);
    for (size_t i = 0; i < x.size(); ++i) ref[i] = expf(x[i] - m) / s;
}

static void run_case(const char* name, const std::vector<float>& h_x) {
    const int N = static_cast<int>(h_x.size());
    printf("==== case: %s (N=%d) ====\n", name, N);

    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_out, sizeof(float) * N);
    cudaMemcpy(d_x, h_x.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    solve(d_x, d_out, N);
    cudaDeviceSynchronize();

    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    std::vector<float> ref;
    cpu_softmax(h_x, ref);

    float max_abs_err = 0.f;
    float gpu_sum = 0.f;
    for (int i = 0; i < N; ++i) {
        max_abs_err = std::max(max_abs_err, fabsf(h_out[i] - ref[i]));
        gpu_sum += h_out[i];
    }

    const int show = std::min(N, 8);
    printf("  gpu[0..%d] :", show - 1);
    for (int i = 0; i < show; ++i) printf(" %.6f", h_out[i]);
    printf("\n  ref[0..%d] :", show - 1);
    for (int i = 0; i < show; ++i) printf(" %.6f", ref[i]);
    printf("\n  gpu_sum = %.6f (expect 1.0)\n", gpu_sum);
    printf("  max_abs_err = %.6e -> %s\n\n", max_abs_err,
           max_abs_err < 1e-4f ? "PASS" : "FAIL");

    cudaFree(d_x);
    cudaFree(d_out);
}

int main() {
    // 用例1: 全 1 数组, 期望每个元素都是 1/N
    run_case("all_ones_small", std::vector<float>(8, 1.f));

    // 用例2: 全 1, 刚好一个 block (1024)
    run_case("all_ones_1024", std::vector<float>(1024, 1.f));

    // 用例3: 全 1, 跨多个 block
    run_case("all_ones_3000", std::vector<float>(3000, 1.f));

    // 用例4: 单个元素, softmax 必为 1.0
    run_case("single", std::vector<float>(1, 5.f));

    // 用例5: 递增序列, 验证非平凡分布
    {
        std::vector<float> x(5);
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(i);
        run_case("ramp_0_4", x);
    }

    return 0;
}