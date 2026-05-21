#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

constexpr int BLOCK_K = 8;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = 16;
constexpr int THREAD_C_X_TILE_SIZE = 8;
constexpr int THREAD_C_Y_TILE_SIZE = 8;
constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

enum LayoutKind {
    LAYOUT_SWIZZLED = 0,
    LAYOUT_NAIVE = 1,
};

enum BenchKind {
    BENCH_A_ONLY = 0,
    BENCH_B_ONLY = 1,
    BENCH_A_AND_B = 2,
};

struct Stats {
    double cycles_per_iter = 0.0;
    double cycles_per_warp = 0.0;
    double min_cycles = 0.0;
    double max_cycles = 0.0;
    float sink = 0.0f;
};

__device__ __forceinline__ float4 lds128(const float* ptr) {
    union {
        float4 f;
        unsigned int u[4];
    } out;

    unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(out.u[0]), "=r"(out.u[1]), "=r"(out.u[2]), "=r"(out.u[3])
        : "r"(smem_addr)
        : "memory");
    return out.f;
}

__device__ __forceinline__ float sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

template <int Layout, int Bench>
__global__ void shared_layout_bench_kernel(unsigned long long* cycles, float* sinks, int iters) {
    __shared__ __align__(16) float sA[BLOCK_K][BLOCK_M];
    __shared__ __align__(16) float sB[BLOCK_K][BLOCK_N];

    int tid = threadIdx.x;
    float* sA_linear = &sA[0][0];
    float* sB_linear = &sB[0][0];

    for (int i = tid; i < BLOCK_K * BLOCK_M; i += blockDim.x) {
        sA_linear[i] = static_cast<float>((i & 31) + 1);
    }
    for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
        sB_linear[i] = static_cast<float>(((i + 17) & 31) + 1);
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int warpx = warp_id & (THREAD_C_WARP_DIM_X - 1);
    int warpy = warp_id / THREAD_C_WARP_DIM_X;

    int lanex;
    int laney;
    if constexpr (Layout == LAYOUT_SWIZZLED) {
        lanex = (lane_id & 15) >> 1;
        laney = ((lane_id >> 4) << 1) + (lane_id & 1);
    } else {
        lanex = lane_id & 7;
        laney = lane_id >> 3;
    }

    int tCx = warpx * THREAD_C_WARP_X + lanex;
    int tCy = warpy * (WARP_SIZE / THREAD_C_WARP_X) + laney;

    float acc = 0.0f;
    __syncwarp();
    unsigned long long start = clock64();

#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        int tk = iter & (BLOCK_K - 1);

        if constexpr (Bench == BENCH_A_ONLY || Bench == BENCH_A_AND_B) {
#pragma unroll
            for (int tm = 0; tm < (THREAD_C_Y_TILE_SIZE >> 2); ++tm) {
                int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                float4 v = lds128(&sA[tk][r ^ (tk << 2)]);
                acc += sum4(v);
            }
        }

        if constexpr (Bench == BENCH_B_ONLY || Bench == BENCH_A_AND_B) {
#pragma unroll
            for (int tn = 0; tn < (THREAD_C_X_TILE_SIZE >> 2); ++tn) {
                int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                float4 v = lds128(&sB[tk][c]);
                acc += sum4(v);
            }
        }
    }

    __syncwarp();
    unsigned long long stop = clock64();

    if (lane_id == 0) {
        int out = blockIdx.x * WARPS_PER_BLOCK + warp_id;
        cycles[out] = stop - start;
        sinks[out] = acc;
    }
}

template <int Layout, int Bench>
Stats run_case(const char* layout_name,
               const char* bench_name,
               unsigned long long* d_cycles,
               float* d_sinks,
               int blocks,
               int iters,
               int repeats) {
    int samples = blocks * WARPS_PER_BLOCK;
    std::vector<unsigned long long> h_cycles(samples);
    std::vector<float> h_sinks(samples);

    Stats best{};
    bool have_best = false;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        shared_layout_bench_kernel<Layout, Bench><<<blocks, BLOCK_SIZE>>>(d_cycles, d_sinks, iters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, samples * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sinks.data(), d_sinks, samples * sizeof(float), cudaMemcpyDeviceToHost));

        unsigned long long sum = 0;
        unsigned long long min_v = h_cycles[0];
        unsigned long long max_v = h_cycles[0];
        float sink = 0.0f;

        for (int i = 0; i < samples; ++i) {
            sum += h_cycles[i];
            min_v = std::min(min_v, h_cycles[i]);
            max_v = std::max(max_v, h_cycles[i]);
            sink += h_sinks[i];
        }

        Stats current{};
        current.cycles_per_warp = static_cast<double>(sum) / samples;
        current.cycles_per_iter = current.cycles_per_warp / iters;
        current.min_cycles = static_cast<double>(min_v);
        current.max_cycles = static_cast<double>(max_v);
        current.sink = sink;

        if (!have_best || current.cycles_per_iter < best.cycles_per_iter) {
            best = current;
            have_best = true;
        }
    }

    std::printf("%-10s %-8s %14.4f %16.1f %12.0f %12.0f %12.1f\n",
                layout_name,
                bench_name,
                best.cycles_per_iter,
                best.cycles_per_warp,
                best.min_cycles,
                best.max_cycles,
                best.sink);
    return best;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : (1 << 16);
    int blocks = (argc > 2) ? std::atoi(argv[2]) : 256;
    int repeats = (argc > 3) ? std::atoi(argv[3]) : 7;

    if (iters <= 0 || blocks <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [iters>0] [blocks>0] [repeats>0]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int samples = blocks * WARPS_PER_BLOCK;
    unsigned long long* d_cycles = nullptr;
    float* d_sinks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cycles, samples * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sinks, samples * sizeof(float)));

    std::printf("Device: %s\n", prop.name);
    std::printf("iters=%d blocks=%d warps/block=%d repeats=%d\n\n",
                iters, blocks, WARPS_PER_BLOCK, repeats);
    std::printf("Each iteration issues two LDS.128 for A, two LDS.128 for B, or both.\n");
    std::printf("The address formulas match cuda/gemm.cu lines 179-190.\n\n");
    std::printf("%-10s %-8s %14s %16s %12s %12s %12s\n",
                "layout", "bench", "cycles/iter", "cycles/warp", "min", "max", "sink");
    std::printf("----------------------------------------------------------------------------------------\n");

    Stats sw_a = run_case<LAYOUT_SWIZZLED, BENCH_A_ONLY>("swizzled", "A-only", d_cycles, d_sinks,
                                                         blocks, iters, repeats);
    Stats sw_b = run_case<LAYOUT_SWIZZLED, BENCH_B_ONLY>("swizzled", "B-only", d_cycles, d_sinks,
                                                         blocks, iters, repeats);
    Stats sw_ab = run_case<LAYOUT_SWIZZLED, BENCH_A_AND_B>("swizzled", "A+B", d_cycles, d_sinks,
                                                           blocks, iters, repeats);
    Stats nv_a = run_case<LAYOUT_NAIVE, BENCH_A_ONLY>("naive", "A-only", d_cycles, d_sinks,
                                                      blocks, iters, repeats);
    Stats nv_b = run_case<LAYOUT_NAIVE, BENCH_B_ONLY>("naive", "B-only", d_cycles, d_sinks,
                                                      blocks, iters, repeats);
    Stats nv_ab = run_case<LAYOUT_NAIVE, BENCH_A_AND_B>("naive", "A+B", d_cycles, d_sinks,
                                                        blocks, iters, repeats);

    std::printf("\nRatios, naive / swizzled:\n");
    std::printf("  A-only: %.3fx\n", nv_a.cycles_per_iter / sw_a.cycles_per_iter);
    std::printf("  B-only: %.3fx\n", nv_b.cycles_per_iter / sw_b.cycles_per_iter);
    std::printf("  A+B   : %.3fx\n", nv_ab.cycles_per_iter / sw_ab.cycles_per_iter);

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_sinks));
    return 0;
}
