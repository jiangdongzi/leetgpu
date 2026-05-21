#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <tuple>

// ============================================================================
// 高性能 SGEMM (Single-precision GEMM) kernel
//
// 计算: C[M, N] = A[M, K] * B[K, N]
//
// 主要优化技巧:
//   1. 三级分块: Block tile (128x128) -> Warp tile -> Thread tile (8x8)
//   2. Shared memory 双缓冲 (ping-pong),隐藏 GMEM 延迟
//   3. Register 双缓冲 + 软件流水(software pipelining),隐藏 SMEM 延迟
//   4. FLOAT4 向量化访存(LDG.128 / LDS.128 / STS.128 / STG.128)
//   5. A 矩阵在 shared memory 中转置存储,便于沿 M 方向连续读取
//   6. XOR swizzle 消除 shared memory bank conflict
//   7. Warp 内 lane 重排(z-order),配合 swizzle 进一步避免 conflict
//   8. Outer product (rank-1 update) 累加,极高的寄存器复用度
// ============================================================================

// 把 4 个连续 float 当作一个 float4 读/写,生成一条 128-bit 访存指令
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// ---------------- Block tile 形状 ----------------
constexpr int BLOCK_M = 128;        // 每个 block 输出 C 的 M 方向 tile 大小
constexpr int BLOCK_N = 128;        // 每个 block 输出 C 的 N 方向 tile 大小
constexpr int BLOCK_K = 8;          // K 方向每次推进的 tile 大小(内积深度)
constexpr int BLOCK_SIZE = 256;     // 每个 block 的线程数

// ---------------- 加载 A 时的线程布局 (256 线程视为 8 x 32 网格) ----------------
// 每行 8 个线程负责 K 方向的 8 列,32 行扫过 M 方向(每行步长 32)
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;  // 32

// ---------------- 加载 B 时的线程布局 (256 线程视为 32 x 8 网格) ----------------
// 每行 32 个线程负责 N 方向的 32 列,8 行扫过 K 方向
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;  // 8

// ---------------- 计算 C 时的线程布局 (256 线程视为 16 x 16 网格) ----------------
// 每个线程负责 (BLOCK_M/16) x (BLOCK_N/16) = 8 x 8 的输出 tile
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;  // 16
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;  // 8: 每线程 N 方向输出数
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;  // 8: 每线程 M 方向输出数

// ---------------- Warp 级 lane 布局 (32 lane 视为 8 x 4 网格) ----------------
// 把一个 warp 在输出空间内布成 8(列) x 4(行) 的 lane 矩阵
constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;              // 4
// 在 block 的 16x16 线程网格里,warp 在 X 方向的 warp 数:16/8 = 2
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;  // 2

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 本 block 负责输出 C 的左上角坐标 (row0, col0),范围 [row0, row0+BLOCK_M) x [col0, col0+BLOCK_N)
    int col0 = blockIdx.x * BLOCK_N;
    int row0 = blockIdx.y * BLOCK_M;
    int tidx = threadIdx.x;

    // 双缓冲 shared memory:第一维 [2] 是 ping/pong buffer
    // sA 注意是 [BLOCK_K][BLOCK_M],即 A 在共享内存中是「转置」存放的:
    //   这样后面读 A 的「一列」(同一个 tk,沿 M 方向的 128 个数) 时是连续的,可被 FLOAT4 一把吃掉
    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];

    // ---- 加载 A 时,本线程在 8x32 网格中的坐标 ----
    // tAx ∈ [0, 8): 决定 K 方向上的列;tAy ∈ [0, 32): 决定 M 方向上的起始行
    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // tidx % 8
    int tAy = tidx / THREAD_A_LAYOUT_X;         // tidx / 8

    // ---- 加载 B 时,本线程在 32x8 网格中的坐标 ----
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);   // tidx % 32
    int tBy = tidx / THREAD_B_LAYOUT_X;         // tidx / 32

    // ============ 计算阶段的 (tCy, tCx):本线程负责 C 中哪一格 ============
    // 256 线程被组织为 8 个 warp,先确定 warp 在 block 内的 2D 位置 (warpx, warpy)
    int warpId = tidx >> 5;                              // tidx / 32
    int laneId = tidx & 31;                              // tidx % 32
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1);      // warpId % 2
    int warpy = warpId / THREAD_C_WARP_DIM_X;            // warpId / 2

    // ★ Warp 内 32 个 lane 的「非常规」重排成 8x4(lanex, laney):
    //   - lanex = (laneId & 15) >> 1     ∈ [0, 8)
    //   - laney = ((laneId>>4)<<1) + (laneId & 1) ∈ [0, 4)
    //   相邻 lane(laneId 差 1) 只在 laney 上差 1,laneId 差 2 时 lanex 才 +1。
    //   配合下面的 XOR swizzle,可以在 LDS.128 / STG.128 时让 32 个 lane 落到 32 个不同 bank。
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);

    // 本线程在 block 输出网格 16x16 中的最终坐标
    int tCx = warpx * THREAD_C_WARP_X + lanex;  // ∈ [0, 16)
    int tCy = warpy * THREAD_C_WARP_Y + laney;  // ∈ [0, 16)

    // 本线程的输出累加寄存器 (8x8 = 64 个),也是寄存器使用的「大头」
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};

    // 寄存器双缓冲:tCsA/tCsB 各开两份,供软件流水使用
    //   tCsA[?][8]: 本线程在 K 维某一列上,A 的 8 个数(沿 M)
    //   tCsB[?][8]: 本线程在 K 维某一行上,B 的 8 个数(沿 N)
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];

    int bufferId = 0;  // 当前用于「计算」的 shared memory buffer 编号

    // ======================================================================
    // 阶段 1: 预取第 0 个 K tile 到 sA[0]/sB[0]
    // ======================================================================

    // 加载 A 的第 0 个 K tile (一段 [0, BLOCK_K) 的列):
    //   256 线程在 8x32 网格里扫过 BLOCK_M=128 行,步长 32,共扫 4 轮
    //
    //   写入位置: sA[0][tAx][(i + tAy) ^ (tAx << 2)]
    //     - 行索引 = tAx (K 方向)
    //     - 列索引 = (i + tAy) ^ (tAx << 2)   ← XOR swizzle,避免 bank conflict
    //   因为 sA 是 [K][M] 转置存放,所以这里 A 的 (r, tAx) 进入 sA[tAx][...]
    # pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f; 
    }

    // 加载 B 的第 0 个 K tile (一段 [0, BLOCK_K) 的行):
    //   256 线程在 32x8 网格里覆盖 BLOCK_K=8 行 x BLOCK_N=128 列,每线程负责 1 行 4 列
    # pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy;
        # pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f; 
        }
    }
    __syncthreads();  // 等所有线程把第 0 个 tile 装好

    // ======================================================================
    // 阶段 2: 主循环 - 计算当前 K tile,同时预取下一 K tile
    //
    // 循环上界写成 K + BLOCK_K 而不是 K,是因为「最后一轮」只算不取(if k<K 才取),
    // 这样可以把所有的取-算流水自然写在同一个 for 里。
    // ======================================================================
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {

        // -----------------------------------------------------------------
        // 内层 K 循环 (tk ∈ [0, BLOCK_K])。共 BLOCK_K+1 次迭代,实现「软件流水」:
        //   tk = 0:               只从 SMEM 预取 tk=0 到寄存器
        //   tk = 1..BLOCK_K-1:    一边算 tk-1,一边预取 tk
        //   tk = BLOCK_K:         只算最后一个 tk-1
        // -----------------------------------------------------------------
        # pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++ tk) {

            // ===== 计算上一轮预取好的 tk-1 =====
            if (tk > 0) {
                // ★ Outer product (rank-1 update):
                //   把 A 的「一列 8 个数」× B 的「一行 8 个数」,累加到 acc[8][8]
                //   一次 16 次 load → 64 次 FMA,算访比 4,寄存器复用极高
                # pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm ++) {
                    # pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn ++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }

            // ===== 预取 tk 那一列/一行到寄存器 (SMEM → Reg) =====
            if (tk < BLOCK_K) {
                // 加载 A 的 tk 列 (本线程负责的 8 个 M 维元素):
                //   8 个分成 2 组,每组 4 个用 FLOAT4 一次取
                //   r 的计算用了 (tCy + tm*16) << 2,把每线程 8 个输出按
                //   「4 连续 + 跨 64 + 4 连续」的方式分布,正好对应写回时的 stride
                # pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm ++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                    // ★ XOR swizzle: 读地址用 (r ^ (tk << 2)),与写入时的 swizzle 模式对偶
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
                // 加载 B 的 tk 行 (本线程负责的 8 个 N 维元素):
                //   sB 没有转置,本身行内就是 N 方向连续,直接 FLOAT4
                # pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }
        
        // -----------------------------------------------------------------
        // 与内层 K 循环并行地: 从 GMEM 预取「下一个」K tile 到另一半 shared memory
        // (双缓冲的关键 —— 计算 buf[bufferId] 的同时写 buf[bufferId^1])
        // -----------------------------------------------------------------
        if (k < K) {
            int c = k + tAx;  // 下一 K tile 中本线程要读的 A 列
            # pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f; 
            }
            # pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                # pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f; 
                }
            }
            __syncthreads();  // 等下一 tile 写完,才能在下一轮把它作为「当前」使用
        }
        bufferId ^= 1;  // 翻转 ping-pong buffer
    } 

    // ======================================================================
    // 阶段 3: 把 acc[8][8] 寄存器写回到 C
    //
    // 本线程的 8x8 输出在 C 上不是连续的 8x8 块,而是
    //   「4 连续 + 跨步 (THREAD_C_LAYOUT_X/Y) + 4 连续」的离散排布,
    // 与前面 FLOAT4 加载时的几何完全对应。这样 32 个 lane 同时 STG.128
    // 时,目标地址是 coalesced 的,带宽最大化。
    //
    // 索引拆解:
    //   r = row0 + (tCy << 2)               // 本线程的 4 连续行起点
    //         + (i & ~3) * THREAD_C_LAYOUT_Y  // 大跨步 (跳到下一个 4-行段)
    //         + (i & 3);                     // 4 连续中的偏移
    // ======================================================================
    # pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i ++) {
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        # pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j ++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// 形状: A[M, K], B[K, N], C[M, N] (行主序)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// ============================================================================
//                       压测 (Benchmark) 代码
// ============================================================================

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS Error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
    std::exit(1); } } while (0)

// 不带 device sync 的 launcher,方便用 cudaEvent 计时
static inline void launch_ours(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);
    matrix_multiplication_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// 用 cuBLAS 计算行主序的 C(M,N) = A(M,K) * B(K,N)
// cuBLAS 是列主序: 把行主序数据当列主序看,即看到的是各矩阵的转置。
//   令 C^T = B^T * A^T (列主序视角即可直接得到行主序的 C),
//   对应调用: m=N, n=M, k=K, A_blas=B(ldb=N), B_blas=A(lda=K), C_blas=C(ldc=N)
static inline void launch_cublas(cublasHandle_t handle,
                                 const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}

struct BenchResult {
    float ms;
    double tflops;
};

template <typename Fn>
static BenchResult time_kernel(Fn&& fn, double flops, int warmup, int iters) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, s, e));
    cudaEventDestroy(s);
    cudaEventDestroy(e);

    BenchResult r;
    r.ms = ms_total / iters;
    r.tflops = flops / (r.ms * 1e9);  // 2*M*N*K / (ms * 1e9) == TFLOPS
    return r;
}

static void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("================ Device Info ================\n");
    printf("  Device         : %s\n", p.name);
    printf("  Compute Cap.   : sm_%d%d\n", p.major, p.minor);
    printf("  SMs            : %d\n", p.multiProcessorCount);
    printf("  Max Threads/SM : %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Shared/Block   : %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("  Global Mem     : %.2f GB\n", p.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  Mem Bus        : %d-bit @ %.0f MHz (peak %.1f GB/s)\n",
           p.memoryBusWidth, p.memoryClockRate / 1e3,
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);
    printf("=============================================\n\n");
}

int main() {
    print_device_info();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 测试形状: 含方阵 + 长方阵 + 非 128 倍数的边界情况
    std::vector<std::tuple<int,int,int>> shapes = {
        {  256,   256,   256},
        {  512,   512,   512},
        { 1024,  1024,  1024},
        { 2048,  2048,  2048},
        { 4096,  4096,  4096},
        { 8192,  8192,  8192},
        // 非方阵
        { 1024,  4096,  1024},
        { 4096,  1024,  4096},
        { 4096,  4096,  1024},
        { 1024,  1024,  8192},
        { 3072,  3072,   768},
        // 非 BLOCK_M/N 倍数(测试边界处理)
        { 1000,  1000,  1000},
        { 4097,  4097,  4097},
    };

    printf("%-22s | %11s | %11s | %10s | %10s | %7s | %10s | %9s\n",
           "Shape (M, N, K)", "Ours (ms)", "cuBLAS (ms)",
           "Ours TF/s", "cuBLAS TF/s", "Ratio", "max|err|", "rel_F");
    printf("-----------------------+-------------+-------------+------------+------------+---------+------------+----------\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& shp : shapes) {
        int M = std::get<0>(shp), N = std::get<1>(shp), K = std::get<2>(shp);

        size_t szA = (size_t)M * K;
        size_t szB = (size_t)K * N;
        size_t szC = (size_t)M * N;

        std::vector<float> hA(szA), hB(szB), hC(szC), hRef(szC);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
        CUDA_CHECK(cudaMalloc(&dA,   szA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,   szB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC,   szC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, szC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

        // ===== 正确性验证 (与 cuBLAS 比较) =====
        launch_ours(dA, dB, dC, M, N, K);
        launch_cublas(handle, dA, dB, dRef, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC.data(),   dC,   szC * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, szC * sizeof(float), cudaMemcpyDeviceToHost));

        // Frobenius 相对误差: ||C - Cref||_F / ||Cref||_F
        // 这是 BLAS 实现间比较的标准做法,逐元素相对误差对接近 0 的元素会过敏(误报)
        double max_abs_err = 0.0, sum_diff2 = 0.0, sum_ref2 = 0.0;
        for (size_t i = 0; i < szC; ++i) {
            double d = (double)hC[i] - (double)hRef[i];
            double r = (double)hRef[i];
            sum_diff2 += d * d;
            sum_ref2  += r * r;
            double ae = std::fabs(d);
            if (ae > max_abs_err) max_abs_err = ae;
        }
        double frob_rel = std::sqrt(sum_diff2 / (sum_ref2 + 1e-30));

        // ===== 计时 =====
        // 小矩阵跑多一些迭代,大矩阵少一点避免太慢
        double flops = 2.0 * M * N * K;
        int iters = (M >= 4096 || N >= 4096) ? 20 : 50;

        BenchResult ro = time_kernel(
            [&]() { launch_ours(dA, dB, dC, M, N, K); }, flops, 5, iters);
        BenchResult rb = time_kernel(
            [&]() { launch_cublas(handle, dA, dB, dRef, M, N, K); }, flops, 5, iters);

        char shape_str[40];
        std::snprintf(shape_str, sizeof(shape_str), "(%d, %d, %d)", M, N, K);
        double ratio = ro.tflops / rb.tflops * 100.0;
        // FP32 SGEMM 在大规模下,Frobenius 相对误差 ~ sqrt(K) * eps_fp32 ≈ 1e-4 量级是正常的
        const char* mark = (frob_rel < 1e-3) ? " OK" : " !!";
        printf("%-22s | %9.3f   | %9.3f   | %8.2f   | %8.2f   | %5.1f%%  | %.3e  | %.2e%s\n",
               shape_str, ro.ms, rb.ms, ro.tflops, rb.tflops, ratio,
               max_abs_err, frob_rel, mark);
        fflush(stdout);

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }

    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
