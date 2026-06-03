#include <cuda_runtime.h>

// ============================================================================
// 高性能 SGEMM (Single-precision GEMM) kernel
//
// 计算: C[M, N] = A[M, K] * B[K, N]
//
// 主要优化技巧:
//   1. 三级分块: Block tile (128x128) -> Warp tile -> Thread tile (8x8)
//   2. FLOAT4 向量化访存(LDG.128 / LDS.128 / STS.128 / STG.128)
//   3. A 矩阵在 shared memory 中转置存储,便于沿 M 方向连续读取
//   4. XOR swizzle 消除 shared memory bank conflict
//   5. Warp 内 lane 重排(z-order),配合 swizzle 进一步避免 conflict
//   6. Outer product (rank-1 update) 累加,极高的寄存器复用度
//
// 注: 为了便于讲解,这里用的是「单缓冲」版本 ——
//     主循环里串行地「加载一个 K tile -> 同步 -> 计算 -> 同步」,
//     没有 shared memory ping-pong,也没有寄存器双缓冲/软件流水。
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

    // 单缓冲 shared memory
    // sA 注意是 [BLOCK_K][BLOCK_M],即 A 在共享内存中是「转置」存放的:
    //   这样后面读 A 的「一列」(同一个 tk,沿 M 方向的 128 个数) 时是连续的,可被 FLOAT4 一把吃掉
    __shared__ float sA[BLOCK_K][BLOCK_M];
    __shared__ float sB[BLOCK_K][BLOCK_N];

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

    // 寄存器单缓冲:从 SMEM 取出本线程当前 tk 需要的那一列/一行
    //   tCsA[8]: 本线程在 K 维某一列上,A 的 8 个数(沿 M)
    //   tCsB[8]: 本线程在 K 维某一行上,B 的 8 个数(沿 N)
    float tCsA[THREAD_C_Y_TILE_SIZE];
    float tCsB[THREAD_C_X_TILE_SIZE];

    // ======================================================================
    // 主循环: 每次处理一个 K tile,串行地「加载 -> 同步 -> 计算 -> 同步」
    // ======================================================================
    for (int k = 0; k < K; k += BLOCK_K) {

        // -----------------------------------------------------------------
        // 阶段 1: 从 GMEM 加载当前 K tile 到 shared memory
        // -----------------------------------------------------------------

        // 加载 A 的当前 K tile (一段 [k, k+BLOCK_K) 的列):
        //   256 线程在 8x32 网格里扫过 BLOCK_M=128 行,步长 32,共扫 4 轮
        //
        //   写入位置: sA[tAx][(i + tAy) ^ (tAx << 2)]
        //     - 行索引 = tAx (K 方向)
        //     - 列索引 = (i + tAy) ^ (tAx << 2)   ← XOR swizzle,避免 bank conflict
        //   因为 sA 是 [K][M] 转置存放,所以这里 A 的 (r, c) 进入 sA[tAx][...]
        # pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
            int r = row0 + i + tAy;
            int c = k + tAx;
            sA[tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

        // 加载 B 的当前 K tile (一段 [k, k+BLOCK_K) 的行):
        //   256 线程在 32x8 网格里覆盖 BLOCK_K=8 行 x BLOCK_N=128 列,每线程负责 1 行 4 列
        # pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
            int r = k + i + tBy;
            # pragma unroll
            for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                int c = col0 + j + tBx;
                sB[i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
            }
        }
        __syncthreads();  // 等所有线程把当前 tile 装好,才能开始读

        // -----------------------------------------------------------------
        // 阶段 2: 计算当前 K tile (沿 BLOCK_K 做 BLOCK_K 次 rank-1 update)
        // -----------------------------------------------------------------
        # pragma unroll
        for (int tk = 0; tk < BLOCK_K; ++tk) {

            // ===== 把 tk 那一列/一行从 SMEM 取到寄存器 (SMEM → Reg) =====
            // 加载 A 的 tk 列 (本线程负责的 8 个 M 维元素):
            //   8 个分成 2 组,每组 4 个用 FLOAT4 一次取
            //   r 的计算用了 (tCy + tm*16) << 2,把每线程 8 个输出按
            //   「4 连续 + 跨 64 + 4 连续」的方式分布,正好对应写回时的 stride
            # pragma unroll
            for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm ++) {
                int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                // ★ XOR swizzle: 读地址用 (r ^ (tk << 2)),与写入时的 swizzle 模式对偶
                FLOAT4(tCsA[tm << 2]) = FLOAT4(sA[tk][r ^ (tk << 2)]);
            }
            // 加载 B 的 tk 行 (本线程负责的 8 个 N 维元素):
            //   sB 没有转置,本身行内就是 N 方向连续,直接 FLOAT4
            # pragma unroll
            for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
                int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                FLOAT4(tCsB[tn << 2]) = FLOAT4(sB[tk][c]);
            }

            // ===== Outer product (rank-1 update) =====
            //   把 A 的「一列 8 个数」× B 的「一行 8 个数」,累加到 acc[8][8]
            //   一次 16 次 load → 64 次 FMA,算访比 4,寄存器复用极高
            # pragma unroll
            for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm ++) {
                # pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn ++) {
                    acc[tm][tn] += tCsA[tm] * tCsB[tn];
                }
            }
        }
        __syncthreads();  // 等所有线程算完,下一轮才能覆盖 shared memory
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
//
// ⚠ 注意:下面调用 kernel 时把 (M, N, K) 传成了 (M, K, N),且 grid.x 用了 K。
//   只有当外部传入的 N == K(或者非方阵但凑巧)时结果才正确;
//   一般情况下应改为:
//     dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N,
//                        (M + BLOCK_M - 1) / BLOCK_M);
//     matrix_multiplication_kernel<<<...>>>(A, B, C, M, N, K);
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
