#include <cuda_runtime.h>
# define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
template<int Bm = 128, int Bn = 128, int Bk=8, int blockSize = 256, int A_BLOCK_X=8,
int B_BLOCK_X = 32, int C_BLOCK_X=16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K,
                                             int N) 
{
    __shared__ float As[Bk][Bm];
    __shared__ float Bs[Bk][Bn];

    // 需要计算的c 的tile 块的左上角
    int r0 = blockIdx.y*Bm; 
    int c0 = blockIdx.x*Bn;
    int tid = threadIdx.x;

    //*----tileA-----*
    constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X; // (8, 32)

    int A_THREAD_Y = tid / A_BLOCK_X;
    int A_THREAD_X = tid % A_BLOCK_X;

    //*----tileB-----*
    constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X; // (32, 8)

    int B_THREAD_Y = tid / B_BLOCK_X;
    int B_THREAD_X = tid % B_BLOCK_X;

    //*----tileC-----*
    constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X; //(16, 16)
    
    // 按照 8*4 排列 warp
    // int C_THREAD_Y = tid / C_BLOCK_X;
    // int C_THREAD_X = tid % C_BLOCK_X;

    // 计算当前 thread 在warp中 的 x,y 坐标
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    //计算总共有几行几列warp
    constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X; // 16/8=2
    
    // 计算 thread 在所在 warp 中 在block中的x, y 坐标
    int warpX = warpId % C_WARP_DIM_X; 
    int warpY = warpId / C_WARP_DIM_X;

    //计算thread在warp中的x, y坐标
    // int laneY = laneId / C_WARP_X;
    // int laneX = laneId % C_WARP_X;

    // z-order 排布
    int laneY = laneId % 2 + laneId /16 * 2;
    int laneX = laneId % 16 / 2;

    // 当前thread 在 blockC中的行列坐标 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
    int C_THREAD_Y = warpY * C_WARP_Y + laneY;
    int C_THREAD_X = warpX * C_WARP_X + laneX;

    // 每个thread 负责 Tm*TN 个元素计算
    constexpr int Tm = Bm / C_BLOCK_Y;
    constexpr int Tn = Bn / C_BLOCK_X;
    float Ct[Tm][Tn] = {0.0f};
    float regA[Tm] = {0.0f};
    float regB[Tn] = {0.0f};
    for(int k = 0; k < K; k += Bk){
        // read global Mem into shared Mem,行方向stride为 A_BLOCK_Y 列方向 stride为 A_BLOCK_X
        for(int i = A_THREAD_Y; i<Bm; i += A_BLOCK_Y){
            int r = r0 + i;
            for(int j = A_THREAD_X; j<Bk; j+= A_BLOCK_X){
                int c = k + j;
                As[j][i^(j << 2)] = (r<M && c <K)?A[r * K + c] : 0.f;
            }
        }

        // read global Mem into shared Mem, 行方向 stride 为 B_BLOCK_Y 列方向 stride为 B_BLOCK_X
        for(int i = B_THREAD_Y; i<Bk; i += B_BLOCK_Y){
            int r = k + i;
            for(int j = B_THREAD_X; j<Bn; j+= B_BLOCK_X){
                int c = c0 + j;
                Bs[i][j] = (r < K && c <N) ? B[r*N + c]: 0.f;
            } 
        }

        __syncthreads();

        // 计算tileA * tileB
        // 先循环 k 维度 计算外积
        for(int p=0; p<Bk; p++){
            // 存储 A 中列向量到 regA
            for(int i=0; i<Tm/4; i++){
                int r = (C_THREAD_Y + i * C_BLOCK_Y)*4;
                FLOAT4(regA[i*4]) = FLOAT4(As[p][r ^ (p<<2)]);
            }

            for(int i=0; i<Tn/4; i++){
                int c = (C_THREAD_X + i * C_BLOCK_X)*4;
                FLOAT4(regB[i*4]) = FLOAT4(Bs[p][c]); 
            }

            for(int i=0; i<Tm; i++){
                for(int j=0; j<Tn; j++){
                    Ct[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < Tm; i++){
        int r  = r0 + 4 * C_THREAD_Y + i/4 * 4 * C_BLOCK_Y + i % 4;
        for(int j = 0; j < Tn; j++){
            int c = c0 + 4 * C_THREAD_X + j/4 * 4 * C_BLOCK_X + j % 4;
            if(r < M && c < N) {C[r * N + c] = Ct[i][j];}
        }
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(256);
    int BN = 128;
    int BM = 128;
    dim3 blocksPerGrid((N + BN - 1) / BN,
                       (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
