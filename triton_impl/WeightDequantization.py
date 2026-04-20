import torch
import triton
import triton.language as tl

@triton.jit
def dequant_kernel(
    x_ptr, s_ptr, y_ptr,
    M, N, T,
    stride_xm, stride_xn,
    stride_sm, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1. 确定当前 Thread Block 的空间坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 计算当前 Block 负责的 M 和 N 维度上的全局坐标数组
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 3. 生成 Mask，防止在矩阵边缘引发越界访存 (Segfault)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 4. 计算量化矩阵 X 和输出矩阵 Y 的内存指针
    # 利用 stride 确保不管是 Row-major 还是 Column-major 的 Tensor 都能正确寻址
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)

    # 5. 核心：计算缩放矩阵 S 的映射指针
    # 直接在坐标层面上除以 TILE_SIZE 向下取整，这就是数学公式 row = i/T, col = j/T 的向量化表达
    s_row = offs_m // T
    s_col = offs_n // T
    s_ptrs = s_ptr + (s_row[:, None] * stride_sm + s_col[None, :] * stride_sn)

    # 6. 从 VRAM 并行 Load 数据
    # Triton 编译器会自动识别出连续的内存块，并在底层翻译为高带宽的全局内存读取指令
    x = tl.load(x_ptrs, mask=mask)
    s = tl.load(s_ptrs, mask=mask)

    # 7. 寄存器内计算：直接执行 element-wise 乘法
    # 这里蕴含了自动广播 (Broadcasting) 机制。同一个 Tile 内的元素会共享读取到的 S 值
    y = x * s

    # 8. 将结果写回 VRAM
    tl.store(y_ptrs, y, mask=mask)

def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    """
    宿主机 (Host) 调度代码，负责定义 Grid 大小并 Launch GPU Kernel
    """
    # 经验法则：对于这种纯粹的 element-wise IO 密集型算子，
    # 64x64 或者 128x128 是比较甜点的值。它能在 SM 的 Occupancy (占用率) 
    # 和 Register Spilling (寄存器溢出) 之间取得很好的平衡。
    BLOCK_M = 64
    BLOCK_N = 64

    # 定义 2D Grid，确保能覆盖整个 MxN 的矩阵
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    # Launch Kernel
    dequant_kernel[grid](
        X, S, Y,
        M, N, TILE_SIZE,
        X.stride(0), X.stride(1),
        S.stride(0), S.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

if __name__ == "__main__":
#     S = [
#   [0.5, 2.0],
#   [4.0, 0.25]
#     ]
    S = torch.tensor([[0.5, 2.0], [4.0, 0.25]], device='cuda')
    print(S.stride())