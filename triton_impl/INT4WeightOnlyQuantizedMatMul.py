import torch
import triton
import triton.language as tl

@triton.jit
def w4a16_matmul_kernel(
    x_ptr, w_q_ptr, scales_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_ym, stride_yn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 1. 获取当前 Block 的 Grid 坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 计算当前 Block 在 M 和 N 维度上的索引
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 3. 越界掩码 (M 和 N 维度在整个 K 循环中是固定的)
    mask_m = offs_am < M
    mask_n = offs_bn < N

    # 4. 计算各张量的基准指针
    # x: (M, K), row-major
    x_ptrs_base = x_ptr + offs_am[:, None] * stride_xm
    # w_q: (N, K//2), row-major
    w_q_ptrs_base = w_q_ptr + offs_bn[:, None] * stride_wn
    # scales: (N, K//group_size), row-major
    scales_ptrs_base = scales_ptr + offs_bn[:, None] * stride_sn

    # 初始化累加器，FP32 精度以防止累加溢出
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 5. 沿着 K 维度进行 Tiling 循环
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        mask_k = k_offs < K

        # 动态计算当前 K Block 的指针
        curr_x_ptrs = x_ptrs_base + k_offs[None, :] * stride_xk
        # 核心：w_q 的 K 维度实际上是 K//2
        curr_w_q_ptrs = w_q_ptrs_base + (k_offs[None, :] // 2) * stride_wk
        # 核心：scales 的 K 维度是 K//group_size
        curr_scales_ptrs = scales_ptrs_base + (k_offs[None, :] // group_size) * stride_sk

        mask_x = mask_m[:, None] & mask_k[None, :]
        mask_w = mask_n[:, None] & mask_k[None, :]

        # --- Load & Dequantize ---
        
        # 加载 Activation (FP16)
        x = tl.load(curr_x_ptrs, mask=mask_x, other=0.0)

        # 加载 Packed Weights (UINT8)
        w_q = tl.load(curr_w_q_ptrs, mask=mask_w, other=0)

        # 极致优化：用位运算代替 tl.where 分支提取 nibble
        # k_offs & 1 判断奇偶，^ 1 反转。
        # 偶数 k (2i):   (0 ^ 1) * 4 = 4 (取高 4 位)
        # 奇数 k (2i+1): (1 ^ 1) * 4 = 0 (取低 4 位)
        shift = ((k_offs & 1) ^ 1) * 4
        
        # 提取并转换为 INT4 (此时仍存在 int32/int16 寄存器中)
        w_int4 = (w_q >> shift[None, :]) & 0x0F

        # 转换为 float16 并减去偏移量 8
        w_deq = w_int4.to(tl.float16) - 8.0

        # 加载 Scales (FP16)，other=0.0 保证越界时 scale 为 0，不影响累加器
        scales = tl.load(curr_scales_ptrs, mask=mask_w, other=0.0)

        # 应用缩放因子 (FP16 * FP16)
        w_deq = w_deq * scales

        # --- MatMul ---
        
        # 此时 w_deq 的 shape 是 (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # 矩阵乘法需要 x @ W^T，所以我们对 w_deq 转置，送入 Tensor Core
        acc += tl.dot(x, tl.trans(w_deq))

    # 6. 转换回 FP16 并存储结果
    acc = acc.to(tl.float16)
    y_ptrs = y_ptr + offs_am[:, None] * stride_ym + offs_bn[None, :] * stride_yn
    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)

def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    # 为当前主流 GPU (如 Ada/Hopper 架构) 配置的 Block Size
    # 保证较高的 SM 占用率 (Occupancy) 和合理的寄存器使用
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )

    w4a16_matmul_kernel[grid](
        x, w_q, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_q.stride(0), w_q.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,    # 每个 CTA 128 个线程
        num_stages=3    # 软流水线级数，掩盖全局显存延迟
    )