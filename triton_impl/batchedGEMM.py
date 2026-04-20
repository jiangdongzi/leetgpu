import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_cl, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # 1. 映射 3D Grid ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # 2. 计算当前 Batch 在显存中的起始偏移量
    a_batch_offset = pid_batch * stride_al
    b_batch_offset = pid_batch * stride_bl
    c_batch_offset = pid_batch * stride_cl

    # 3. 计算当前 Block 在 M 和 N 维度上的索引范围
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 4. 初始化指针（基于 Batch 偏移量计算当前的 Block 位置）
    a_ptrs = a_ptr + a_batch_offset + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + b_batch_offset + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 5. 初始化累加器 (FP32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 6. K 维度的分块主循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 边界 Mask 保护：同时处理 M/N 的边界和 K 在最后一块的越界
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)

        # 加载数据，越界部分补零
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 矩阵乘加
        accumulator += tl.dot(a, b)
        
        # 推进指针到下一个 K Block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 7. 结果写回显存
    c_ptrs = c_ptr + c_batch_offset + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    # 针对 256x256x256 性能测试，64 的 Block Size 配合适当的 block_k 可以较好地平衡寄存器压力和计算吞吐
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # 配置 3D Grid: (num_pid_m, num_pid_n, num_batches)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        BATCH
    )

    # 启动 Kernel 并传入步长
    bmm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )