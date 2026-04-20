import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    高性能单精度矩阵乘法 Kernel
    """
    pid = tl.program_id(axis=0)
    
    # 1. L2 Cache Swizzling: 重排 Block 的执行顺序，最大化同一组数据在 L2 Cache 中的复用率
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 计算当前 Block 的数据指针偏移
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. 核心计算循环
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 严格边界保护：应对 N 不是 BLOCK_SIZE 整数倍的情况 (如示例2中 N=3)
        a_mask = (offs_am[:, None] < M) & ((k * BLOCK_SIZE_K + offs_k)[None, :] < K)
        b_mask = ((k * BLOCK_SIZE_K + offs_k)[:, None] < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 题目要求标准的 32-bit 浮点乘法，关闭 TF32 以保证绝对精度
        accumulator += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 4. 写回显存
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    # 动态分配中间结果的显存
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 静态配置参数，对于 N=512 及附近规模的矩阵，64x64 的 Block 性价比最高
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c

def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    # 1. 恢复二维视图以便于步长(stride)处理
    A = input.view(N, N)
    
    # 2. 矩阵快速幂逻辑
    base = A
    res = None
    
    while P > 0:
        if P % 2 == 1:
            if res is None:
                # 首次赋值，避免与单位矩阵相乘的额外开销
                res = base
            else:
                res = matmul(res, base)
        P //= 2
        if P > 0:
            base = matmul(base, base)
            
    # 3. 将最终结果拍平并写入提供的 output 张量中
    if res is not None:
        output.copy_(res.view(-1))