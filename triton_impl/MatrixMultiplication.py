import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 获取一维的 Program ID
    pid = tl.program_id(axis=0)
    
    # 计算在 M 和 K 维度上的块数
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # L2 Cache 优化 (Swizzling):
    # 将线性的 pid 重新映射为一个 Group 结构，以提高数据在 L2 Cache 中的命中率
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    # 计算内存的指针偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # 预计算用于越界检查的 Mask（因为外部两维度的越界情况对于整个 N 的循环是不变的）
    mask_m = offs_m < M
    mask_k = offs_k < K

    # 初始化 A 和 B 的指针 (利用广播机制生成二维矩阵指针)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # 分配 FP32 类型的累加器寄存器 (块大小)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # 沿中间维度 N 进行分块迭代
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # 内部迭代的 N 维度 offset 和 Mask
        offs_n_curr = n * BLOCK_SIZE_N + offs_n
        mask_n = offs_n_curr < N
        
        # 安全地加载数据块（遇到越界则补 0.0）
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # 计算块矩阵的点积，并累加到 acc 寄存器中
        acc += tl.dot(a, b)
        
        # 指针步进：沿 N 维度步进一个 Block Size
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn

    # 将最终结果写回到 C 矩阵中
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    # 使用 a,b,c 自带的 strides 更加安全，能自动适配不同的连续性
    stride_am, stride_an = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_ck = c.stride()
    #print(stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck)
    print("输入矩阵的形状和 strides:")
    print(f"A: shape={a.shape}, strides={a.stride()}")
    print(f"B: shape={b.shape}, strides={b.stride()}")
    print(f"C: shape={c.shape}, strides={c.stride()}")


    # 针对大规模矩阵运算的超参数（经过 L2 调优的标准经验值）
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 32
    GROUP_SIZE_M = 8

    # 将网格定义为一维，以配合 L2 Swizzling 缓存优化策略
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), 
    )

    # 调用核函数
    matrix_multiplication_kernel[grid](
        a, b, c, 
        M, N, K, 
        stride_am, stride_an, 
        stride_bn, stride_bk, 
        stride_cm, stride_ck,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M
    )

if __name__ == "__main__":
    # 测试代码
    M, N, K = 512, 512, 512
    a = torch.randn(M, N, device='cuda')
    b = torch.randn(N, K, device='cuda')
    c = torch.zeros(M, K, device='cuda')

    solve(a, b, c, M, N, K)

    # 验证结果
    print ("验证结果...")
    #print c, a @ b
    print(torch.allclose(c, a @ b, atol=1e-1))
    assert torch.allclose(c, a @ b, atol=1e-2), "结果不正确！"

    print("测试通过！")