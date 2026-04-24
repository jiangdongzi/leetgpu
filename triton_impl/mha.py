import torch
import triton
import triton.language as tl

@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, d_model, h, d_k, sm_scale,
    stride_qn, stride_qh, stride_qd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # 获取当前的 Query Block ID 和 Head ID
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)

    # 计算行/列偏移量
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # 根据 head_id 和偏移量定位各矩阵在该头的基础指针
    q_ptrs = Q_ptr + head_id * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    o_ptrs = O_ptr + head_id * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd

    # 维护 Flash Attention 在线 Softmax 需要的统计量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 载入 Q 块 (BLOCK_M, BLOCK_D)
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d_k)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 遍历 K 和 V
    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + offs_n

        # 注意 K 需要按转置的维度加载: (BLOCK_D, BLOCK_N) 方便 tl.dot 做矩阵乘法
        k_ptrs = K_ptr + head_id * stride_qh + offs_d[:, None] * stride_qd + offs_n_curr[None, :] * stride_qn
        v_ptrs = V_ptr + head_id * stride_qh + offs_n_curr[:, None] * stride_qn + offs_d[None, :] * stride_qd

        k_mask = (offs_d[:, None] < d_k) & (offs_n_curr[None, :] < N)
        v_mask = (offs_n_curr[:, None] < N) & (offs_d[None, :] < d_k)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 1. Compute Q * K^T
        qk = tl.dot(q, k) * sm_scale
        # 对于超出边界的部分赋予极小值
        qk = tl.where((offs_m[:, None] < N) & (offs_n_curr[None, :] < N), qk, float("-inf"))

        # 2. Flash Attention 在线 Softmax 逻辑
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # 规避 padding row 产生 NaN
        m_i_new = tl.where(offs_m < N, m_i_new, 0.0) 
        
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_i_new)
        alpha = tl.where(offs_m < N, alpha, 0.0)
        
        l_i = l_i * alpha + l_ij
        
        # 3. Compute Attention * V
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # 归一化 Softmax 结果
    l_i = tl.where(offs_m < N, l_i, 1.0) # 避免除零
    acc = acc / l_i[:, None]
    
    # 将最终结果写回内存
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d_k)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    # 计算每个头的维度大小和缩放因子
    d_k = d_model // h
    sm_scale = 1.0 / (d_k ** 0.5)

    # 为满足 Triton 的 dot 算子要求，BLOCK_D 需要向上取到 2 的整数次幂，且至少为 16
    BLOCK_D = 16
    while BLOCK_D < d_k:
        BLOCK_D *= 2

    # 根据 BLOCK_D 动态调整 M 和 N 块大小，以防超量占用共享内存(Shared Memory OOM)
    BLOCK_M = 64
    BLOCK_N = 64
    if BLOCK_D >= 512:
        BLOCK_M = 16
        BLOCK_N = 16
    elif BLOCK_D >= 256:
        BLOCK_M = 16
        BLOCK_N = 32
    elif BLOCK_D >= 128:
        BLOCK_M = 32
        BLOCK_N = 32

    # 设置 Grid 空间：维度0负责 N 切分，维度1负责所有的 Head (h)
    grid = (triton.cdiv(N, BLOCK_M), h)

    # 启动 Kernel，stride 参数按行优先 `(N, d_model)` 的方式传入
    _mha_kernel[grid](
        Q, K, V, output,
        N, d_model, h, d_k, sm_scale,
        d_model, d_k, 1,  # stride_qn, stride_qh, stride_qd
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )