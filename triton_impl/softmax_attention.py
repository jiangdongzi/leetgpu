import torch
import triton
import triton.language as tl

@triton.jit
def _attn_kernel(
    Q, K, V, Out,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    M, N, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    # 确定当前 block 处理的 M 和 D 的索引范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Q 的数据指针与 Mask
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 初始化 Online Softmax 需要的统计量
    # m_i 维护每一行的局部最大值，初始为负无穷
    # l_i 维护每一行的指数和
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 累加器，用于最终乘以 V 的结果
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 沿 N 的维度分块迭代
    for start_n in range(0, N, BLOCK_N):
        start_n_offs = start_n + offs_n
        
        # K 和 V 的数据指针与 Mask
        k_ptrs = K + (start_n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = V + (start_n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        
        kv_mask = (start_n_offs[:, None] < N) & (offs_d[None, :] < d)
        
        # Load 分块的 K 和 V
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # 1. 计算 Q @ K^T 并缩放
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # 处理边界，防止越界区域影响最大值和 Softmax 计算
        valid_mask = (offs_m[:, None] < M) & (start_n_offs[None, :] < N)
        qk = tl.where(valid_mask, qk, float("-inf"))

        # 2. Online Softmax 核心逻辑
        # 更新局部最大值
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # 计算修正因子 (FlashAttention 消除数值溢出与重计算的关键)
        alpha = tl.exp(m_i - m_i_new)
        # 计算当前 block 的概率 P
        p = tl.exp(qk - m_i_new[:, None])
        
        # 更新分母
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # 3. 更新 Attention 输出累加器 (O = P @ V)
        acc = acc * alpha[:, None]
        # p.to(v.dtype) 是为了对齐数据类型以便走 Tensor Core
        acc += tl.dot(p.to(v.dtype), v)

        # 步进更新统计量
        m_i = m_i_new
        l_i = l_i_new

    # 4. 最终 Normalize
    acc = acc / l_i[:, None]

    # 将结果写回 Global Memory 的 Out 矩阵中
    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    out_mask = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)

def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    # tl.dot 在 GPU 上通常需要维度至少为 16 才能高效利用 Tensor Core
    # 这里将 BLOCK_D 向上取整到 2 的幂，并保底为 16，避免访存时的 Register Bank Conflict
    BLOCK_D = triton.next_power_of_2(d) if d > 0 else 16
    if BLOCK_D < 16:
        BLOCK_D = 16

    # 设定 SRAM Tiling 参数。可以根据具体的 L1 cache 规模做 Tuning
    BLOCK_M = 64
    BLOCK_N = 64
    
    # 缩放因子 1 / sqrt(d)
    sm_scale = 1.0 / (d ** 0.5)

    # 一维 Grid：沿着 M 维度切分 Blocks，N 维度在 Kernel 内部循环聚合
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    _attn_kernel[grid](  # pyright: ignore[reportArgumentType]
        Q, K, V, output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        M, N, d,
        sm_scale,
        BLOCK_M=BLOCK_M,  # pyright: ignore[reportArgumentType]
        BLOCK_N=BLOCK_N,  # pyright: ignore[reportArgumentType]
        BLOCK_D=BLOCK_D   # pyright: ignore[reportArgumentType]
    )

if __name__ == "__main__":
    # 测试代码
    M, N, d = 128, 128, 64
    Q = torch.randn(M, d, device='cuda')
    K = torch.randn(N, d, device='cuda')
    V = torch.randn(N, d, device='cuda')
    output = torch.empty(M, d, device='cuda')

    solve(Q, K, V, output, M, N, d)

    # 验证结果
    attn_ref = torch.nn.functional.softmax(Q @ K.T / (d ** 0.5), dim=-1) @ V
    assert torch.allclose(output, attn_ref, atol=1e-3), "结果不匹配！"
    print("测试通过！")