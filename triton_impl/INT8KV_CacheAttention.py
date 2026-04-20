import torch
import triton
import triton.language as tl
import math

@triton.jit
def decode_attention_int8_kv_kernel(
    Q, K_int8, V_int8, k_scale, v_scale, Out,
    stride_qh,
    stride_kh, stride_ks,
    stride_vh, stride_vs,
    stride_k_sh, stride_v_sh,
    stride_oh,
    seq_len, head_dim, sm_scale,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # 1. 获取当前 Block 处理的 Head 索引
    pid = tl.program_id(0)

    # 2. 生成特征维度 (head_dim) 的偏移量和掩码
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_dim

    # 3. 加载当前 Head 对应的 Query 向量
    # Q shape: [num_heads, head_dim]
    q_ptr = Q + pid * stride_qh + offs_d
    q = tl.load(q_ptr, mask=mask_d, other=0.0)

    # 4. 初始化 Online Softmax 的累加器
    m_i = -float('inf')  # 局部最大值
    l_i = 0.0            # 局部指数和
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32) # 输出聚合向量

    # 序列维度的固定切块偏移
    offs_seq = tl.arange(0, BLOCK_SEQ)

    # 5. 沿着 Sequence 维度进行分块循环 (FlashAttention 核心逻辑)
    for start_m in range(0, seq_len, BLOCK_SEQ):
        start_seq = start_m + offs_seq
        mask_seq = start_seq < seq_len

        # --- a. 加载 K 并反量化 ---
        # k_ptrs shape: [BLOCK_SEQ, BLOCK_DMODEL]
        k_ptrs = K_int8 + pid * stride_kh + start_seq[:, None] * stride_ks + offs_d[None, :]
        mask_k = mask_seq[:, None] & mask_d[None, :]
        k_i8 = tl.load(k_ptrs, mask=mask_k, other=0.0)
        
        # k_s_ptrs shape: [BLOCK_SEQ]
        k_s_ptrs = k_scale + pid * stride_k_sh + start_seq
        k_s = tl.load(k_s_ptrs, mask=mask_seq, other=0.0)
        
        # 实时反量化为 FP32
        k_fp32 = k_i8.to(tl.float32) * k_s[:, None]

        # --- b. 计算 Scaled Dot-Product (Q * K^T) ---
        # q (广播到矩阵) * k_fp32，然后在 head_dim 维度求和
        qk = tl.sum(q[None, :] * k_fp32, axis=1) * sm_scale
        # 对于超出 seq_len 真实长度的 padding 部分，将分数置为负无穷
        qk = tl.where(mask_seq, qk, -float('inf'))

        # --- c. Online Softmax 状态更新 ---
        m_ij = tl.max(qk, axis=0)             # 当前 Block 的最大值
        m_new = tl.maximum(m_i, m_ij)         # 迄今为止的全局最大值
        
        alpha = tl.exp(m_i - m_new)           # 旧权重的衰减系数
        p = tl.exp(qk - m_new)                # 当前 Block 的相对概率
        
        l_new = l_i * alpha + tl.sum(p, axis=0) # 更新全局分母

        # --- d. 加载 V 并反量化 ---
        v_ptrs = V_int8 + pid * stride_vh + start_seq[:, None] * stride_vs + offs_d[None, :]
        v_i8 = tl.load(v_ptrs, mask=mask_k, other=0.0)
        
        v_s_ptrs = v_scale + pid * stride_v_sh + start_seq
        v_s = tl.load(v_s_ptrs, mask=mask_seq, other=0.0)
        
        v_fp32 = v_i8.to(tl.float32) * v_s[:, None]

        # --- e. 聚合 Output ---
        # acc 乘以衰减系数，并加上当前 Block 的贡献
        acc = acc * alpha + tl.sum(p[:, None] * v_fp32, axis=0)

        # 推进全局状态
        m_i = m_new
        l_i = l_new

    # 6. Softmax 最终除以全局分母归一化
    acc = acc / l_i

    # 7. 将结果写回主显存
    out_ptr = Out + pid * stride_oh + offs_d
    tl.store(out_ptr, acc, mask=mask_d)


def solve(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim):
    # 按照公式，缩放因子为 1 / sqrt(head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # 按照多头注意力机制的特性，将 1D Grid 设为 num_heads
    grid = (num_heads, )
    
    # 调优参数设定：
    # BLOCK_SEQ 控制每次处理的 token 数，对于 decode 阶段 64 或 128 是比较优的配置
    BLOCK_SEQ = 128
    # Triton 要求特征维度的 block size 必须是 2 的幂次，所以我们向上取整
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    # 启动 Kernel
    decode_attention_int8_kv_kernel[grid](
        Q, K_int8, V_int8, k_scale, v_scale, output,
        Q.stride(0),
        K_int8.stride(0), K_int8.stride(1),
        V_int8.stride(0), V_int8.stride(1),
        k_scale.stride(0),
        v_scale.stride(0),
        output.stride(0),
        seq_len, head_dim, sm_scale,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL
    )