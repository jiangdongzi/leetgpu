import torch
import math

def solve(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, seq_len: int):
    # --- 超参数定义 ---
    d_model = 768
    n_heads = 12
    d_k = 64
    ffn_dim = 3072

    # --- 1. 权重解包 (Weight Unpacking) ---
    # 利用预先计算的偏移量(Offset)进行零拷贝(Zero-copy)的 Tensor 视图切片
    gamma1 = weights[0 : 768]
    beta1  = weights[768 : 1536]
    
    W_qkv  = weights[1536 : 1771008].view(d_model, d_model * 3)
    print("W_qkv shape:", W_qkv.shape)  # (768, 2304)
    b_qkv  = weights[1771008 : 1773312]
    
    W_attn = weights[1773312 : 2363136].view(d_model, d_model)
    b_attn = weights[2363136 : 2363904]
    
    gamma2 = weights[2363904 : 2364672]
    beta2  = weights[2364672 : 2365440]
    
    W_fc   = weights[2365440 : 4724736].view(d_model, ffn_dim)
    b_fc   = weights[4724736 : 4727808]
    
    W_proj = weights[4727808 : 7087104].view(ffn_dim, d_model)
    b_proj = weights[7087104 : 7087872]

    # --- Step 1: Layer Norm 1 ---
    # PyTorch 原生的 layer_norm 默认使用的是无偏方差 (divide by N)，完全吻合题意公式
    x_norm = torch.nn.functional.layer_norm(x, (d_model,), weight=gamma1, bias=beta1, eps=1e-5)

    # --- Step 2: QKV Projection ---
    # x_norm: (seq_len, 768) @ W_qkv: (768, 2304) -> qkv: (seq_len, 2304)
    qkv = torch.matmul(x_norm, W_qkv) + b_qkv

    # --- Step 3: Multi-Head Attention ---
    # 按照最后一个维度等分为 Q, K, V，每个形状为 (seq_len, 768)
    print("QKV shape:", qkv.shape)  # (seq_len, 2304)
    Q, K, V = qkv.split(d_model, dim=-1)

    # Reshape 并转置为 (n_heads, seq_len, d_k) 以便进行批量矩阵乘法
    Q = Q.view(seq_len, n_heads, d_k).transpose(0, 1)  # (12, seq_len, 64)
    K = K.view(seq_len, n_heads, d_k).transpose(0, 1)  # (12, seq_len, 64)
    V = V.view(seq_len, n_heads, d_k).transpose(0, 1)  # (12, seq_len, 64)

    # 计算 Scaled Dot-Product Attention (题目明确指出不使用 Causal Mask)
    # scores 形状: (12, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    
    # head_out 形状: (12, seq_len, 64)
    print("Attention weights shape:", attn_weights.shape)  # (12, seq_len, seq_len)
    print("V shape:", V.shape)  # (12, seq_len, 64)
    head_out = torch.matmul(attn_weights, V)

    # 拼接多头：先将头维度换回去 (seq_len, 12, 64)，再 view 展平为 (seq_len, 768)
    A = head_out.transpose(0, 1).reshape(seq_len, d_model)

    # --- Step 4: Output Projection & Step 5: Residual 1 ---
    P = torch.matmul(A, W_attn) + b_attn
    x_prime = x + P

    # --- Step 6: Layer Norm 2 ---
    h_norm = torch.nn.functional.layer_norm(x_prime, (d_model,), weight=gamma2, bias=beta2, eps=1e-5)

    # --- Step 7: Feed-Forward ---
    h = torch.matmul(h_norm, W_fc) + b_fc
    
    # 严格按照题意提供的 Tanh 近似公式计算 GELU
    gelu_h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * torch.pow(h, 3))))
    
    F = torch.matmul(gelu_h, W_proj) + b_proj

    # --- Step 8: Residual 2 ---
    # 使用 inplace copy_ 将结果写入预分配的 output Tensor 显存中，维持原函数签名
    output.copy_(x_prime + F)

if __name__ == "__main__":
    # 简单的功能测试
    seq_len = 128
    x = torch.randn(seq_len, 768)
    output = torch.empty_like(x)
    weights = torch.randn(7087872)  # 根据题目提供的权重总大小

    solve(x, output, weights, seq_len)
    print("Output shape:", output.shape)  # 应该是 (seq_len, 768)