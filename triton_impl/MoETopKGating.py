import torch
import triton
import triton.language as tl

@triton.jit
def moe_topk_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    M, E,
    stride_logits_m, stride_logits_e,
    stride_weights_m, stride_weights_k,
    stride_indices_m, stride_indices_k,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # 1D Grid 映射：每一个 Program 负责处理一个 Token（矩阵的一行）
    pid = tl.program_id(0)
    if pid >= M:
        return

    # 计算当前行的起始指针
    logits_row_ptr = logits_ptr + pid * stride_logits_m
    
    # 构建当前行的内存偏移和 Mask（应对 E 不是 2 的幂次的情况）
    offsets_e = tl.arange(0, BLOCK_E)
    mask_e = offsets_e < E

    # 将该 Token 对应的所有专家的 Logits 一次性加载到寄存器
    # 越界部分填充负无穷大，避免影响 max 计算
    logits = tl.load(logits_row_ptr + offsets_e * stride_logits_e, mask=mask_e, other=-float('inf'))

    # ==========================================
    # 阶段一：迭代提取 Top-K 并计算 Exp (防溢出)
    # ==========================================
    
    # 1. 抽取第 1 大的值（特殊处理，因为它是 Softmax 防溢出的基准值）
    max_idx = tl.argmax(logits, axis=0)
    max_of_topk = tl.max(logits, axis=0) 
    
    # exp(max - max) = exp(0) = 1.0
    exp_val = 1.0 
    sum_exp = 1.0
    
    # 将第一个结果临时存入 HBM 输出张量中
    tl.store(topk_indices_ptr + pid * stride_indices_m + 0 * stride_indices_k, max_idx)
    tl.store(topk_weights_ptr + pid * stride_weights_m + 0 * stride_weights_k, exp_val)
    
    # 核心技巧：Mask 掉刚刚选中的最大值（设为负无穷），方便下一轮找次大值
    logits = tl.where(offsets_e == max_idx, -float('inf'), logits)

    # 2. 循环抽取剩余的 K-1 个最大值
    # 注意：Triton 中 range 的上界 K 如果是 tl.constexpr，循环会在编译期完美展开
    for i in range(1, K):
        max_idx = tl.argmax(logits, axis=0)
        max_val = tl.max(logits, axis=0)
        
        # 计算 Exp，必须减去全局最大值 max_of_topk
        exp_val = tl.exp(max_val - max_of_topk)
        sum_exp += exp_val
        
        # 存入 HBM
        tl.store(topk_indices_ptr + pid * stride_indices_m + i * stride_indices_k, max_idx)
        tl.store(topk_weights_ptr + pid * stride_weights_m + i * stride_weights_k, exp_val)
        
        # 继续 Mask
        logits = tl.where(offsets_e == max_idx, -float('inf'), logits)

    # ==========================================
    # 阶段二：计算最终的 Softmax 概率
    # ==========================================
    
    # 虽然我们需要对这 K 个值除以 sum_exp，但 Triton 不支持动态大小的局部数组。
    # 考虑到 K 通常极小（比如 K=2），直接从 L2 Cache/HBM 读回刚才存的 exp_val，
    # 做一次除法再写回去，性能依然可以拉满。
    for i in range(K):
        val_ptr = topk_weights_ptr + pid * stride_weights_m + i * stride_weights_k
        exp_val = tl.load(val_ptr)
        tl.store(val_ptr, exp_val / sum_exp)


def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    # 计算刚好能包裹住 E 的 2 的幂次作为 BLOCK_E
    # 例如 E=8 时 BLOCK_E=16，E=120 时 BLOCK_E=128
    BLOCK_E = triton.next_power_of_2(E)
    if BLOCK_E < 16:
        BLOCK_E = 16 

    grid = (M, )
    
    # 启动 Kernel，注意步长（stride）参数需要从 PyTorch Tensor 中提取
    moe_topk_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M, E,
        logits.stride(0), logits.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        K=k,
        BLOCK_E=BLOCK_E,
    )