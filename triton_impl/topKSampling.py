import torch

def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    # 初始化独立的随机数生成器，避免污染全局的 torch.manual_seed
    gen = torch.Generator(device=logits.device)
    gen.manual_seed(int(seed.item()))

    # ==========================================
    # 1. Softmax 转换为概率分布
    # PyTorch 底层已自动实现了减去 max(logits) 的数值稳定优化
    # ==========================================
    probs = torch.softmax(logits, dim=-1)

    # ==========================================
    # 2. 概率降序排序
    # 同时返回排序后的概率和在原词表中的索引
    # ==========================================
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # ==========================================
    # 3. 寻找 Nucleus (核心集合)
    # ==========================================
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 核心逻辑：当前 token 之前的累加概率（不含自身）严格小于 p
    # 这能确保刚好跨过阈值 p 的那一个 token 被包含在内，而之后的全部被截断
    prev_cumsum = cumsum_probs - sorted_probs
    nucleus_mask = prev_cumsum < p

    # ==========================================
    # 4. 重新归一化
    # ==========================================
    # 将不在 Nucleus 中的概率置为 0
    nucleus_probs = sorted_probs * nucleus_mask
    renorm_probs = nucleus_probs / nucleus_probs.sum()

    # ==========================================
    # 5. 随机采样与写回
    # ==========================================
    # torch.multinomial 要求输入一维权重张量，底层使用 Alias Method 或二分查找完成 O(1)/O(logN) 采样
    sampled_sorted_idx = torch.multinomial(renorm_probs, num_samples=1, generator=gen)

    # 通过排好序的 index 映射回原始词表中的真实 token id
    final_token = sorted_indices[sampled_sorted_idx]

    # 原地 (in-place) 覆盖传入的输出张量，保持外部指针有效
    sampled_token.copy_(final_token.view_as(sampled_token))

if __name__ == "__main__":
    # 模拟输入 (Example 1)
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
    p = torch.tensor(0.9)
    seed = torch.tensor(42)
    vocab_size = 4
    
    # 【核心修正】：分配一个长度为 1 的 long 类型张量来接收结果，而不是长度为 4
    sampled_token = torch.empty((1,), dtype=torch.long)
    
    # 调用函数
    solve(logits, p, seed, sampled_token, vocab_size)
    
    # 打印结果
    print(f"Sampled Token ID: {sampled_token.item()}")