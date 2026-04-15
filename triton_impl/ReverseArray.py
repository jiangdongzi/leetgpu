import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(input_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 1. 获取当前 block 在网格中的 ID
    pid = tl.program_id(axis=0)
    
    # 2. 计算当前 block 负责的左侧索引 (从左向右)
    left_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 生成掩码：确保我们最多只处理到数组的中点 (N // 2)
    mask = left_offsets < (N // 2)
    
    # 4. 计算与左侧对称的右侧索引 (从右向左)
    # 对于索引 i，其对称位置是 N - 1 - i
    right_offsets = N - 1 - left_offsets
    
    # 5. 读取左右两侧的值
    left_vals = tl.load(input_ptr + left_offsets, mask=mask)
    right_vals = tl.load(input_ptr + right_offsets, mask=mask)
    
    # 6. 交叉写回，实现 In-place 反转
    tl.store(input_ptr + left_offsets, right_vals, mask=mask)
    tl.store(input_ptr + right_offsets, left_vals, mask=mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    # 边界情况保护：如果数组长度为 0 或 1，不需要反转，也可防止 grid=(0,) 导致内核启动报错
    if N <= 1:
        return
        
    BLOCK_SIZE = 1024
    
    # 由于每次迭代同时处理左右两端的元素，总工作量仅为 N // 2
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)