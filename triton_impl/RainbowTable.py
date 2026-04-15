import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261

    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)

    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val


@triton.jit
def fnv1a_hash_kernel(input, output, n_elements, n_rounds, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # 1. 从全局内存加载数据 (此时 x 默认是 int32)
    x = tl.load(input + offsets, mask=mask)
    
    # 2. 修复点：强制转换为无符号的 uint32 
    # 这既保持了循环内变量类型的一致性，也确保了位运算的绝对正确
    x = x.to(tl.uint32)
    
    # 3. 迭代 R 轮进行 Hash 计算 (类型保持为 uint32 不变)
    for _ in range(n_rounds):
        x = fnv1a_hash(x)
        
    # 4. 将结果写回内存前，再转回原有的数据类型 (通常是 int32)
    tl.store(output + offsets, x.to(tl.int32), mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](input, output, N, R, BLOCK_SIZE)