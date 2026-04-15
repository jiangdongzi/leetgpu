import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(
    a_ptr,           # 第一个矩阵的指针
    b_ptr,           # 第二个矩阵的指针
    c_ptr,           # 结果矩阵的指针
    n_elements,      # 矩阵中元素的总数 (N * N)
    BLOCK_SIZE: tl.constexpr # 每个程序实例（Program Instance）处理的元素数量
):
    # 1. 获取当前 Program 的 ID (在 1D Grid 中的位置)
    pid = tl.program_id(0)
    
    # 2. 计算当前 Block 处理的元素索引范围
    # offsets 是一个向量，表示当前 Block 在全局内存中的起始偏移量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. 创建 Mask 防止内存越界（当 n_elements 不是 BLOCK_SIZE 的整数倍时）
    mask = offsets < n_elements
    
    # 4. 从全局内存加载数据到 SRAM
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 5. 执行元素级加法
    c = a + b
    
    # 6. 将结果写回全局内存
    tl.store(c_ptr + offsets, c, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    # 设置 Block Size (通常为 2 的幂，如 1024)
    BLOCK_SIZE = 1024
    n_elements = N * N
    
    # 计算 Grid 大小：需要多少个 Block 才能覆盖所有元素
    # triton.cdiv 是向上取整除法
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动 Kernel
    matrix_add_kernel[grid](
        a, b, c, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )