import torch
import triton
import triton.language as tl

@triton.jit
def add_fn(a, b):
    return a + b

@triton.jit
def local_scan_kernel(
    data_ptr, output_ptr, block_sums_ptr, n,
    STORE_BLOCK_SUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 从全局内存加载数据到 SRAM，越界部分填充 0.0
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # 1. 块内局部前缀和 (Block-local Prefix Sum)
    local_scan = tl.associative_scan(data, axis=0, combine_fn=add_fn)
    tl.store(output_ptr + offsets, local_scan, mask=mask)
    
    # 2. 如果需要，将整个 Block 的总和存入中间数组
    if STORE_BLOCK_SUM:
        block_sum = tl.sum(data, axis=0)
        tl.store(block_sums_ptr + pid, block_sum)

@triton.jit
def add_base_kernel(
    output_ptr, block_sums_prefix_ptr, n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Block 0 前面没有元素，不需要加上偏移量基底
    if pid == 0:
        return
    
    # 基底是当前 Block 之前所有 Block 总和的前缀和
    base = tl.load(block_sums_prefix_ptr + pid - 1)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 加载第一阶段的局部结果，累加全局基底，写回全局内存
    local_scan = tl.load(output_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, local_scan + base, mask=mask)

def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    if n <= 0:
        return
        
    # 定义块大小：对于 FP32 数据，16384 个元素占用 64KB Shared Memory。
    # 这能很好地平衡 H100 等现代架构下的 SRAM 限制与 SM Occupancy (并发度)。
    BLOCK_SIZE = 16384 
    grid = int(triton.cdiv(n, BLOCK_SIZE))
    
    # 边界情况：数据足够小，单 Block 即可解决
    if grid == 1:
        B = max(16, triton.next_power_of_2(n))
        local_scan_kernel[(1,)](data, output, data, n, False, BLOCK_SIZE=B)  # type: ignore
        return
        
    # --- Pass 1: 计算局部前缀和，并提取每个 Block 的总和 ---
    block_sums = torch.empty(grid, dtype=torch.float32, device=data.device)
    local_scan_kernel[(grid,)](data, output, block_sums, n, True, BLOCK_SIZE=BLOCK_SIZE)  # type: ignore
    
    # --- Pass 2: 对 Block 总和进行全局前缀和 ---
    # 由于 N 最大 1 亿，Grid 最大约为 6104，可以完全放入单一的 Block 中处理
    block_sums_prefix = torch.empty_like(block_sums)
    B_grid = max(16, triton.next_power_of_2(grid))
    local_scan_kernel[(1,)](block_sums, block_sums_prefix, block_sums, grid, False, BLOCK_SIZE=B_grid)  # type: ignore
    
    # --- Pass 3: 将全局偏移量基底 (Base) 广播累加回各个 Block 中 ---
    add_base_kernel[(grid,)](output, block_sums_prefix, n, BLOCK_SIZE=BLOCK_SIZE)  # type: ignore