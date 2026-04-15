import torch
import triton
import triton.language as tl

@triton.jit
def reduce_sum_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    num_jobs, 
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前线程块的 ID
    pid = tl.program_id(axis=0)
    
    # 局部累加器
    acc = 0.0
    
    # 计算当前块处理的起始索引和步长
    start_idx = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    
    # Grid-Stride Loop: 即使 N 很大，也能用有限的 block 处理完所有数据
    for i in range(start_idx, n_elements, step):
        # 计算偏移量并生成掩码以防越界
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # 加载数据，越界部分填充为 0.0
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # 块内局部规约求和
        acc += tl.sum(x, axis=0)
        
    # 利用原子加法将每个 block 的局部总和汇总到全局 output
    tl.atomic_add(output_ptr, acc)

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # 确保输出张量被初始化为 0，因为我们将使用 atomic_add 累加
    output.zero_()
    
    # 定义每个 block 一次处理的元素数量
    BLOCK_SIZE = 4096
    
    # 计算需要的 block 数量。
    # 为了避免数万个 block 带来的 atomic_add 竞争冲突，将最大并发任务数限制在 1024。
    grid_size = min(triton.cdiv(N, BLOCK_SIZE), 1024)
    
    # 启动 Triton Kernel
    reduce_sum_kernel[(grid_size,)](
        input, 
        output, 
        N, 
        grid_size,
        BLOCK_SIZE=BLOCK_SIZE
    )