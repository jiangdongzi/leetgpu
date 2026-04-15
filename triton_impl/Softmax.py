import torch
import triton
import triton.language as tl

# ==========================================
# Kernel 1: 寻找每个 Block 负责区域的局部最大值
# ==========================================
@triton.jit
def get_local_max_kernel(input, max_workspace, N, BLOCK_SIZE: tl.constexpr):
    # 1. 强制类型转换：把传入的内存地址转为 float32 的指针
    input_ptr = input.to(tl.pointer_type(tl.float32))
    max_workspace_ptr = max_workspace.to(tl.pointer_type(tl.float32))
    
    # 2. 获取当前是第几个 Block (快递员编号)
    pid = tl.program_id(0)
    
    # 3. 计算当前 Block 需要读取的门牌号 (Offsets)
    # tl.arange(0, 4096) 会生成一个 [0, 1, ..., 4095] 的向量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 4. 生成安全掩码，防止越界
    mask = offsets < N
    
    # 5. 从显存读取数据。如果越界(mask为False)，用极小值 -inf 填充，这样求最大值时就不会受影响
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # 6. 计算当前 4096 个数据中的最大值，并写回到 workspace 的对应格子里
    # 因为 pid 是 0, 1, 2... 所以每个 Block 刚好写自己的那个格子
    tl.store(max_workspace_ptr + pid, tl.max(x))


# ==========================================
# Kernel 2: 求全局最大值，并计算各 Block 的局部指数和
# ==========================================
@triton.jit
def get_local_sum_kernel(
    input, max_workspace, sum_workspace, 
    num_blocks, N, 
    BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr
):
    input_ptr = input.to(tl.pointer_type(tl.float32))
    max_workspace_ptr = max_workspace.to(tl.pointer_type(tl.float32))
    sum_workspace_ptr = sum_workspace.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    
    # --- 神仙操作：跨 Block 通信 ---
    # 此时 max_workspace 里面存了 123 个局部最大值。
    # 我们让 *每一个* Block 都把这 123 个数字读进来自己求最大值，这样大家都知道了全局最大值是多少。
    blocks_offsets = tl.arange(0, MAX_BLOCKS)
    blocks_mask = blocks_offsets < num_blocks
    max_vals = tl.load(max_workspace_ptr + blocks_offsets, mask=blocks_mask, other=-float('inf'))
    global_max = tl.max(max_vals) # 这里就拿到了 500,000 个数中的真正全局最大值
    
    # 继续处理自己的 4096 个数据
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # 应用 Max Trick: e^(x - max)
    num = tl.exp(x - global_max)
    
    # 计算局部指数累加和。越界的地方设为 0.0，不影响加法
    local_sum = tl.sum(tl.where(mask, num, 0.0))
    
    # 把自己算出的局部和存入 sum_workspace
    tl.store(sum_workspace_ptr + pid, local_sum)


# ==========================================
# Kernel 3: 求全局指数和，并计算最终 Softmax 结果
# ==========================================
@triton.jit
def compute_softmax_kernel(
    input, max_workspace, sum_workspace, output,
    num_blocks, N, 
    BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr
):
    input_ptr = input.to(tl.pointer_type(tl.float32))
    max_workspace_ptr = max_workspace.to(tl.pointer_type(tl.float32))
    sum_workspace_ptr = sum_workspace.to(tl.pointer_type(tl.float32))
    output_ptr = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    
    # 同样的思路，读取所有的局部最大值和局部指数和
    blocks_offsets = tl.arange(0, MAX_BLOCKS)
    blocks_mask = blocks_offsets < num_blocks
    
    # 复原全局最大值
    max_vals = tl.load(max_workspace_ptr + blocks_offsets, mask=blocks_mask, other=-float('inf'))
    global_max = tl.max(max_vals)
    
    # 把所有 Block 算出的 sum 加起来，得到全局的分母
    sum_vals = tl.load(sum_workspace_ptr + blocks_offsets, mask=blocks_mask, other=0.0)
    global_sum = tl.sum(sum_vals)
    
    # 最后一次读取自己的数据
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # 计算真正的概率值 out = e^(x - max) / sum(e^(x - max))
    out = tl.exp(x - global_max) / global_sum
    
    # 将结果写回内存的 output 数组中
    tl.store(output_ptr + offsets, out, mask=mask)


# ==========================================
# Host 端代码 (运行在 CPU 上，负责发号施令)
# ==========================================
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 4096
    # MAX_BLOCKS 是一个编译时常量(constexpr)，它必须是 2 的幂（Triton 要求 tl.arange 的大小是 2 的幂）。
    # 256 * 4096 = 1,048,576，完全够装 N=500,000 的数据了。
    MAX_BLOCKS = 256 
    
    # triton.cdiv 是向上取整除法。比如 500,000 / 4096 ≈ 122.07，会向上取整为 123 个 Block
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    
    # Workspace 是内存中开辟的极小暂存区，长度为 123，用来做跨 Block 的消息传递
    max_workspace = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    sum_workspace = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    
    # 依次发射三个 Kernel，由于它们处于同一个 Stream 中，GPU 会按顺序依次执行
    get_local_max_kernel[grid](input, max_workspace, N, BLOCK_SIZE)
    get_local_sum_kernel[grid](input, max_workspace, sum_workspace, num_blocks, N, BLOCK_SIZE, MAX_BLOCKS)
    compute_softmax_kernel[grid](input, max_workspace, sum_workspace, output, num_blocks, N, BLOCK_SIZE, MAX_BLOCKS)