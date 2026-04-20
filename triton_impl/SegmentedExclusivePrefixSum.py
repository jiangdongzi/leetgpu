import torch
import triton
import triton.language as tl

# =====================================================================
# 自定义关联扫描算子 (Associative Scan Operator)
# =====================================================================
@triton.jit
def seg_scan_op(a_sum, a_flag, b_sum, b_flag):
    # 如果右侧 (b) 包含了新段的起点 (b_flag > 0)，
    # 则左侧 (a) 的累加和不再传入右侧，右侧和保持不变。
    # 否则，段落延续，左侧和累加到右侧。
    c_flag = a_flag | b_flag
    c_sum = tl.where(b_flag > 0, b_sum, a_sum + b_sum)
    return c_sum, c_flag

# =====================================================================
# 核心 Kernel 1：局部扫描并提取 Block Summary
# =====================================================================
@triton.jit
def local_scan_kernel(
    vals_ptr, flags_ptr,
    out_sums_ptr, out_flags_ptr,
    block_sums_ptr, block_flags_ptr,
    HAS_BLOCK_SUMS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 加载当前 Block 的数据，将 flags 转为 int32 方便操作
    vals = tl.load(vals_ptr + offsets, mask=mask, other=0.0)
    flags = tl.load(flags_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # 在 Block 内部执行并行扫描
    loc_sums, loc_flags = tl.associative_scan((vals, flags), 0, seg_scan_op)

    # 存回局部计算结果
    tl.store(out_sums_ptr + offsets, loc_sums, mask=mask)
    tl.store(out_flags_ptr + offsets, loc_flags, mask=mask)

    # 提取并保存 Block 末尾的元素作为 summary
    if HAS_BLOCK_SUMS:
        last_idx = tl.minimum(BLOCK_SIZE, N - pid * BLOCK_SIZE) - 1
        store_mask = (tl.arange(0, BLOCK_SIZE) == last_idx) & mask
        
        # 将局部扫描的最后有效元素映射存入 block summaries
        ptrs_sums = block_sums_ptr + pid + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        ptrs_flags = block_flags_ptr + pid + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        tl.store(ptrs_sums, loc_sums, mask=store_mask)
        tl.store(ptrs_flags, loc_flags, mask=store_mask)

# =====================================================================
# 核心 Kernel 2：将 Block Summary 添加回局部结果中
# =====================================================================
@triton.jit
def add_block_sums_kernel(
    sums_ptr, flags_ptr, block_sums_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid == 0:
        return  # 第 0 个 block 前面没有内容，无需累加

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    loc_sums = tl.load(sums_ptr + offsets, mask=mask, other=0.0)
    loc_flags = tl.load(flags_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # 取出前一个 Block 累加完成的全局总和
    g_prev_sum = tl.load(block_sums_ptr + pid - 1)

    # 如果当前元素及之前没有新段落标记(loc_flags == 0)，才加上前一个 block 的遗留总和
    final_sums = loc_sums + tl.where(loc_flags > 0, 0.0, g_prev_sum)

    tl.store(sums_ptr + offsets, final_sums, mask=mask)

# =====================================================================
# 核心 Kernel 3：排他性转换 (Exclusive = Inclusive - values)
# =====================================================================
@triton.jit
def final_exclusive_kernel(
    inclusive_sums_ptr, values_ptr, output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    inc_sums = tl.load(inclusive_sums_ptr + offsets, mask=mask, other=0.0)
    vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)

    # 包含性变排他性：减去它自身原本的数值
    exc_sums = inc_sums - vals
    tl.store(output_ptr + offsets, exc_sums, mask=mask)

# =====================================================================
# 调度与递归入口
# =====================================================================
def compute_inclusive_segmented_scan(vals, flgs, out_sums, out_flgs, N):
    BLOCK_SIZE = 4096
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 递归终止条件：当数组可以放入一个 Block 时，一次扫描即可结束
    if num_blocks == 1:
        local_scan_kernel[(1,)](
            vals, flgs, out_sums, out_flgs,
            None, None,
            HAS_BLOCK_SUMS=False,
            N=N, BLOCK_SIZE=BLOCK_SIZE
        )
        return

    # 分配 Summary 数组来记录每个 Block 的末尾累加状态
    b_sums = torch.empty(num_blocks, dtype=torch.float32, device=vals.device)
    b_flgs = torch.empty(num_blocks, dtype=torch.int32, device=vals.device)

    # 1. 局部扫描并导出 Summary
    local_scan_kernel[(num_blocks,)](
        vals, flgs, out_sums, out_flgs,
        b_sums, b_flgs,
        HAS_BLOCK_SUMS=True,
        N=N, BLOCK_SIZE=BLOCK_SIZE
    )

    # 2. 递归：对 Block Summary 执行 In-place Inclusive Scan
    # 比如 N=50,000,000 时，第二层的元素只有 12,208 个，大概经过 3 层即可收敛
    compute_inclusive_segmented_scan(b_sums, b_flgs, b_sums, b_flgs, num_blocks)

    # 3. 将扫描后的全局 Summary 添加回局部结果中
    add_block_sums_kernel[(num_blocks,)](
        out_sums, out_flgs, b_sums,
        N=N, BLOCK_SIZE=BLOCK_SIZE
    )


# values, flags, output are tensors on the GPU
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    if N <= 0:
        return

    # 分配一个临时张量存放中间扫描出来的 cumulative flags
    inc_flgs = torch.empty(N, dtype=torch.int32, device=flags.device)

    # 首先将包含性分段扫描 (Inclusive Segmented Scan) 结果原地写入 output
    compute_inclusive_segmented_scan(values, flags, output, inc_flgs, N)

    # 最后执行一次 O(N) 的 Kernel 将 Inclusive 转化为题目要求的 Exclusive
    BLOCK_SIZE = 4096
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    final_exclusive_kernel[(num_blocks,)](
        output, values, output,
        N=N, BLOCK_SIZE=BLOCK_SIZE
    )