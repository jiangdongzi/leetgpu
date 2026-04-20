import torch
import triton
import triton.language as tl

@triton.jit
def nearest_neighbor_kernel(
    points_ptr,
    indices_ptr,
    N,
    stride_pn,
    stride_pd,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 1. 获取当前 Block 处理的 Query 索引
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # 扩展为 2D shape [BLOCK_SIZE_M, 1] 以便后续 Broadcast
    offs_m_2d = tl.expand_dims(offs_m, 1)
    mask_m_2d = offs_m_2d < N
    
    # 2. 从 HBM 加载 Query 点坐标
    # stride_pn 控制在点(N)之间跳转，stride_pd 控制在 xyz(3) 之间跳转
    q_x = tl.load(points_ptr + offs_m_2d * stride_pn + 0 * stride_pd, mask=mask_m_2d, other=0.0)
    q_y = tl.load(points_ptr + offs_m_2d * stride_pn + 1 * stride_pd, mask=mask_m_2d, other=0.0)
    q_z = tl.load(points_ptr + offs_m_2d * stride_pn + 2 * stride_pd, mask=mask_m_2d, other=0.0)
    
    # 3. 初始化全局最小距离和对应索引，Shape: [BLOCK_SIZE_M]
    min_dist = tl.full([BLOCK_SIZE_M], float('inf'), dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE_M], -1, dtype=tl.int32)
    
    # 4. 内层循环：分块遍历所有的 Target 点
    for start_n in range(0, N, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_n_2d = tl.expand_dims(offs_n, 0) # Shape: [1, BLOCK_SIZE_N]
        mask_n_2d = offs_n_2d < N
        
        # 加载 Target 点坐标
        t_x = tl.load(points_ptr + offs_n_2d * stride_pn + 0 * stride_pd, mask=mask_n_2d, other=0.0)
        t_y = tl.load(points_ptr + offs_n_2d * stride_pn + 1 * stride_pd, mask=mask_n_2d, other=0.0)
        t_z = tl.load(points_ptr + offs_n_2d * stride_pn + 2 * stride_pd, mask=mask_n_2d, other=0.0)
        
        # 计算欧式距离的平方
        # [BLOCK_SIZE_M, 1] 减去 [1, BLOCK_SIZE_N] 会 Broadcast 产生 [BLOCK_SIZE_M, BLOCK_SIZE_N] 的矩阵
        dx = q_x - t_x
        dy = q_y - t_y
        dz = q_z - t_z
        dist = dx * dx + dy * dy + dz * dz
        
        # 屏蔽自身点 (i == j) 的比较，以及越界的位置
        is_self = offs_m_2d == offs_n_2d
        dist = tl.where(is_self, float('inf'), dist)
        dist = tl.where(mask_n_2d, dist, float('inf'))
        
        # 5. 在当前 Block 内寻找最小值和其局部索引
        block_min_dist = tl.min(dist, axis=1)
        block_min_idx_local = tl.argmin(dist, axis=1)
        
        # 转换为全局索引
        block_min_idx = start_n + block_min_idx_local
        
        # 6. 更新全局最小值
        update_mask = block_min_dist < min_dist
        min_dist = tl.where(update_mask, block_min_dist, min_dist)
        min_idx = tl.where(update_mask, block_min_idx, min_idx)
    
    # 7. 写回 HBM 显存
    mask_m = offs_m < N
    tl.store(indices_ptr + offs_m, min_idx, mask=mask_m)


def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    # 修复：评测机传入的 points 是 1D Tensor，形如 [x0, y0, z0, x1, y1, z1...]
    # 所以直接硬编码 AoS 的跨步大小
    stride_pn = 3  # 点与点之间的跨步
    stride_pd = 1  # 坐标轴(xyz)之间的跨步
    
    # Block 参数调优策略
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    
    # 一维 Grid 拆分
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_M']), )
    
    nearest_neighbor_kernel[grid](
        points,
        indices,
        N,
        stride_pn,
        stride_pd,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )