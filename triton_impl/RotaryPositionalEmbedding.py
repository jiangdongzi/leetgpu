import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    # 指针变量
    q_ptr, cos_ptr, sin_ptr, out_ptr,
    # Stride（步长）变量，用于在 1D 内存中定位 2D 矩阵的元素
    stride_qm, stride_qd,
    stride_cm, stride_cd,
    stride_sm, stride_sd,
    stride_om, stride_od,
    # 编译期常量
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 映射视角：启动的每一个 Program（线程块）专门负责处理矩阵中的一行（也就是一个 Token）
    pid_m = tl.program_id(axis=0)

    # 2. 指针偏移：定位到当前负责的这一行 (Row pid_m) 的起始内存地址
    q_row_ptr = q_ptr + pid_m * stride_qm
    c_row_ptr = cos_ptr + pid_m * stride_cm
    s_row_ptr = sin_ptr + pid_m * stride_sm
    o_row_ptr = out_ptr + pid_m * stride_om

    # 3. 向量化索引生成：生成从 0 到 D/2 - 1 的索引
    half_D = D // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    # 掩码 mask：防止当 D/2 不是 2 的幂次时产生内存越界访问
    mask = offsets < half_D

    # 4. 极致优化的精髓：零内存搬运实现 rotate_half
    # 我们不去做真实的数组倒序或交换，而是直接通过改变加载地址，把前半段(x1)和后半段(x2)读进寄存器
    x1 = tl.load(q_row_ptr + offsets * stride_qd, mask=mask)
    x2 = tl.load(q_row_ptr + (offsets + half_D) * stride_qd, mask=mask)

    # 同理，加载 cos 和 sin 的前后两半部分
    c1 = tl.load(c_row_ptr + offsets * stride_cd, mask=mask)
    c2 = tl.load(c_row_ptr + (offsets + half_D) * stride_cd, mask=mask)
    s1 = tl.load(s_row_ptr + offsets * stride_sd, mask=mask)
    s2 = tl.load(s_row_ptr + (offsets + half_D) * stride_sd, mask=mask)

    # 5. 核心融合计算：在寄存器中直接执行 FMA (乘加指令)
    # 根据公式 RoPE(x) = x * cos + rotate_half(x) * sin 展开：
    # 旋转后的前半部分：x1 * c1 - x2 * s1
    # 旋转后的后半部分：x2 * c2 + x1 * s2
    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2

    # 6. 写回显存：将计算结果放回 output 张量对应的位置
    tl.store(o_row_ptr + offsets * stride_od, out1, mask=mask)
    tl.store(o_row_ptr + (offsets + half_D) * stride_od, out2, mask=mask)


def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    # 【GPU 调度策略】
    # 我们采用最直接的 1D Grid 划分方式：启动 M 个 Program 实例。
    # 因为 M = 1,048,576（约一百万），GPU 的硬件调度器（Hardware Scheduler）
    # 会非常高效地把这一百万个任务分发给所有可用的 SM（流多处理器）并发执行。
    grid = (M,)
    
    # 【对齐约束】
    # Triton 要求使用 tl.arange 时，大小必须是 2 的整数次幂。
    # 由于我们在 Kernel 中是一次性处理 D/2 个元素，所以需要计算向上取整的下一个 2 的幂。
    # 题目说明中 D=128，所以 D//2=64，本身就是 2 的幂，但这样写保证了 D 变动时的泛化性。
    BLOCK_SIZE = triton.next_power_of_2(D // 2)

    # 【启动 Kernel】
    rope_kernel[grid](
        Q, cos, sin, output,
        Q.stride(0), Q.stride(1),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        output.stride(0), output.stride(1),
        D=D,
        BLOCK_SIZE=BLOCK_SIZE
    )