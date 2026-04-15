import triton.language as tl
import torch
import triton

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = num_pid_k * GROUP_SIZE_M;
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    msk_m = offs_m < M
    msk_k = offs_k < K

    aptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    bptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n_cur = n * BLOCK_SIZE_N + offs_n
        msk_n = offs_n_cur < N

        a = tl.load(aptrs, mask=msk_m[:, None] & msk_n[None, :], other=0.0)
        b = tl.load(bptrs, mask=msk_n[:, None] & msk_k[None, :], other=0.0)

        acc += tl.dot(a, b)

        aptrs += BLOCK_SIZE_N * stride_an
        bptrs += BLOCK_SIZE_N * stride_bn
    
    cptrs = c_ptr + (offs_m[: None] * stride_cm + offs_k[None, :] * stride_ck)

    tl.store(cptrs, acc, mask=msk_m[: None] & msk_k[None, :])

    pass

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_ck = c.stride()

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 32
    GROUP_SIZE_M = 8

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    matrix_multiplication_kernel[grid](
        a, b, c, 
        M, N, K, 
        stride_am, stride_an, 
        stride_bn, stride_bk, 
        stride_cm, stride_ck,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M
    )

if __name__ == "__main__":
    # 测试代码
    M, N, K = 512, 512, 512
    a = torch.randn(M, N, device='cuda')
    b = torch.randn(N, K, device='cuda')
    c = torch.zeros(M, K, device='cuda')

    solve(a, b, c, M, N, K)

    # 验证结果
    print ("验证结果...")
    #print c, a @ b
    print(torch.allclose(c, a @ b, atol=1e-1))
    assert torch.allclose(c, a @ b, atol=1e-1), "结果不正确！"

    print("测试通过！")