import torch
import triton
import triton.language as tl

@triton.jit
def matrix_copy_kernel(a, b, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    msk = offs < N * N
    inptr = a + offs
    outptr = b + offs
    x = tl.load(inptr, mask=msk)
    tl.store(outptr, x, mask=msk)
    pass

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (tl.cdiv(N * N, BLOCK_SIZE),)
    matrix_copy_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)
    pass

if __name__ == "__main__":
    N = 1024
    a = torch.randn(N, N, device='cuda')
    b = torch.empty_like(a)
    solve(a, b, N)
    assert torch.allclose(a, b)