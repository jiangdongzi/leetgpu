import torch
import triton
import triton.language as tl

@triton.jit
def block_count(A, N, block_counts, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = BLOCK_SIZE*pid + tl.arange(0, BLOCK_SIZE)

    local_input = tl.load(A+offsets, offsets<N, other=0.)
    local_count = tl.sum(tl.where(local_input>0., 1, 0), axis=0)
    
    tl.store(block_counts + pid, local_count)

@triton.jit
def compact(A, N, out, block_offsets, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = BLOCK_SIZE*pid + tl.arange(0, BLOCK_SIZE)

    local_A = tl.load(A+offsets, offsets<N)

    block_offset = tl.load(block_offsets+pid)

    in_set = local_A > 0.
    out_offsets = tl.cumsum(in_set.to(tl.int32)) - in_set.to(tl.int32)

    tl.store(out+block_offset+out_offsets, local_A, in_set & (offsets < N))



# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE), )

    block_counts = torch.zeros(triton.cdiv(N,BLOCK_SIZE), device=A.device, dtype=torch.int32)
    
    block_count[grid](A, N, block_counts, BLOCK_SIZE)

    block_offsets = torch.cumsum(block_counts, dim=0) - block_counts

    compact[grid](A, N, out, block_offsets, BLOCK_SIZE)