import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr, 
    rows, cols, 
    stride_ir, stride_ic, 
    stride_or, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1. Identify which block (tile) this program instance is handling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 1D arrays of indices for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2. Setup 2D input indices and load the tile from DRAM to SRAM
    rm_in = rm[:, None]  # Column vector
    rn_in = rn[None, :]  # Row vector
    
    in_ptrs = input_ptr + (rm_in * stride_ir + rn_in * stride_ic)
    mask_in = (rm_in < rows) & (rn_in < cols)
    
    # Load data (coalesced read)
    x = tl.load(in_ptrs, mask=mask_in, other=0.0)

    # 3. Transpose the block in fast SRAM
    # If x is shape (BLOCK_M, BLOCK_N), x_t becomes (BLOCK_N, BLOCK_M)
    x_t = tl.trans(x)

    # 4. Setup 2D output indices for the transposed block
    rm_out = rn[:, None] # Note how rn is now the column vector (rows of output)
    rn_out = rm[None, :] # rm is now the row vector (cols of output)
    
    out_ptrs = output_ptr + (rm_out * stride_or + rn_out * stride_oc)
    mask_out = (rm_out < cols) & (rn_out < rows)

    # Store data (coalesced write)
    tl.store(out_ptrs, x_t, mask=mask_out)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    # Tile sizes (64x64 is highly optimized for memory coalescing on standard GPUs)
    BLOCK_M = 64
    BLOCK_N = 64

    # The grid defines how many blocks we launch, not individual elements
    grid = lambda meta: (
        triton.cdiv(rows, meta['BLOCK_M']),
        triton.cdiv(cols, meta['BLOCK_N'])
    )

    matrix_transpose_kernel[grid](
        input, output, 
        rows, cols, 
        stride_ir, stride_ic, 
        stride_or, stride_oc,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

if __name__ == "__main__":
    # Example usage
    rows, cols = 1024, 1024
    input = torch.randn(rows, cols, device='cuda')
    output = torch.empty(cols, rows, device='cuda')  # Transposed shape

    solve(input, output, rows, cols)

    # Verify correctness
    assert torch.allclose(output, input.t()), "Transpose failed!"