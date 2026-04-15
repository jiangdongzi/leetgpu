import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. Identify the range of elements this specific program instance handles
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2. Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # 3. Load the data from DRAM to SRAM
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 4. Compute SiLU: x * sigmoid(x)
    # Note: sigmoid(x) = 1 / (1 + exp(-x))
    # Triton provides tl.sigmoid or you can use 1.0 / (1.0 + tl.exp(-x))
    res = x * tl.sigmoid(x)
    
    # 5. Store the result back to DRAM
    tl.store(output_ptr + offsets, res, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    # The grid is a 1D launch grid covering all elements
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    silu_kernel[grid](
        input, 
        output, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )