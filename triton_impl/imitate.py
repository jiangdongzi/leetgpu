import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    n_pixels = width * height
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 4
    offs = tl.arange(0, BLOCK_SIZE) * 4
    ir = start_idx + offs + image
    ig = start_idx + offs + 1 + image
    ib = start_idx + offs + 2 + image
    r = tl.load(ir, mask=(ir < n_pixels))
    g = tl.load(ig, mask=(ig < n_pixels))
    b = tl.load(ib, mask=(ib < n_pixels))
    r = 255 - r
    g = 255 - g
    b = 255 - b

    tl.store(ir, r, mask=(ir < n_pixels))
    tl.store(ig, g, mask=(ig < n_pixels))
    tl.store(ib, b, mask=(ib < n_pixels))
    pass

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    # 计算需要的 grid 数量，cdiv 向上取整以覆盖所有像素
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)

if __name__ == "__main__":
    # Example usage
    width, height = 1024, 1024
    image = torch.randint(0, 256, (height, width, 4), dtype=torch.uint8, device='cuda')

    solve(image, width, height)

    # Verify correctness
    expected = 255 - image
    assert torch.allclose(image[..., :3], expected[..., :3]), "Color inversion failed!"