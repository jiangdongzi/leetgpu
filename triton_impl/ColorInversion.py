import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    # 1. 计算图像的总像素数量
    n_pixels = width * height
    
    # 2. 获取当前 block 的 program_id
    pid = tl.program_id(axis=0)
    
    # 3. 计算当前 block 处理的像素的全局起始索引
    block_start = pid * BLOCK_SIZE
    
    # 4. 生成当前 block 内所有像素的偏移量 (0 到 BLOCK_SIZE - 1)
    pixel_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 5. 创建掩码(mask)，防止当 n_pixels 不能被 BLOCK_SIZE 整除时发生越界访问
    mask = pixel_offsets < n_pixels
    
    # 6. 计算 R, G, B 分量的内存指针偏移
    # 每个像素由 RGBA 4 个元素组成，所以基地址偏移需要乘以 4
    base_ptrs = image + pixel_offsets * 4
    
    r_ptrs = base_ptrs + 0  # Red
    g_ptrs = base_ptrs + 1  # Green
    b_ptrs = base_ptrs + 2  # Blue
    # A_ptrs = base_ptrs + 3 # Alpha 不需要修改，所以不获取它的指针
    
    # 7. 从全局内存中加载 RGB 的值
    r = tl.load(r_ptrs, mask=mask)
    g = tl.load(g_ptrs, mask=mask)
    b = tl.load(b_ptrs, mask=mask)
    
    # 8. 执行颜色反转计算
    r_inv = 255 - r
    g_inv = 255 - g
    b_inv = 255 - b
    
    # 9. 将计算后的结果写回原数组（原地修改）
    tl.store(r_ptrs, r_inv, mask=mask)
    tl.store(g_ptrs, g_inv, mask=mask)
    tl.store(b_ptrs, b_inv, mask=mask)

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