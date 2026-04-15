import torch
import torch.nn as nn

# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    # 禁用梯度计算，加速推理并减少显存消耗
    with torch.no_grad():
        # model(input) 计算前向传播
        # .copy_() 将计算结果原地写入到传入的 output 张量内存中
        output.copy_(model(input))