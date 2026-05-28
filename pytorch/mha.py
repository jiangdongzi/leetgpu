import torch
import torch.nn.functional as F

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          N: int, d_model: int, h: int):
      # N x d
      N, _ = Q.shape
      head_dim = d_model // h
      # split into heads: (h, N, head_dim)
      Q = Q.view(N, h, head_dim).transpose(0, 1)
      K = K.view(N, h, head_dim).transpose(0, 1)
      V = V.view(N, h, head_dim).transpose(0, 1)

      # scaled dot product
      scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (h, N, N)
      attn = F.softmax(scores, dim=-1)
      res = torch.matmul(attn, V)  # (h, N, head_dim)

      # combine heads back: (N, d_model)
      back = res.transpose(0, 1).contiguous().view(N, d_model)
      output.copy_(back)
      
    