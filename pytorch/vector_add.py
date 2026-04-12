import torch


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    torch.add(a[:n], b[:n], out=c[:n])
