import ctypes
from pathlib import Path

import torch


_LIB = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "cuda" / "vector_add.so"))
_LIB.solve.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_LIB.solve.restype = None


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    _LIB.solve(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_size_t(n),
    )
