import argparse
import math
import sys
from typing import Callable, Dict, Iterable, Tuple
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cuda_impl.vector_add import solve as cuda_solve
from pytorch.vector_add import solve as pytorch_solve
from triton_impl.vector_add import solve as triton_solve


Case = Tuple[str, int]


def build_cases(mode: str) -> Iterable[Case]:
    base_cases = [
        ("scalar_tail_1", 1),
        ("scalar_tail_2", 2),
        ("scalar_tail_3", 3),
        ("basic_small", 4),
        ("warp_plus_one", 33),
        ("non_power_of_two", 257),
        ("medium", 4096),
        ("large", 65535),
    ]
    if mode == "functional":
        return base_cases
    return [("performance_smoke", 25_000_000)]


def make_inputs(size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(size)
    a = torch.empty(size, device="cuda", dtype=torch.float32).uniform_(-100.0, 100.0, generator=gen)
    b = torch.empty(size, device="cuda", dtype=torch.float32).uniform_(-100.0, 100.0, generator=gen)
    expected = torch.add(a, b)
    return a, b, expected


def run_impl(name: str, fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None], size: int) -> Dict[str, float]:
    a, b, expected = make_inputs(size)
    out = torch.empty_like(a)

    fn(a, b, out, size)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(a, b, out, size)
    end.record()
    torch.cuda.synchronize()

    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-5):
        max_diff = (out - expected).abs().max().item()
        raise AssertionError(f"{name} failed for size={size}, max_diff={max_diff}")

    return {"ms": start.elapsed_time(end), "gbps": throughput_gbps(size, start.elapsed_time(end))}


def throughput_gbps(size: int, elapsed_ms: float) -> float:
    if elapsed_ms == 0:
        return math.inf
    bytes_moved = size * 4 * 3
    return bytes_moved / (elapsed_ms / 1000.0) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["functional", "performance"], default="functional")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project.")

    implementations = {
        "pytorch": pytorch_solve,
        "cuda": cuda_solve,
        "triton": triton_solve,
    }

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    for case_name, size in build_cases(args.mode):
        print(f"\n[{case_name}] N={size}")
        for impl_name, impl_fn in implementations.items():
            metrics = run_impl(impl_name, impl_fn, size)
            print(f"  {impl_name:<7} ok  {metrics['ms']:>8.3f} ms  {metrics['gbps']:>8.2f} GB/s")


if __name__ == "__main__":
    main()
