import os
import sys

import torch

# Make fp8_bmm importable as a namespace package (no __init__.py in the repo):
# run as a script (python3 fp8_quant_bench.py) has no parent package context.
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

from fp8_utils import FP8Format  # noqa: E402
from utils import time_consumption  # noqa: E402
from fp8_warper import fp8_quant_triton  # noqa: E402

@torch.compile
def fp8_quant_torch(
    x: torch.Tensor,
    quant_dtype=FP8Format.E4M3.value.dtype_fwd,
    fp8_range=FP8Format.E4M3.value.max_fwd,
    eps=1e-12
):
    x_min, x_max = x.aminmax()
    x_max_abs = torch.maximum(x_min.abs(), x_max.abs())
    x_max_abs = x_max_abs.clamp(min=eps)
    quant_factor = fp8_range / x_max_abs
    reciprocal_factor = quant_factor.reciprocal()
    quanted_x = (x * quant_factor).to(quant_dtype)
    return quanted_x, reciprocal_factor

bench_shapes = [
    (32, 2048, 960),
    (80, 2048, 640),
    (14, 2048, 8192),
]

def do_bench():
    for shape in bench_shapes:
        x = torch.randn(shape, device="cuda")

        torch_time = time_consumption(fp8_quant_torch, x)
        triton_time = time_consumption(fp8_quant_triton, x)
        print(
            f"shape: {shape}, compiled torch time: {torch_time} ms, triton time: {triton_time} ms"
        )

if __name__ == "__main__":
    do_bench()