from __future__ import annotations

import torch
import triton

from fp8_bench.kernels.quant.per_channel import (
    fp8_per_channel_quant_kernel,
    fp8_per_channel_quant_kernel_autotuned,
)
from fp8_bench.registry import (
    QuantResult,
    register_quant,
)


def triton_per_channel_quant(
    x: torch.Tensor,
    *,
    channel_axis: int = -1,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
    profile: bool = False,
) -> QuantResult:
    assert x.ndim in {2, 3}, f"quant input must be 2D or 3D, got shape={tuple(x.shape)}"
    assert channel_axis in {-1, -2}, "channel_axis must be -1 or -2"

    X_N_MAJOR = x.stride(-1) == 1
    Y_N_MAJOR = True  # output is always contiguous    
    output = torch.empty_like(x, dtype=fp8_dtype)
    if x.ndim == 2:
        m, n = x.shape
        batch = 1
        stride_xb = stride_yb = 0
    else:
        batch, m, n = x.shape
        stride_xb = x.stride(0)
        stride_yb = output.stride(0)
    grid = lambda meta: (
        batch,
        triton.cdiv(m, meta["BLOCK_M"]) if channel_axis == -2 else triton.cdiv(n, meta["BLOCK_N"])
    )
    dequant_scale = torch.empty((batch, m if channel_axis == -2 else n), device=x.device, dtype=torch.float32)

    kernel = fp8_per_channel_quant_kernel if profile else fp8_per_channel_quant_kernel_autotuned
    launch_kwargs = {}
    if profile:
        launch_kwargs = {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "num_warps": 4,
            "num_stages": 3,
        }
    kernel[grid](
        x,
        output,
        m,
        n,
        stride_xb,
        x.stride(-2),
        x.stride(-1),
        stride_yb,
        output.stride(-2),
        output.stride(-1),
        dequant_scale,
        dequant_scale.stride(0),
        dequant_scale.stride(1),
        channel_axis=channel_axis,
        X_N_MAJOR=X_N_MAJOR,
        Y_N_MAJOR=Y_N_MAJOR,
        fp8_max=torch.finfo(fp8_dtype).max,
        dim=x.ndim,
        **launch_kwargs,
    )
    return QuantResult(
        tensor=output,
        dequant_scale=dequant_scale,
        impl="triton_per_channel",
        meta={
            "logical_shape": tuple(x.shape),
            "fp8_dtype": str(fp8_dtype),
            "channel_axis": channel_axis,
        },
    )


register_quant(
    "triton_per_channel",
    triton_per_channel_quant,
    "per-channel scale quantization; supports E4M3 and E5M2.",
)
