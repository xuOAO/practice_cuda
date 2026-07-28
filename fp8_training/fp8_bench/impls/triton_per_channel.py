from __future__ import annotations

from typing import Optional

import torch
import triton

from fp8_bench.kernels.bmm.per_channel import (
    batch_fp8_per_channel_bmm_kernel,
    batch_fp8_per_channel_bmm_kernel_autotuned,
)
from fp8_bench.kernels.quant.per_channel import (
    fp8_per_channel_quant_kernel,
    fp8_per_channel_quant_kernel_autotuned,
)
from fp8_bench.registry import (
    QuantResult,
    register_bmm,
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

def prepare_b_layout(value: QuantResult, layout: str) -> QuantResult:
    if layout == "n":
        return value
    if layout != "k":
        raise ValueError(f"unknown B layout: {layout}")
    return QuantResult(
        tensor=value.tensor.transpose(1, 2).contiguous(),
        inv_scale=value.inv_scale,
        impl=value.impl,
        meta={**value.meta, "layout": "k"},
    )


def logical_b_tensor(value: QuantResult, layout: str) -> torch.Tensor:
    if layout == "n":
        return value.tensor
    return value.tensor.transpose(1, 2)


def _triton_bmm(
    a: QuantResult,
    b: QuantResult,
    *,
    layout: str,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    profile: bool = False,
) -> torch.Tensor:
    if a.tensor.ndim != 3 or b.tensor.ndim != 3:
        raise ValueError("BMM expects 3D quantized tensors")
    batch, m, k = a.tensor.shape
    if layout == "n":
        b_batch, b_k, n = b.tensor.shape
    else:
        b_batch, n, b_k = b.tensor.shape
    if batch != b_batch or k != b_k:
        raise ValueError(f"shape mismatch: A={a.tensor.shape}, B={b.tensor.shape}")

    if out is None:
        out = torch.empty((batch, m, n), device=a.tensor.device, dtype=out_dtype)
    if bias is None:
        bias_ptr = a.tensor
        stride_biasb = stride_biasm = stride_biasn = 0
    else:
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride()

    if layout == "n":
        stride_bb, stride_bk, stride_bn = b.tensor.stride()
    else:
        stride_bb = b.tensor.stride(0)
        stride_bk = b.tensor.stride(2)
        stride_bn = b.tensor.stride(1)

    combined_inv_scale = (
        quant_scale if quant_scale is not None else a.inv_scale * b.inv_scale
    )
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        batch,
    )
    kernel = batch_fp8_bmm_kernel if profile else batch_fp8_bmm_kernel_autotuned
    launch_kwargs = {}
    if profile:
        launch_kwargs = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    kernel[grid](
        a.tensor,
        b.tensor,
        out,
        bias_ptr,
        combined_inv_scale,
        m,
        n,
        k,
        *a.tensor.stride(),
        stride_bb,
        stride_bk,
        stride_bn,
        *out.stride(),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIAS=bias is not None,
        B_N_ORDER=layout == "n",
        **launch_kwargs,
    )
    return out


def triton_per_channel_bmm_n(*args, **kwargs) -> torch.Tensor:
    return _triton_bmm(*args, layout="n", **kwargs)


# def triton_per_tensor_bmm_k(*args, **kwargs) -> torch.Tensor:
#     return _triton_bmm(*args, layout="k", **kwargs)


register_quant(
    "triton_per_channel",
    triton_per_channel_quant,
    "per-channel scale quantization; supports E4M3 and E5M2.",
)
# register_bmm(
#     "triton_per_tensor_n",
#     triton_per_tensor_bmm_n,
#     quant_impl="triton_per_tensor",
#     layout="n",
#     prepare_b=lambda value: prepare_b_layout(value, "n"),
#     logical_b=lambda value: logical_b_tensor(value, "n"),
#     description="Migrated FP8 BMM with contiguous [B,K,N] right operand.",
# )
# register_bmm(
#     "triton_per_tensor_k",
#     triton_per_tensor_bmm_k,
#     quant_impl="triton_per_tensor",
#     layout="k",
#     prepare_b=lambda value: prepare_b_layout(value, "k"),
#     logical_b=lambda value: logical_b_tensor(value, "k"),
#     description="Migrated FP8 BMM with contiguous [B,N,K] right operand.",
# )
