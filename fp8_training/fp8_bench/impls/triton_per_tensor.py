from __future__ import annotations

from typing import Optional

import torch
import triton

from fp8_bench.kernels.bmm.per_tensor import (
    batch_fp8_per_tensor_bmm_kernel,
    batch_fp8_per_tensor_bmm_kernel_autotuned,
)
from fp8_bench.kernels.quant.per_tensor import (
    fp8_per_tensor_quant_kernel,
    fp8_per_tensor_quant_kernel_autotuned,
)
from fp8_bench.registry import (
    QuantResult,
    register_bmm,
    register_quant,
)

def triton_per_tensor_quant(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
    profile: bool = False,
) -> QuantResult:
    assert x.ndim == 3, "BMM expects 3D quantized tensors"

    x_min, x_max = x.aminmax()
    max_abs = torch.maximum(x_min.abs(), x_max.abs()).clamp(min=eps)
    fp8_max = torch.finfo(fp8_dtype).max
    quant_scale = fp8_max / max_abs
    inv_scale = quant_scale.reciprocal()
    output = torch.empty_like(x, dtype=fp8_dtype)

    batch, m, n = x.shape
    stride_xb = x.stride(0)
    stride_yb = output.stride(0)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        batch,
    )
    kernel = fp8_per_tensor_quant_kernel if profile else fp8_per_tensor_quant_kernel_autotuned
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
        quant_scale,
        fp8_max=fp8_max,
        dim=x.ndim,
        **launch_kwargs,
    )
    return QuantResult(
        tensor=output,
        inv_scale=inv_scale,
        impl="triton_per_tensor",
        meta={"logical_shape": tuple(x.shape), "fp8_dtype": str(fp8_dtype)},
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


def triton_per_tensor_bmm(
    a: QuantResult,
    b: QuantResult,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    do_transpose_b: bool = False,
    profile: bool = False,
) -> torch.Tensor:
    if a.tensor.ndim != 3 or b.tensor.ndim != 3:
        raise ValueError("BMM expects 3D quantized tensors")
    batch, m, k = a.tensor.shape

    layout = "n-major" if b.tensor.stride(1) > b.tensor.stride(2) else "k-major"
    assert not do_transpose_b or layout == "n-major", "do_transpose_b is only supported for layout 'n-major'"

    b_batch, b_k, n = b.tensor.shape

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

    if do_transpose_b:
        b.tensor = b.tensor.transpose(1, 2).contiguous().transpose(1, 2)

    dequant_scale = a.dequant_scale * b.dequant_scale

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        batch,
    )
    kernel = batch_fp8_per_tensor_bmm_kernel if profile else batch_fp8_per_tensor_bmm_kernel_autotuned
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
        dequant_scale,
        m,
        n,
        k,
        *a.tensor.stride(),
        *b.tensor.stride(),
        *out.stride(),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIAS=bias is not None,
        B_N_ORDER=layout == "n",
        **launch_kwargs,
    )
    return out

register_quant(
    "triton_per_tensor",
    triton_per_tensor_quant,
    "Migrated per-tensor scale quantization; supports E4M3 and E5M2.",
)
register_bmm(
    "triton_per_tensor_n",
    triton_per_tensor_bmm,
    quant_impl="triton_per_tensor",
    layout="n",
    prepare_b=lambda value: prepare_b_layout(value, "n"),
    logical_b=lambda value: logical_b_tensor(value, "n"),
    description="Migrated FP8 BMM with contiguous [B,K,N] right operand.",
)
register_bmm(
    "triton_per_tensor_k",
    triton_per_tensor_bmm,
    quant_impl="triton_per_tensor",
    layout="k",
    prepare_b=lambda value: prepare_b_layout(value, "k"),
    logical_b=lambda value: logical_b_tensor(value, "k"),
    description="Migrated FP8 BMM with contiguous [B,N,K] right operand.",
)
