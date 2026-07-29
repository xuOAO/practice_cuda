from __future__ import annotations

from functools import partial
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
    if x.ndim not in {2, 3}:
        raise ValueError(
            f"per-tensor quant expects a 2D or 3D tensor, got shape={tuple(x.shape)}"
        )

    x_min, x_max = x.aminmax()
    max_abs = torch.maximum(x_min.abs(), x_max.abs()).float().clamp(min=eps)
    fp8_max = torch.finfo(fp8_dtype).max
    quant_scale = fp8_max / max_abs
    dequant_scale = quant_scale.reciprocal()
    output = torch.empty_like(x, dtype=fp8_dtype)

    if x.ndim == 2:
        batch = 1
        m, n = x.shape
        stride_xb = stride_yb = 0
    else:
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
        dequant_scale=dequant_scale,
        impl="triton_per_tensor",
        granularity="tensor",
        meta={"logical_shape": tuple(x.shape), "fp8_dtype": str(fp8_dtype)},
    )


def prepare_b_layout(value: QuantResult, layout: str) -> QuantResult:
    if layout == "n":
        return value
    if layout != "k":
        raise ValueError(f"unknown B layout: {layout}")
    return QuantResult(
        tensor=value.tensor.transpose(-1, -2).contiguous().transpose(-1, -2),
        dequant_scale=value.dequant_scale,
        impl=value.impl,
        granularity=value.granularity,
        meta={**value.meta, "layout": "k"},
    )


def _b_layout(tensor: torch.Tensor) -> str:
    if tensor.stride(-1) == 1:
        return "n-major"
    if tensor.stride(-2) == 1:
        return "k-major"
    raise ValueError(
        "B must be contiguous along either N or K; "
        f"shape={tuple(tensor.shape)}, strides={tensor.stride()}"
    )


def triton_per_tensor_bmm(
    a: QuantResult,
    b: QuantResult,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    do_transpose_b: bool = False,
    dequant_scale: Optional[torch.Tensor] = None,
    profile: bool = False,
) -> torch.Tensor:
    if a.tensor.ndim != 3 or b.tensor.ndim != 3:
        raise ValueError("BMM expects 3D quantized tensors")

    a_tensor = a.tensor
    if a_tensor.stride(-1) != 1:
        raise ValueError(
            "A must be contiguous along K; "
            f"shape={tuple(a_tensor.shape)}, strides={a_tensor.stride()}"
        )

    input_b_layout = _b_layout(b.tensor)
    if do_transpose_b:
        if input_b_layout != "n-major":
            raise ValueError(
                "do_transpose_b=True requires an N-major B input; "
                f"strides={b.tensor.stride()}"
            )
        b_tensor = b.tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        b_tensor = b.tensor
    b_n_major = _b_layout(b_tensor) == "n-major"

    batch, m, k = a_tensor.shape
    b_batch, b_k, n = b_tensor.shape

    if batch != b_batch or k != b_k:
        raise ValueError(
            f"shape mismatch: A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    if a_tensor.device != b_tensor.device:
        raise ValueError(
            f"A and B must be on the same device: A={a_tensor.device}, B={b_tensor.device}"
        )
    if a_tensor.dtype != b_tensor.dtype:
        raise ValueError(
            f"A and B must have the same FP8 dtype: A={a_tensor.dtype}, B={b_tensor.dtype}"
        )

    if out is None:
        out = torch.empty((batch, m, n), device=a_tensor.device, dtype=out_dtype)
    elif out.shape != (batch, m, n):
        raise ValueError(
            f"out must have shape {(batch, m, n)}, got {tuple(out.shape)}"
        )
    elif out.device != a_tensor.device:
        raise ValueError(
            f"out must be on {a_tensor.device}, got {out.device}"
        )
    elif out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    elif out.stride(-1) != 1:
        raise ValueError(
            "out must be contiguous along N; "
            f"shape={tuple(out.shape)}, strides={out.stride()}"
        )
    if bias is None:
        bias_ptr = a_tensor
        stride_biasb = stride_biasm = stride_biasn = 0
    else:
        if bias.shape != (batch, m, n):
            raise ValueError(
                f"bias must have shape {(batch, m, n)}, got {tuple(bias.shape)}"
            )
        if bias.device != a_tensor.device:
            raise ValueError(
                f"bias must be on {a_tensor.device}, got {bias.device}"
            )
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride()

    if dequant_scale is None:
        dequant_scale = a.dequant_scale * b.dequant_scale
    if dequant_scale.numel() != 1:
        raise ValueError(
            "per-tensor BMM expects one combined dequant scale, "
            f"got shape={tuple(dequant_scale.shape)}"
        )
    if dequant_scale.dtype != torch.float32:
        raise ValueError(
            f"dequant_scale must be float32, got {dequant_scale.dtype}"
        )
    if dequant_scale.device != a_tensor.device:
        raise ValueError(
            "dequant_scale must be on the same device as A and B: "
            f"scale={dequant_scale.device}, tensors={a_tensor.device}"
        )

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        batch,
    )
    kernel = (
        batch_fp8_per_tensor_bmm_kernel
        if profile
        else batch_fp8_per_tensor_bmm_kernel_autotuned
    )
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
        a_tensor,
        b_tensor,
        out,
        bias_ptr,
        dequant_scale,
        m,
        n,
        k,
        *a_tensor.stride(),
        *b_tensor.stride(),
        *out.stride(),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIAS=bias is not None,
        B_N_MAJOR=b_n_major,
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
    partial(triton_per_tensor_bmm, do_transpose_b=False),
    quant_impl="triton_per_tensor",
    layout="n",
    prepare_b=lambda value: prepare_b_layout(value, "n"),
    prepare_call_kwargs=lambda a, b: {
        "dequant_scale": a.dequant_scale * b.dequant_scale
    },
    description="Migrated FP8 BMM with contiguous [B,K,N] right operand.",
)
register_bmm(
    "triton_per_tensor_k",
    partial(triton_per_tensor_bmm, do_transpose_b=False),
    quant_impl="triton_per_tensor",
    layout="k",
    prepare_b=lambda value: prepare_b_layout(value, "k"),
    prepare_call_kwargs=lambda a, b: {
        "dequant_scale": a.dequant_scale * b.dequant_scale
    },
    description="FP8 BMM with a prepacked K-major [B,K,N] right operand.",
)
register_bmm(
    "triton_per_tensor_n_transpose",
    partial(triton_per_tensor_bmm, do_transpose_b=True),
    quant_impl="triton_per_tensor",
    layout="n",
    prepare_b=lambda value: prepare_b_layout(value, "n"),
    prepare_call_kwargs=lambda a, b: {
        "dequant_scale": a.dequant_scale * b.dequant_scale
    },
    description="N-major FP8 BMM with explicit N-to-K packing inside the call.",
)
