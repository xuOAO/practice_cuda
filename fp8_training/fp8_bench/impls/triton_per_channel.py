from __future__ import annotations

from functools import partial
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


def _matrix_layout(tensor: torch.Tensor, name: str) -> str:
    if tensor.stride(-1) == 1:
        return "n-major"
    if tensor.stride(-2) == 1:
        return "k-major"
    raise ValueError(
        f"{name} must be contiguous along one matrix dimension; "
        f"shape={tuple(tensor.shape)}, strides={tensor.stride()}"
    )


def triton_per_channel_quant(
    x: torch.Tensor,
    *,
    channel_axis: int = -1,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
    profile: bool = False,
) -> QuantResult:
    if x.ndim not in {2, 3}:
        raise ValueError(
            f"per-channel quant expects a 2D or 3D tensor, got {tuple(x.shape)}"
        )
    if channel_axis not in {-1, -2}:
        raise ValueError(f"channel_axis must be -1 or -2, got {channel_axis}")

    x_n_major = _matrix_layout(x, "quant input") == "n-major"
    output = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
    if x.ndim == 2:
        batch = 1
        m, n = x.shape
        stride_xb = stride_yb = 0
    else:
        batch, m, n = x.shape
        stride_xb = x.stride(0)
        stride_yb = output.stride(0)

    channel_size = m if channel_axis == -2 else n
    scale_storage = torch.empty(
        (batch, channel_size),
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(
            m if channel_axis == -2 else n,
            meta["BLOCK_M"] if channel_axis == -2 else meta["BLOCK_N"],
        ),
        batch,
    )

    kernel = (
        fp8_per_channel_quant_kernel
        if profile
        else fp8_per_channel_quant_kernel_autotuned
    )
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
        scale_storage,
        scale_storage.stride(0),
        scale_storage.stride(1),
        channel_axis=channel_axis,
        X_N_MAJOR=x_n_major,
        Y_N_MAJOR=True,
        fp8_max=torch.finfo(fp8_dtype).max,
        dim=x.ndim,
        EPS=eps,
        **launch_kwargs,
    )
    dequant_scale = scale_storage.squeeze(0) if x.ndim == 2 else scale_storage
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
        tensor=value.tensor.transpose(-1, -2).contiguous().transpose(-1, -2),
        dequant_scale=value.dequant_scale,
        impl=value.impl,
        meta={**value.meta, "layout": "k"},
    )


def _validate_scale(
    value: QuantResult,
    *,
    expected_shape: tuple[int, int],
    expected_axis: int,
    operand: str,
) -> None:
    scale = value.dequant_scale
    if scale.shape != expected_shape:
        raise ValueError(
            f"{operand} dequant scale must have shape {expected_shape}, "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype != torch.float32:
        raise ValueError(
            f"{operand} dequant scale must be float32, got {scale.dtype}"
        )
    if scale.device != value.tensor.device:
        raise ValueError(
            f"{operand} dequant scale must be on {value.tensor.device}, "
            f"got {scale.device}"
        )
    if value.meta.get("channel_axis") != expected_axis:
        raise ValueError(
            f"{operand} must be quantized with channel_axis={expected_axis}, "
            f"got {value.meta.get('channel_axis')}"
        )


def triton_per_channel_bmm(
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

    a_tensor = a.tensor
    if a_tensor.stride(-1) != 1:
        raise ValueError(
            "A must be contiguous along K; "
            f"shape={tuple(a_tensor.shape)}, strides={a_tensor.stride()}"
        )

    input_b_layout = _matrix_layout(b.tensor, "B")
    if do_transpose_b:
        if input_b_layout != "n-major":
            raise ValueError(
                "do_transpose_b=True requires an N-major B input; "
                f"strides={b.tensor.stride()}"
            )
        b_tensor = b.tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        b_tensor = b.tensor
    b_n_major = _matrix_layout(b_tensor, "B") == "n-major"

    batch, m, k = a_tensor.shape
    b_batch, b_k, n = b_tensor.shape
    if batch != b_batch or k != b_k:
        raise ValueError(
            f"shape mismatch: A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    if a_tensor.device != b_tensor.device:
        raise ValueError(
            f"A and B must be on the same device: A={a_tensor.device}, "
            f"B={b_tensor.device}"
        )
    if a_tensor.dtype != b_tensor.dtype:
        raise ValueError(
            f"A and B must have the same FP8 dtype: A={a_tensor.dtype}, "
            f"B={b_tensor.dtype}"
        )

    _validate_scale(
        a,
        expected_shape=(batch, m),
        expected_axis=-2,
        operand="A",
    )
    _validate_scale(
        b,
        expected_shape=(batch, n),
        expected_axis=-1,
        operand="B",
    )

    if out is None:
        out = torch.empty((batch, m, n), device=a_tensor.device, dtype=out_dtype)
    elif out.shape != (batch, m, n):
        raise ValueError(
            f"out must have shape {(batch, m, n)}, got {tuple(out.shape)}"
        )
    elif out.device != a_tensor.device:
        raise ValueError(f"out must be on {a_tensor.device}, got {out.device}")
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
            raise ValueError(f"bias must be on {a_tensor.device}, got {bias.device}")
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride()

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        batch,
    )
    kernel = (
        batch_fp8_per_channel_bmm_kernel
        if profile
        else batch_fp8_per_channel_bmm_kernel_autotuned
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
        a.dequant_scale,
        b.dequant_scale,
        m,
        n,
        k,
        *a_tensor.stride(),
        *b_tensor.stride(),
        *out.stride(),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        *a.dequant_scale.stride(),
        *b.dequant_scale.stride(),
        USE_BIAS=bias is not None,
        B_N_MAJOR=b_n_major,
        **launch_kwargs,
    )
    return out


register_quant(
    "triton_per_channel",
    triton_per_channel_quant,
    "Per-channel scale quantization; supports E4M3 and E5M2.",
)
register_bmm(
    "triton_per_channel_n",
    partial(triton_per_channel_bmm, do_transpose_b=False),
    quant_impl="triton_per_channel",
    quant_a_kwargs={"channel_axis": -2},
    quant_b_kwargs={"channel_axis": -1},
    layout="n",
    prepare_b=lambda value: prepare_b_layout(value, "n"),
    description="Per-channel FP8 BMM with an N-major [B,K,N] right operand.",
)
register_bmm(
    "triton_per_channel_k",
    partial(triton_per_channel_bmm, do_transpose_b=False),
    quant_impl="triton_per_channel",
    quant_a_kwargs={"channel_axis": -2},
    quant_b_kwargs={"channel_axis": -1},
    layout="k",
    prepare_b=lambda value: prepare_b_layout(value, "k"),
    description="Per-channel FP8 BMM with a prepacked K-major right operand.",
)
register_bmm(
    "triton_per_channel_n_transpose",
    partial(triton_per_channel_bmm, do_transpose_b=True),
    quant_impl="triton_per_channel",
    quant_a_kwargs={"channel_axis": -2},
    quant_b_kwargs={"channel_axis": -1},
    layout="n",
    prepare_b=lambda value: prepare_b_layout(value, "n"),
    description="N-major per-channel FP8 BMM with packing inside the call.",
)
