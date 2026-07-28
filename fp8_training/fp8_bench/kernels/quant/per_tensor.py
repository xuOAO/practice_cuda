from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fp8_per_tensor_quant_kernel(
    x_ptr,
    y_ptr,
    m,
    n,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_yb,
    stride_ym,
    stride_yn,
    scale_ptr,
    fp8_max: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if dim == 3:
        x_ptr += batch * stride_xb
        y_ptr += batch * stride_yb

    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    x_offsets = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_offsets = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    value = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    value = tl.clamp(value * scale, min=-fp8_max, max=fp8_max)
    tl.store(y_ptr + y_offsets, value, mask=mask)


_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

fp8_per_tensor_quant_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n"],
)(fp8_per_tensor_quant_kernel)
