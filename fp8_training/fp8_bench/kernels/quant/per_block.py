from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def fp8_per_block_quant_kernel(
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
    dequant_scale_ptr,
    stride_sb,
    stride_sm,
    stride_sn,
    fp8_max: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch = tl.program_id(2)
    if dim == 3:
        x_ptr += batch * stride_xb
        y_ptr += batch * stride_yb
        dequant_scale_ptr += batch * stride_sb
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    offs_x = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
    x_amax = tl.max(tl.abs(x))
    x_safe_amax = tl.maximum(x_amax, EPS)
    quant_scale = fp8_max / x_safe_amax
    dequant_scale = 1.0 / quant_scale
    tl.store(dequant_scale_ptr + pid_m * stride_sm + pid_n * stride_sn, dequant_scale)
    y = tl.clamp(x * quant_scale, min=-fp8_max, max=fp8_max)
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_y, y, mask=mask)
        
_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
]

fp8_per_block_quant_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n", "BLOCK_M", "BLOCK_N"],
)(fp8_per_block_quant_kernel)
