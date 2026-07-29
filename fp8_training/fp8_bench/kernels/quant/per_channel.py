from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def fp8_per_channel_quant_kernel(
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
    stride_sx,
    channel_axis: tl.constexpr, # -1 or -2
    X_N_MAJOR: tl.constexpr,
    Y_N_MAJOR: tl.constexpr,
    fp8_max: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if dim == 3:
        x_ptr += batch * stride_xb
        y_ptr += batch * stride_yb
        dequant_scale_ptr += batch * stride_sb

    if channel_axis == -2:
        pid_m = pid
        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(m, n),
            strides=(stride_xm, stride_xn),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_N_MAJOR else (0, 1)
        )

        # pass-one
        x_safe_amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for i in tl.range(0, n, BLOCK_N):
            x_value = tl.load(
                x_block,
                boundary_check=(0, 1),
                padding_option="zero"
            ).to(tl.float32)
            x_amax = tl.max(tl.abs(x_value), axis=1)
            x_safe_amax = tl.maximum(x_amax, x_safe_amax)
            x_block = tl.advance(x_block, (0, BLOCK_N))
        # get scale and store it
        x_safe_amax = tl.maximum(x_safe_amax, EPS)
        quant_scale = fp8_max / x_safe_amax
        dequant_scale = x_safe_amax / fp8_max
        offs_dequant_scale = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(
            dequant_scale_ptr + offs_dequant_scale * stride_sx,
            dequant_scale,
            mask=offs_dequant_scale < m
        )

        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(m, n),
            strides=(stride_xm, stride_xn),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_N_MAJOR else (0, 1)
        )
        y_block = tl.make_block_ptr(
            base=y_ptr,
            shape=(m, n),
            strides=(stride_ym, stride_yn),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_N_MAJOR else (0, 1)
        )

        # pass-two
        for i in tl.range(0, n, BLOCK_N):
            x_value = tl.load(
                x_block,
                boundary_check=(0, 1),
                padding_option="zero"
            ).to(tl.float32)
            y_value = tl.clamp(x_value * quant_scale[:, None], min=-fp8_max, max=fp8_max)
            y_value = y_value.to(y_ptr.dtype.element_ty)
            tl.store(
                y_block,
                y_value,
                boundary_check=(0, 1),
            )
            x_block = tl.advance(x_block, (0, BLOCK_N))
            y_block = tl.advance(y_block, (0, BLOCK_N))
    else:
        pid_n = pid
        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(m, n),
            strides=(stride_xm, stride_xn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_N_MAJOR else (0, 1)
        )

        # pass-one
        x_safe_amax = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for i in tl.range(0, m, BLOCK_M):
            x_value = tl.load(
                x_block,
                boundary_check=(0, 1),
                padding_option="zero"
            ).to(tl.float32)
            x_amax = tl.max(tl.abs(x_value), axis=0)
            x_safe_amax = tl.maximum(x_amax, x_safe_amax)
            x_block = tl.advance(x_block, (BLOCK_M, 0))
        # get scale and store it
        x_safe_amax = tl.maximum(x_safe_amax, EPS)
        quant_scale = fp8_max / x_safe_amax
        dequant_scale = x_safe_amax / fp8_max
        offs_dequant_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            dequant_scale_ptr + offs_dequant_scale * stride_sx,
            dequant_scale,
            mask=offs_dequant_scale < n
        )

        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(m, n),
            strides=(stride_xm, stride_xn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_N_MAJOR else (0, 1)
        )
        y_block = tl.make_block_ptr(
            base=y_ptr,
            shape=(m, n),
            strides=(stride_ym, stride_yn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_N_MAJOR else (0, 1)
        )

        # pass-two
        for i in tl.range(0, m, BLOCK_M):
            x_value = tl.load(
                x_block,
                boundary_check=(0, 1),
                padding_option="zero"
            ).to(tl.float32)
            y_value = tl.clamp(x_value * quant_scale[None, :], min=-fp8_max, max=fp8_max)
            y_value = y_value.to(y_ptr.dtype.element_ty)
            tl.store(
                y_block,
                y_value,
                boundary_check=(0, 1),
            )
            x_block = tl.advance(x_block, (BLOCK_M, 0))
            y_block = tl.advance(y_block, (BLOCK_M, 0))
        
_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]

fp8_per_channel_quant_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n"],
)(fp8_per_channel_quant_kernel)