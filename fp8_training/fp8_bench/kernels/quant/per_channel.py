from __future__ import annotations

import torch
import triton
import triton.language as tl


# 融合后的性能不是很好
@torch.compile
def fp8_per_channel_quant_torch_compile(
    x: torch.Tensor,
    channel_axis: int,
    fp8_dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduction_dim = -1 if channel_axis == -2 else -2
    x_safe_amax = torch.amax(
        torch.abs(x.float()),
        dim=reduction_dim,
    ).clamp_min(eps)
    fp8_max = torch.finfo(fp8_dtype).max
    quant_scale = fp8_max / x_safe_amax
    dequant_scale = x_safe_amax / fp8_max
    output = torch.clamp(
        x.float() * quant_scale.unsqueeze(reduction_dim),
        min=-fp8_max,
        max=fp8_max,
    ).to(fp8_dtype)
    return output, dequant_scale

# one-pass的优势不明显，并且对主序要求高，如果行主序，但是按列规约，访存不合并
# 性能不优势的原因如下：
#   tl.arange要2的幂，对于N = 640这样的shape，BLOCK_SIZE=1024，会导致大量的无效读
#   2-pass第二遍读取会命中L2cache，并不完全等价于读HBM
@triton.jit
def fp8_per_channel_quant_one_pass_kernel(
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
    channel_axis: tl.constexpr,  # -1 or -2
    fp8_max: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """Quantize one complete channel after loading its values only once."""
    tl.static_assert(channel_axis == -1 or channel_axis == -2)

    channel = tl.program_id(0)
    batch = tl.program_id(1)
    if dim == 3:
        x_ptr += batch * stride_xb
        y_ptr += batch * stride_yb
        dequant_scale_ptr += batch * stride_sb

    reduction_offsets = tl.arange(0, BLOCK_SIZE)
    if channel_axis == -2:
        # Each program owns a logical row and reduces across N.
        reduction_mask = reduction_offsets < n
        x_offsets = channel * stride_xm + reduction_offsets * stride_xn
        y_offsets = channel * stride_ym + reduction_offsets * stride_yn
    else:
        # Each program owns a logical column and reduces across M.
        reduction_mask = reduction_offsets < m
        x_offsets = reduction_offsets * stride_xm + channel * stride_xn
        y_offsets = reduction_offsets * stride_ym + channel * stride_yn

    # Keep the complete channel live so quantization can reuse it after the
    # reduction instead of reading it from global memory a second time.
    x_value = tl.load(
        x_ptr + x_offsets,
        mask=reduction_mask,
        other=0.0,
    ).to(tl.float32)
    x_safe_amax = tl.maximum(tl.max(tl.abs(x_value), axis=0), EPS)
    quant_scale = fp8_max / x_safe_amax
    dequant_scale = x_safe_amax / fp8_max

    y_value = tl.clamp(
        x_value * quant_scale,
        min=-fp8_max,
        max=fp8_max,
    ).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + y_offsets,
        y_value,
        mask=reduction_mask,
    )
    tl.store(
        dequant_scale_ptr + channel * stride_sx,
        dequant_scale,
    )


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
    # Row-wise reduction (channel_axis=-2): process a small group of rows
    # while reducing wider contiguous K chunks.
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    # Balanced candidates shared by both reduction directions.
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    # Column-wise reduction (channel_axis=-1): reduce taller M chunks while
    # keeping enough adjacent output channels for coalesced memory traffic.
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    # Higher-warp variants for larger tiles and long reductions.
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
]

fp8_per_channel_quant_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n", "channel_axis", "X_N_MAJOR", "Y_N_MAJOR"],
)(fp8_per_channel_quant_kernel)
