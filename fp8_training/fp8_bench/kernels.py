from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fp8_quant_kernel(
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


_QUANT_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

fp8_quant_kernel_autotuned = triton.autotune(
    configs=_QUANT_CONFIGS,
    key=["m", "n"],
)(fp8_quant_kernel)


@triton.jit
def batch_fp8_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    scale_ptr,
    m,
    n,
    k,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    stride_biasm,
    stride_biasn,
    USE_BIAS: tl.constexpr,
    B_N_ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    batch = tl.program_id(1)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_group) % group_m
    pid_n = (pid % num_pid_group) // group_m

    a_block = tl.make_block_ptr(
        base=a_ptr + batch * stride_ab,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr + batch * stride_bb,
        shape=(k, n),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0) if B_N_ORDER else (0, 1),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in tl.range(0, k, BLOCK_K):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        accumulator = tl.dot(a, b, accumulator)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    accumulator *= tl.load(scale_ptr)
    if USE_BIAS:
        bias_block = tl.make_block_ptr(
            base=bias_ptr + batch * stride_biasb,
            shape=(m, n),
            strides=(stride_biasm, stride_biasn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        accumulator += tl.load(
            bias_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )

    c_block = tl.make_block_ptr(
        base=c_ptr + batch * stride_cb,
        shape=(m, n),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


_BMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
]

batch_fp8_bmm_kernel_autotuned = triton.autotune(
    configs=_BMM_CONFIGS,
    key=["m", "n", "k", "B_N_ORDER", "USE_BIAS"],
)(batch_fp8_bmm_kernel)
