from __future__ import annotations

import triton
import triton.language as tl

from fp8_bench.kernels.bmm.activation import apply_activation


@triton.jit
def batch_fp8_per_tensor_bmm_kernel(
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
    A_K_MAJOR: tl.constexpr,
    B_N_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
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
        order=(1, 0) if A_K_MAJOR else (0, 1),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr + batch * stride_bb,
        shape=(k, n),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0) if B_N_MAJOR else (0, 1),
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
    accumulator = apply_activation(accumulator, ACTIVATION)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def batch_fp8_per_tensor_bmm_tma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    scale_a_ptr,
    scale_b_ptr,
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
    A_K_MAJOR: tl.constexpr,
    B_N_MAJOR: tl.constexpr,
    SCALES_ARE_QUANT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
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

    if A_K_MAJOR:
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr + batch * stride_ab,
            shape=[m, k],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K],
            padding_option="zero",
        )
    else:
        # M-major A: fetch the logical [M,K] tile as [K,M] so the TMA
        # descriptor's last dimension remains contiguous, then transpose it.
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr + batch * stride_ab,
            shape=[k, m],
            strides=[stride_ak, stride_am],
            block_shape=[BLOCK_K, BLOCK_M],
            padding_option="zero",
        )
    if B_N_MAJOR:
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + batch * stride_bb,
            shape=[k, n],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_K, BLOCK_N],
            padding_option="zero",
        )
    else:
        # K-major B: the [K,N] tile is fetched as a transposed [N,K] block so
        # that the descriptor's last (contiguous) dimension is K.
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + batch * stride_bb,
            shape=[n, k],
            strides=[stride_bn, stride_bk],
            block_shape=[BLOCK_N, BLOCK_K],
            padding_option="zero",
        )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kk in tl.range(0, k, BLOCK_K):
        if A_K_MAJOR:
            a = a_desc.load([pid_m * BLOCK_M, kk])
        else:
            a = tl.trans(a_desc.load([kk, pid_m * BLOCK_M]))
        if B_N_MAJOR:
            b = b_desc.load([kk, pid_n * BLOCK_N])
        else:
            b = tl.trans(b_desc.load([pid_n * BLOCK_N, kk]))
        accumulator = tl.dot(a, b, accumulator)

    if SCALES_ARE_QUANT:
        # torchAO quantizes with q = fp8(x * scale), so the accumulated FP8
        # product is dequantized by 1 / (scale_a * scale_b). Keeping this in
        # the BMM avoids a separate scalar GPU kernel before every GEMM.
        accumulator *= 1.0 / (tl.load(scale_a_ptr) * tl.load(scale_b_ptr))
    else:
        # Benchmark/registry callers already provide one combined dequant
        # scale. scale_b_ptr is unused in this mode.
        accumulator *= tl.load(scale_a_ptr)
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

    c_desc = tl.make_tensor_descriptor(
        base=c_ptr + batch * stride_cb,
        shape=[m, n],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    accumulator = apply_activation(accumulator, ACTIVATION)
    c_desc.store(
        [pid_m * BLOCK_M, pid_n * BLOCK_N],
        accumulator.to(c_ptr.dtype.element_ty),
    )


_CONFIGS = [
    # Legacy a_k_b_n winners.
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy a_k_b_k winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    # Shared legacy a_k_b_k and a_m_b_n winner.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy a_m_b_n winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
]

_TMA_CONFIGS = [
    # Legacy TMA a_k_b_n winner.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy TMA a_k_b_k winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    # Legacy TMA a_m_b_n winners.
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
]

batch_fp8_per_tensor_bmm_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n", "k", "A_K_MAJOR", "B_N_MAJOR", "USE_BIAS", "ACTIVATION"],
)(batch_fp8_per_tensor_bmm_kernel)

batch_fp8_per_tensor_bmm_tma_kernel_autotuned = triton.autotune(
    configs=_TMA_CONFIGS,
    key=[
        "m",
        "n",
        "k",
        "A_K_MAJOR",
        "B_N_MAJOR",
        "SCALES_ARE_QUANT",
        "USE_BIAS",
        "ACTIVATION",
    ],
)(batch_fp8_per_tensor_bmm_tma_kernel)
