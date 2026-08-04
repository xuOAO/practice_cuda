from __future__ import annotations

import triton
import triton.language as tl

from fp8_bench.kernels.bmm.activation import apply_activation


@triton.jit
def batch_fp8_per_block_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    dequant_scale_a_ptr,
    dequant_scale_b_ptr,
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
    stride_scale_ab,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bb,
    stride_scale_bk,
    stride_scale_bn,
    USE_BIAS: tl.constexpr,
    A_K_MAJOR: tl.constexpr,
    B_N_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    QUANT_BLOCK_M: tl.constexpr,
    QUANT_BLOCK_K: tl.constexpr,
    QUANT_BLOCK_N: tl.constexpr,
    NUM_QUANT_BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    tl.static_assert(QUANT_BLOCK_K >= BLOCK_K)
    tl.static_assert(QUANT_BLOCK_K % BLOCK_K == 0)

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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_quant_m = offs_m // QUANT_BLOCK_M
    offs_quant_n = offs_n // QUANT_BLOCK_N
    mask_quant_m = (offs_m < m)[:, None] # shape: (BLOCK_M, 1)
    mask_quant_n = (offs_n < n)[None, :] # shape: (1, BLOCK_N)
    # pid_quant_m = pid_m * BLOCK_M // QUANT_BLOCK_M
    # pid_quant_n = pid_n * BLOCK_N // QUANT_BLOCK_N
    a_dequant_scale_ptr = (
        dequant_scale_a_ptr + batch * stride_scale_ab
        + offs_quant_m * stride_scale_am
    )[:, None] # shape: (BLOCK_M, 1)
    b_dequant_scale_ptr = (
        dequant_scale_b_ptr + batch * stride_scale_bb
        + offs_quant_n * stride_scale_bn
    )[None, :] # shape: (1, BLOCK_N)

    cuda_core_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for quant_k in tl.range(0, NUM_QUANT_BLOCK_K):
        tensor_core_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for _ in tl.static_range(0, QUANT_BLOCK_K, BLOCK_K):
            a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
            tensor_core_acc = tl.dot(a, b, tensor_core_acc)
            a_block = tl.advance(a_block, (0, BLOCK_K))
            b_block = tl.advance(b_block, (BLOCK_K, 0))
        # a_dequant_scale: shape (BLOCK_M, 1), b_dequant_scale: shape (1, BLOCK_N)
        a_dequant_scale = tl.load(a_dequant_scale_ptr + quant_k * stride_scale_ak, mask=mask_quant_m, other=0.0)
        b_dequant_scale = tl.load(b_dequant_scale_ptr + quant_k * stride_scale_bk, mask=mask_quant_n, other=0.0)
        cuda_core_acc += tensor_core_acc * (a_dequant_scale * b_dequant_scale)

    if USE_BIAS:
        bias_block = tl.make_block_ptr(
            base=bias_ptr + batch * stride_biasb,
            shape=(m, n),
            strides=(stride_biasm, stride_biasn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        cuda_core_acc += tl.load(
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
    cuda_core_acc = apply_activation(cuda_core_acc, ACTIVATION)
    tl.store(c_block, cuda_core_acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def batch_fp8_per_block_bmm_tma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    dequant_scale_a_ptr,
    dequant_scale_b_ptr,
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
    stride_scale_ab,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bb,
    stride_scale_bk,
    stride_scale_bn,
    USE_BIAS: tl.constexpr,
    A_K_MAJOR: tl.constexpr,
    B_N_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    QUANT_BLOCK_M: tl.constexpr,
    QUANT_BLOCK_K: tl.constexpr,
    QUANT_BLOCK_N: tl.constexpr,
    NUM_QUANT_BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    tl.static_assert(QUANT_BLOCK_K >= BLOCK_K)
    tl.static_assert(QUANT_BLOCK_K % BLOCK_K == 0)

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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_quant_m = offs_m // QUANT_BLOCK_M
    offs_quant_n = offs_n // QUANT_BLOCK_N
    mask_quant_m = (offs_m < m)[:, None]  # shape: (BLOCK_M, 1)
    mask_quant_n = (offs_n < n)[None, :]  # shape: (1, BLOCK_N)
    a_dequant_scale_ptr = (
        dequant_scale_a_ptr + batch * stride_scale_ab
        + offs_quant_m * stride_scale_am
    )[:, None]  # shape: (BLOCK_M, 1)
    b_dequant_scale_ptr = (
        dequant_scale_b_ptr + batch * stride_scale_bb
        + offs_quant_n * stride_scale_bn
    )[None, :]  # shape: (1, BLOCK_N)

    cuda_core_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for quant_k in tl.range(0, NUM_QUANT_BLOCK_K):
        tensor_core_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_base = quant_k * QUANT_BLOCK_K
        for inner in tl.static_range(0, QUANT_BLOCK_K, BLOCK_K):
            kk = k_base + inner
            if A_K_MAJOR:
                a = a_desc.load([pid_m * BLOCK_M, kk])
            else:
                a = tl.trans(a_desc.load([kk, pid_m * BLOCK_M]))
            if B_N_MAJOR:
                b = b_desc.load([kk, pid_n * BLOCK_N])
            else:
                b = tl.trans(b_desc.load([pid_n * BLOCK_N, kk]))
            tensor_core_acc = tl.dot(a, b, tensor_core_acc)
        # a_dequant_scale: shape (BLOCK_M, 1), b_dequant_scale: shape (1, BLOCK_N)
        a_dequant_scale = tl.load(a_dequant_scale_ptr + quant_k * stride_scale_ak, mask=mask_quant_m, other=0.0)
        b_dequant_scale = tl.load(b_dequant_scale_ptr + quant_k * stride_scale_bk, mask=mask_quant_n, other=0.0)
        cuda_core_acc += tensor_core_acc * (a_dequant_scale * b_dequant_scale)

    if USE_BIAS:
        bias_block = tl.make_block_ptr(
            base=bias_ptr + batch * stride_biasb,
            shape=(m, n),
            strides=(stride_biasm, stride_biasn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        cuda_core_acc += tl.load(
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
    cuda_core_acc = apply_activation(cuda_core_acc, ACTIVATION)
    c_desc.store(
        [pid_m * BLOCK_M, pid_n * BLOCK_N],
        cuda_core_acc.to(c_ptr.dtype.element_ty),
    )


_CONFIGS = [
    # Legacy a_k_b_n winner.
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Shared legacy a_k_b_n and a_k_b_k winner.
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy a_k_b_k winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy a_m_b_n winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=5,
    ),
]

_TMA_CONFIGS = [
    # Shared legacy TMA a_k_b_n and a_k_b_k winner.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy TMA a_k_b_k winners.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    # Legacy TMA a_m_b_n winner.
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
]

batch_fp8_per_block_bmm_kernel_autotuned = triton.autotune(
    configs=_CONFIGS,
    key=["m", "n", "k", "A_K_MAJOR", "B_N_MAJOR", "USE_BIAS",
         "QUANT_BLOCK_M", "QUANT_BLOCK_K", "QUANT_BLOCK_N", "ACTIVATION"]
)(batch_fp8_per_block_bmm_kernel)

batch_fp8_per_block_bmm_tma_kernel_autotuned = triton.autotune(
    configs=_TMA_CONFIGS,
    key=["m", "n", "k", "A_K_MAJOR", "B_N_MAJOR", "USE_BIAS",
         "QUANT_BLOCK_M", "QUANT_BLOCK_K", "QUANT_BLOCK_N", "ACTIVATION"]
)(batch_fp8_per_block_bmm_tma_kernel)
