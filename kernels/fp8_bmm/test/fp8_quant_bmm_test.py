import os
import sys

import pytest
import torch
import triton

# Make fp8_bmm importable as a namespace package (no __init__.py in the repo):
# the kernel module lives at kernel/fp8_quant_bmm_kernel.py and imports resolve
# only when the fp8_bmm dir is on sys.path.
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

from fp8_utils import FP8Format  # noqa: E402
from kernel.fp8_quant_bmm_kernel import (  # noqa: E402
    batch_quant_fp8_mm_kernel_test as fp8_bmm_kernel,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 bmm kernel requires CUDA"
)


# ---------------------------------------------------------------------------
# launcher (test-only; mirrors the fp8_quant_triton warper pattern)
# ---------------------------------------------------------------------------
#
# Computes, for 3D batched inputs a:[B,M,K], b:[B,K,N]:
#     C[b,m,n] = ((a[b,m,:] @ b[b,:,n]) * quant_scale + bias[b,m,n]).to(out_dtype)
#
# `quant_scale` is a scalar applied to the fp32 accumulator before the optional
# bias is added and the result is cast to the output dtype.
def fp8_quant_bmm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    quant_scale,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    assert a.dim() == 3 and b.dim() == 3, "a and b must be 3D: (B,M,K) and (B,K,N)"
    B, M, K = a.shape
    Bb, Kk, N = b.shape
    assert B == Bb and K == Kk, f"shape mismatch: a {a.shape}, b {b.shape}"

    c = torch.empty((B, M, N), device=a.device, dtype=out_dtype)

    use_bias = bias is not None
    if use_bias:
        # Kernel indexes bias with bid*stride_biasb, so materialise a batch dim.
        if bias.dim() == 2:
            bias = bias.unsqueeze(0).expand(B, -1, -1).contiguous()
        bias_ptr = bias
        sb, sm, sn = bias.stride(0), bias.stride(1), bias.stride(2)
    else:
        # Placeholder pointer; never dereferenced because USE_BIASE is constexpr.
        bias_ptr = a
        sb = sm = sn = 0

    if not torch.is_tensor(quant_scale):
        quant_scale = torch.tensor(
            float(quant_scale), device=a.device, dtype=torch.float32
        )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        B,
    )

    fp8_bmm_kernel[grid](
        a,
        b,
        c,
        bias_ptr,
        quant_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        sb,
        sm,
        sn,
        USE_BIASE=use_bias,
    )
    return c


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------
def fp8_quant_bmm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    quant_scale,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Reference: exact f32 matmul of the (already fp8) inputs, then scale,
    bias, and cast. Same math the kernel performs, accumulation order aside."""
    acc = a.to(torch.float32) @ b.to(torch.float32)
    scale = quant_scale.to(torch.float32) if torch.is_tensor(quant_scale) else float(quant_scale)
    acc = acc * scale
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    return acc.to(out_dtype)


# Shapes: (B, M, K, N). Aligned, small-batch, unaligned (exercises boundary
# masking on M/K/N), and a larger bench-like shape.
SHAPES = [
    pytest.param((1, 128, 128, 128), id="aligned"),
    pytest.param((2, 64, 64, 64), id="small-batch"),
    pytest.param((3, 33, 47, 65), id="unaligned"),
    pytest.param((4, 256, 512, 256), id="bench-like"),
]


@pytest.fixture(autouse=True)
def _seed():  # noqa: PT004 - fixture used for side effect (seeding)
    torch.manual_seed(0)


def _rand_fp8(shape, dtype=FP8Format.E4M3.value.dtype_fwd):
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.5).to(dtype)


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------
def test_output_dtype_bf16():
    a, b = _rand_fp8((1, 64, 64)), _rand_fp8((1, 64, 64))
    out = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16


def test_output_dtype_fp8():
    a, b = _rand_fp8((1, 64, 64)), _rand_fp8((1, 64, 64))
    out = fp8_quant_bmm_triton(
        a, b, quant_scale=1.0, out_dtype=FP8Format.E4M3.value.dtype_fwd
    )
    assert out.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# correctness vs reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape):
    B, M, K, N = shape
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    ref = fp8_quant_bmm_ref(a, b, quant_scale=1.0, out_dtype=torch.bfloat16)
    tri = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.bfloat16)
    assert tri.shape == ref.shape == (B, M, N)
    # Same fp8 inputs; ref accumulates in f32, kernel in f32 via tensor cores.
    # Differences are summation-order only, well under bf16 precision.
    assert torch.allclose(tri.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference_fp8_out(shape):
    B, M, K, N = shape
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    ref = fp8_quant_bmm_ref(
        a, b, quant_scale=1.0, out_dtype=FP8Format.E4M3.value.dtype_fwd
    )
    tri = fp8_quant_bmm_triton(
        a, b, quant_scale=1.0, out_dtype=FP8Format.E4M3.value.dtype_fwd
    )
    # E4M3 has 3 mantissa bits -> 1 ULP is ~12.5% relative. A tiny f32
    # accumulation-order difference (kernel vs ref) can flip an element's
    # rounding direction at a boundary, so allow 1 e4m3 ULP (rtol=0.2).
    assert torch.allclose(tri.to(torch.float32), ref.to(torch.float32), rtol=2e-1, atol=1e-1)


# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_with_bias(shape):
    B, M, K, N = shape
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    bias = torch.randn((B, M, N), device="cuda", dtype=torch.float32)
    ref = fp8_quant_bmm_ref(a, b, quant_scale=1.0, bias=bias, out_dtype=torch.bfloat16)
    tri = fp8_quant_bmm_triton(a, b, quant_scale=1.0, bias=bias, out_dtype=torch.bfloat16)
    assert torch.allclose(tri.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-2)


def test_with_2d_bias_broadcast():
    # bias given as (M, N) should broadcast across the batch dim.
    B, M, K, N = 4, 64, 64, 64
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    bias = torch.randn((M, N), device="cuda", dtype=torch.float32)
    ref = fp8_quant_bmm_ref(a, b, quant_scale=1.0, bias=bias, out_dtype=torch.bfloat16)
    tri = fp8_quant_bmm_triton(a, b, quant_scale=1.0, bias=bias, out_dtype=torch.bfloat16)
    assert torch.allclose(tri.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# quant_scale / output range (clamping when casting to fp8)
# ---------------------------------------------------------------------------
def test_quant_scale_applied():
    # A non-unit scale must actually scale the accumulator.
    B, M, K, N = 1, 64, 64, 64
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    out_s1 = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.float32)
    out_s2 = fp8_quant_bmm_triton(a, b, quant_scale=2.0, out_dtype=torch.float32)
    # Skip entries that saturated to 0 to avoid divide-by-zero noise.
    nonzero = out_s1.abs() > 1e-3
    ratio = (out_s2.to(torch.float32)[nonzero] / out_s1.to(torch.float32)[nonzero])
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=1e-2, atol=1e-2)


def test_output_range_clamped():
    # Large scale forces the fp32 accumulator past the e4m3 range; the fp8
    # cast must saturate rather than overflow / wrap.
    B, M, K, N = 1, 64, 128, 64
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    out = fp8_quant_bmm_triton(
        a, b, quant_scale=100.0, out_dtype=FP8Format.E4M3.value.dtype_fwd
    )
    fp8_range = FP8Format.E4M3.value.max_fwd
    assert out.to(torch.float32).abs().amax().item() <= fp8_range + 1e-3


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_all_zeros():
    a = torch.zeros((2, 32, 64), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros((2, 64, 32), device="cuda", dtype=torch.float8_e4m3fn)
    out = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.float32)
    assert not torch.isnan(out).any()
    assert out.abs().amax().item() == 0.0


def test_batch_independence():
    # Each batch element must use its own a[b], b[b] slices.
    B, M, K, N = 2, 32, 32, 32
    a = _rand_fp8((B, M, K))
    b = _rand_fp8((B, K, N))
    out = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.float32)
    for i in range(B):
        single = fp8_quant_bmm_triton(
            a[i : i + 1], b[i : i + 1], quant_scale=1.0, out_dtype=torch.float32
        )
        assert torch.allclose(out[i], single[0], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_determinism():
    a = _rand_fp8((2, 64, 64))
    b = _rand_fp8((2, 64, 64))
    out1 = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.bfloat16)
    out2 = fp8_quant_bmm_triton(a, b, quant_scale=1.0, out_dtype=torch.bfloat16)
    assert torch.equal(out1.to(torch.float32), out2.to(torch.float32))
