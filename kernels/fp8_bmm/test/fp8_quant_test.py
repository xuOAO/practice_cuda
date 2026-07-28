import os
import sys
import functools

import pytest
import torch

# Make fp8_bmm importable as a namespace package (no __init__.py in the repo):
# fp8_warper.py imports `from fp8_utils import ...` and `from kernel import ...`,
# which only resolve when the fp8_bmm dir is on sys.path.
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

import triton
from fp8_utils import FP8Format  # noqa: E402
from fp8_warper import fp8_quant_triton  # noqa: E402

fp8_quant_triton = functools.partial(fp8_quant_triton, is_test=True) 



pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 quant kernel requires CUDA"
)


def fp8_quant_ref(
    x: torch.Tensor,
    quant_dtype=FP8Format.E4M3.value.dtype_fwd,
    fp8_range=FP8Format.E4M3.value.max_fwd,
    eps=1e-12,
):
    """Reference implementation. Same math as bench/fp8_quant_bench.fp8_quant_torch,
    but without @torch.compile so float semantics are unambiguous."""
    x_min, x_max = x.aminmax()
    x_max_abs = torch.maximum(x_min.abs(), x_max.abs())
    x_max_abs = x_max_abs.clamp(min=eps)
    quant_factor = fp8_range / x_max_abs
    reciprocal_factor = quant_factor.reciprocal()
    quanted_x = (x * quant_factor).to(quant_dtype)
    return quanted_x, reciprocal_factor


# Shapes: bench 3D shapes + a small 2D shape + an unaligned shape (17, 33)
# specifically to exercise mask boundary handling (M,N not divisible by BLOCK).
SHAPES = [
    pytest.param((128, 256), id="2d-aligned"),
    pytest.param((17, 33), id="2d-unaligned"),
    pytest.param((32, 2048, 960), id="3d-bench0"),
    pytest.param((80, 2048, 640), id="3d-bench1"),
    pytest.param((14, 2048, 8192), id="3d-bench2"),
]


@pytest.fixture(autouse=True)
def _seed():  # noqa: PT004 - fixture used for side effect (seeding)
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------

def test_output_dtype_2d():
    x = torch.randn((128, 256), device="cuda", dtype=torch.float32)
    out, _ = fp8_quant_triton(x)
    assert out.dtype == torch.float8_e4m3fn


def test_output_dtype_3d():
    x = torch.randn((4, 64, 128), device="cuda", dtype=torch.float32)
    out, _ = fp8_quant_triton(x)
    assert out.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# bit-exact match against the reference (after both are cast to float32)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    ref_out, _ = fp8_quant_ref(x)
    tri_out, _ = fp8_quant_triton(x)
    assert tri_out.shape == ref_out.shape
    assert torch.equal(tri_out.to(torch.float32), ref_out.to(torch.float32))


# ---------------------------------------------------------------------------
# reciprocal factor (dequant scale)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_reciprocal_factor(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    _, ref_recip = fp8_quant_ref(x)
    _, tri_recip = fp8_quant_triton(x)
    # recip == max_abs / fp8_range; compare against the value computed from x.
    x_max_abs = x.abs().amax().clamp(min=1e-12)
    expected = x_max_abs / FP8Format.E4M3.value.max_fwd
    assert torch.allclose(tri_recip, expected, rtol=1e-5, atol=1e-7)
    assert torch.allclose(tri_recip, ref_recip, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# dequant round-trip accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_dequant_roundtrip(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    out, recip = fp8_quant_triton(x)
    # dequantize: x_hat = quanted_x * recip
    x_hat = out.to(torch.float32) * recip
    # E4M3 has ~2-3 mantissa bits; allow generous relative tolerance.
    # absolute tolerance scales with the input magnitude.
    atol = 0.05 * x.abs().amax().item()
    assert torch.allclose(x_hat, x, rtol=1e-1, atol=atol)


# ---------------------------------------------------------------------------
# output range / clamping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_output_range(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32) * 1000.0
    out, _ = fp8_quant_triton(x)
    fp8_range = FP8Format.E4M3.value.max_fwd
    assert out.to(torch.float32).abs().amax().item() <= fp8_range + 1e-3


def test_clamping_with_outliers():
    # Mostly small values, a few huge outliers that must be clamped to +-fp8_range.
    x = torch.randn((64, 64), device="cuda", dtype=torch.float32)
    x[0, 0] = 1e6
    x[1, 1] = -1e6
    out, _ = fp8_quant_triton(x)
    fp8_range = FP8Format.E4M3.value.max_fwd
    out_f32 = out.to(torch.float32)
    assert out_f32.abs().amax().item() <= fp8_range + 1e-3
    # The outlier entries themselves should be exactly at the range edge.
    assert out_f32[0, 0].item() == pytest.approx(fp8_range, abs=1e-2)
    assert out_f32[1, 1].item() == pytest.approx(-fp8_range, abs=1e-2)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_all_zeros():
    x = torch.zeros((32, 64), device="cuda", dtype=torch.float32)
    out, recip = fp8_quant_triton(x)
    assert not torch.isnan(out.to(torch.float32)).any()
    assert out.to(torch.float32).abs().amax().item() == 0.0
    assert torch.isfinite(recip).item()


# ---------------------------------------------------------------------------
# E5M2 format
# ---------------------------------------------------------------------------

def test_e5m2_format():
    x = torch.randn((64, 128), device="cuda", dtype=torch.float32)
    ref_out, _ = fp8_quant_ref(
        x, quant_dtype=FP8Format.E5M2.value.dtype_fwd, fp8_range=FP8Format.E5M2.value.max_fwd
    )
    tri_out, _ = fp8_quant_triton(
        x, quant_dtype=FP8Format.E5M2, fp8_range=FP8Format.E5M2.value.max_fwd
    )
    assert tri_out.dtype == torch.float8_e5m2
    assert torch.equal(tri_out.to(torch.float32), ref_out.to(torch.float32))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_determinism():
    x = torch.randn((64, 128), device="cuda", dtype=torch.float32)
    out1, _ = fp8_quant_triton(x)
    out2, _ = fp8_quant_triton(x)
    assert torch.equal(out1.to(torch.float32), out2.to(torch.float32))
