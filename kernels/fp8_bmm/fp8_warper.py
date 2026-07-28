import torch
import triton
from fp8_utils import FP8Format
from kernel.fp8_quant_kernel import batch_per_matrix_fp8_quant_kernel as fp8_quant_kernel
from kernel.fp8_quant_kernel import batch_per_matrix_fp8_quant_kernel_test as fp8_quant_kernel_test
from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel_test as fp8_quant_bmm_kernel_test
from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel_test as fp8_quant_bmm_kernel

def fp8_quant_triton(
    x: torch.Tensor,
    quant_dtype=FP8Format.E4M3.value.dtype_fwd,
    fp8_range=FP8Format.E4M3.value.max_fwd,
    eps=1e-12,
    is_test=False
):
    kl = fp8_quant_kernel if not is_test else fp8_quant_kernel_test
    # Allow callers to pass either an FP8Format enum member or a raw torch.dtype.
    if isinstance(quant_dtype, FP8Format):
        quant_dtype = quant_dtype.value.dtype_fwd

    x_min, x_max = x.aminmax()
    x_max_abs = torch.maximum(x_min.abs(), x_max.abs())
    x_max_abs = x_max_abs.clamp(min=eps)
    quant_factor = fp8_range / x_max_abs
    reciprocal_factor = quant_factor.reciprocal()
    quanted_x = torch.empty_like(x, dtype=quant_dtype)
    
    DIM = x.dim()

    if DIM == 2:
        M, N = x.shape
        grid = lambda meta : (
             triton.cdiv(M, meta["BLOCK_SIZE_M"])
             * triton.cdiv(N, meta["BLOCK_SIZE_N"])
             ,
        )
        kl[grid](
            x,
            quanted_x,
            M,
            N,
            None,
            x.stride(0),
            x.stride(1),
            None,
            quanted_x.stride(0),
            quanted_x.stride(1),
            quant_factor,
            fp8_range=fp8_range,
            DIM=DIM,
        )
    elif DIM == 3:
        B, M, N = x.shape
        grid = lambda meta : (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            B,
        )
        kl[grid](
            x,
            quanted_x,
            M,
            N,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            quanted_x.stride(0),
            quanted_x.stride(1),
            quanted_x.stride(2),
            quant_factor,
            fp8_range=fp8_range,
            DIM=DIM,
        )
    else:
        raise RuntimeError("dim must eq 2 or 3")

    return quanted_x, reciprocal_factor

def fp8_bmm_quant_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor=None,
    *,
    quant_scale: torch.Tensor,
    out_dtype: torch.dtype=torch.float16,
    is_test=False
):
    kl = fp8_quant_bmm_kernel if not is_test else fp8_quant_bmm_kernel_test
    Bs, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((Bs, M, N), dtype=out_dtype, device=A.device)

    if bias is not None:
        USE_BIASE = 1
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride(0), bias.stride(1), bias.stride(2)
    else:
        USE_BIASE = 0
        bias_ptr = None
        stride_biasb = stride_biasm = stride_biasn = 0

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        Bs,
    )
    kl[grid](
        A,
        B,
        C,
        bias_ptr,
        quant_scale,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIASE,
    )
    return C
    