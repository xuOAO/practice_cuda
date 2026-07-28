import torch
import triton
from functools import partial


import os
import sys

_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

from utils import time_consumption
from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel

data_shapes = [
    # B, M, N, K
    (16, 512, 960, 1280),
    (16, 2048, 640, 1280),
    (16, 2048, 1280, 1280),
    (16, 2048, 1280, 960),
    (32, 2048, 640, 1280),
    (32, 2048, 960, 1600),
    (32, 2048, 1280, 1600),
    (32, 2048, 1600, 1600),
    (32, 2048, 1280, 960),
    (80, 512, 640, 1280),
    (80, 2048, 640, 640),
    (80, 2048, 960, 640),
    (80, 2048, 1280, 640),
    (80, 2048, 1280, 960),
    (80, 2048, 640, 1280),
]

@torch.compile
def fp16_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor=None
):
    if bias is not None:
        return torch.bmm(A, B) + bias
    else:
        return torch.bmm(A, B)
    
def fp8_triton_bmm_n_order(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor=None,
):
    Bs, M, K = A.shape
    _, _, N = B.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        Bs,
    )

    if bias is not None:
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride(0), bias.stride(1), bias.stride(2)
        USE_BIASE = 1
    else:
        bias_ptr = None
        stride_biasb = stride_biasm = stride_biasn = 0
        USE_BIASE = 0

    quant_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    batch_quant_fp8_mm_kernel[grid](
        A,
        B,
        C,
        bias_ptr,
        quant_scale,
        M,
        N,
        K,
        # A strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        # B strides
        B.stride(0),
        B.stride(1),
        B.stride(2),
        # C strides
        C.stride(0),
        C.stride(1),
        C.stride(2),
        # bias strides
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIASE,
        B_N_ORDER=True,
    )
    return C

def fp8_triton_bmm_k_order(
    A: torch.Tensor,
    B_t: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor=None,
):
    Bs, M, K = A.shape
    _, N, _ = B_t.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        Bs,
    )

    if bias is not None:
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride(0), bias.stride(1), bias.stride(2)
        USE_BIASE = 1
    else:
        bias_ptr = None
        stride_biasb = stride_biasm = stride_biasn = 0
        USE_BIASE = 0

    quant_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    batch_quant_fp8_mm_kernel[grid](
        A,
        B_t,
        C,
        bias_ptr,
        quant_scale,
        M,
        N,
        K,
        # A strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        # B strides
        B_t.stride(0),
        B_t.stride(2),
        B_t.stride(1),
        # C strides
        C.stride(0),
        C.stride(1),
        C.stride(2),
        # bias strides
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIASE,
        B_N_ORDER=False,
    )
    return C

def do_bench(use_bias: bool=False):
    for shape in data_shapes:
        B, M, N, K = shape
        A_fp16 = torch.randn((B, M, K), device="cuda", dtype=torch.float16)
        B_fp16 = torch.randn((B, K, N), device="cuda", dtype=torch.float16)
        A_fp8 = A_fp16.to(torch.float8_e4m3fn)
        B_fp8 = B_fp16.to(torch.float8_e4m3fn)
        B_fp8_t = B_fp8.transpose(1, 2).contiguous()
        C_fp32 = torch.empty((B, M, N), device="cuda", dtype=torch.float32)
        if use_bias:
            bias = torch.rand((B, M, N), device="cuda", dtype=torch.float32)

        total_flops = 2 * B * M * N * K
        torch_time = time_consumption(fp16_bmm, A_fp16, B_fp16, bias=bias if use_bias else None)
        triton_n_order_time = time_consumption(
            fp8_triton_bmm_n_order, A_fp8, B_fp8, C_fp32, bias=bias if use_bias else None
        )
        triton_k_order_time = time_consumption(
            fp8_triton_bmm_k_order, A_fp8, B_fp8_t, C_fp32, bias=bias if use_bias else None
        )
        # time 单位为 ms: TFLOPS = total_flops / (time_ms / 1000) / 1e12
        torch_tflops = total_flops / torch_time / 1e9
        triton_n_order_tflops = total_flops / triton_n_order_time / 1e9
        triton_k_order_tflops = total_flops / triton_k_order_time / 1e9
        print(f"shape: {shape}")
        print(f"  torch:  {torch_time:.4f} ms / {torch_tflops:.2f} TFLOPS")
        print(f"  triton(n): {triton_n_order_time:.4f} ms / {triton_n_order_tflops:.2f} TFLOPS")
        print(f"  triton(k): {triton_k_order_time:.4f} ms / {triton_k_order_tflops:.2f} TFLOPS")

if __name__ == "__main__":
    do_bench()