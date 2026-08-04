"""Fixed-shape FP8 BMM layout variants for compact NCU source views."""

from __future__ import annotations

import argparse
from typing import Optional

import torch
import triton
import triton.language as tl


M = 64
N = 64
K = 128


def tma_alloc(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def fixed_a_k_b_k_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(64, 128),
        strides=(128, 1),
        offsets=(0, 0),
        block_shape=(64, 128),
        order=(1, 0),
    )
    # Logical B[K,N] has K as its contiguous dimension: strides [1, K].
    b_block = tl.make_block_ptr(
        base=b_ptr,
        shape=(128, 64),
        strides=(1, 128),
        offsets=(0, 0),
        block_shape=(128, 64),
        order=(0, 1),
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = tl.load(a_block)
    b = tl.load(b_block)
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def fixed_a_k_b_k_tma_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=[64, 128],
        strides=[128, 1],
        block_shape=[64, 128],
    )
    # Fetch logical B[K,N] as contiguous-last [N,K], then transpose.
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[64, 128],
        strides=[128, 1],
        block_shape=[64, 128],
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = a_desc.load([0, 0])
    b = tl.trans(b_desc.load([0, 0]))
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def fixed_a_k_b_n_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(64, 128),
        strides=(128, 1),
        offsets=(0, 0),
        block_shape=(64, 128),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr,
        shape=(128, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(128, 64),
        order=(1, 0),
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = tl.load(a_block)
    b = tl.load(b_block)
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def fixed_a_k_b_n_tma_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=[64, 128],
        strides=[128, 1],
        block_shape=[64, 128],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[128, 64],
        strides=[64, 1],
        block_shape=[128, 64],
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def fixed_a_m_b_n_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    # Logical A[M,K] has M as its contiguous dimension: strides [1, M].
    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(64, 128),
        strides=(1, 64),
        offsets=(0, 0),
        block_shape=(64, 128),
        order=(0, 1),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr,
        shape=(128, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(128, 64),
        order=(1, 0),
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = tl.load(a_block)
    b = tl.load(b_block)
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def fixed_a_m_b_n_tma_kernel(a_ptr, b_ptr, c_ptr, scale_ptr):
    # Fetch logical A[M,K] as the contiguous-last [K,M] view, then transpose.
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr,
        shape=[128, 64],
        strides=[64, 1],
        block_shape=[128, 64],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[128, 64],
        strides=[64, 1],
        block_shape=[128, 64],
    )
    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(64, 64),
        strides=(64, 1),
        offsets=(0, 0),
        block_shape=(64, 64),
        order=(1, 0),
    )

    a = tl.trans(a_desc.load([0, 0]))
    b = b_desc.load([0, 0])
    accumulator = tl.dot(a, b)
    accumulator *= tl.load(scale_ptr)
    tl.store(c_block, accumulator.to(c_ptr.dtype.element_ty))


KERNELS = {
    "a_k_b_k": fixed_a_k_b_k_kernel,
    "a_k_b_k_tma": fixed_a_k_b_k_tma_kernel,
    "a_k_b_n": fixed_a_k_b_n_kernel,
    "a_k_b_n_tma": fixed_a_k_b_n_tma_kernel,
    "a_m_b_n": fixed_a_m_b_n_kernel,
    "a_m_b_n_tma": fixed_a_m_b_n_tma_kernel,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(KERNELS), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.variant.startswith("a_m"):
        a_storage = (
            torch.randn((K, M), device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
        a = a_storage.transpose(0, 1)
    else:
        a = (
            torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
    if "_b_k" in args.variant:
        b_storage = (
            torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
        b = b_storage.transpose(0, 1)
    else:
        b = (
            torch.randn((K, N), device="cuda", dtype=torch.bfloat16) * 0.25
        ).to(torch.float8_e4m3fn)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)

    if args.variant.endswith("_tma"):
        triton.set_allocator(tma_alloc)
    kernel = KERNELS[args.variant]

    def launch() -> None:
        kernel[(1,)](
            a,
            b,
            out,
            scale,
            num_warps=4,
            num_stages=1,
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()

    reference = (a.float() @ b.float()) * scale
    torch.testing.assert_close(
        out.float(),
        reference.to(torch.bfloat16).float(),
        rtol=2e-2,
        atol=2e-2,
    )
    print(
        f"fixed_{args.variant}_kernel completed: "
        "B=1 M=64 N=64 K=128 BM=64 BN=64 BK=128 warps=4 stages=1"
    )


if __name__ == "__main__":
    main()
