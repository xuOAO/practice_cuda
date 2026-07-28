from __future__ import annotations

import argparse

import torch

from fp8_bench.cases import find_bmm_case, find_quant_case
from fp8_bench.registry import (
    BMM_IMPLS,
    QUANT_IMPLS,
    get_bmm,
    get_quant,
    load_builtin_impls,
)
from fp8_bench.utils import parse_dtype, seed_everything


def profile_quant(args: argparse.Namespace) -> None:
    case = find_quant_case(args.case)
    impl = get_quant(args.impl)
    x = torch.randn(case.shape, device="cuda", dtype=parse_dtype(args.input_dtype))
    fp8_dtype = parse_dtype(args.fp8_dtype)
    for _ in range(args.warmup):
        impl.fn(x, fp8_dtype=fp8_dtype, profile=True)
    torch.cuda.synchronize()
    impl.fn(x, fp8_dtype=fp8_dtype, profile=True)
    torch.cuda.synchronize()


def profile_bmm(args: argparse.Namespace) -> None:
    case = find_bmm_case(args.case)
    impl = get_bmm(args.impl)
    quant_impl = get_quant(impl.quant_impl)
    batch, m, n, k = case.shape
    input_dtype = parse_dtype(args.input_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    a = torch.randn((batch, m, k), device="cuda", dtype=input_dtype)
    b = torch.randn((batch, k, n), device="cuda", dtype=input_dtype)
    qa = quant_impl.fn(
        a,
        fp8_dtype=fp8_dtype,
        profile=True,
        **impl.quant_a_kwargs,
    )
    qb = impl.prepare_b(
        quant_impl.fn(
            b,
            fp8_dtype=fp8_dtype,
            profile=True,
            **impl.quant_b_kwargs,
        )
    )
    call_kwargs = impl.prepare_call_kwargs(qa, qb)
    out = torch.empty((batch, m, n), device="cuda", dtype=out_dtype)
    for _ in range(args.warmup):
        impl.fn(
            qa,
            qb,
            out_dtype=out_dtype,
            out=out,
            profile=True,
            **call_kwargs,
        )
    torch.cuda.synchronize()
    impl.fn(
        qa,
        qb,
        out_dtype=out_dtype,
        out=out,
        profile=True,
        **call_kwargs,
    )
    torch.cuda.synchronize()


def main() -> None:
    load_builtin_impls()
    parser = argparse.ArgumentParser(
        description="Launch exactly one post-warmup kernel for Nsight Compute."
    )
    parser.add_argument("--op", choices=["quant", "bmm"], required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--input-dtype", default="bf16")
    parser.add_argument("--out-dtype", default="bf16")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    choices = QUANT_IMPLS if args.op == "quant" else BMM_IMPLS
    if args.impl not in choices:
        raise ValueError(f"unknown {args.op} impl {args.impl}; choices={sorted(choices)}")
    seed_everything(args.seed)
    if args.op == "quant":
        profile_quant(args)
    else:
        profile_bmm(args)
    print(f"profile target completed: op={args.op} case={args.case} impl={args.impl}")


if __name__ == "__main__":
    main()
