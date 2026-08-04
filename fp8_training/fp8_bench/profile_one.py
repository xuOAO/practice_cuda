from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from fp8_bench.cases import find_bmm_case, find_quant_case
from fp8_bench.registry import (
    BMM_IMPLS,
    QUANT_IMPLS,
    bmm_impl_names,
    get_bmm,
    get_quant,
    load_builtin_impls,
)
from fp8_bench.utils import parse_dtype, seed_everything


_PROFILE_NVTX_RANGE = "fp8_bench_profile"


def _profile_once(function: Callable[[], None]) -> None:
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(_PROFILE_NVTX_RANGE)
    try:
        function()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _bmm_quant_kwargs(
    impl,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    a_kwargs = dict(impl.quant_a_kwargs)
    b_kwargs = dict(impl.quant_b_kwargs)
    if not impl.quant_impl.startswith("triton_per_block_"):
        return a_kwargs, b_kwargs

    if args.quant_block_m is not None:
        a_kwargs["block_m"] = args.quant_block_m
    if args.quant_block_k is not None:
        a_kwargs["block_n"] = args.quant_block_k
        b_kwargs["block_m"] = args.quant_block_k
    if args.quant_block_n is not None:
        b_kwargs["block_n"] = args.quant_block_n
    return a_kwargs, b_kwargs


def profile_quant(args: argparse.Namespace) -> None:
    case = find_quant_case(args.case)
    impl = get_quant(args.impl)
    x = torch.randn(case.shape, device="cuda", dtype=parse_dtype(args.input_dtype))
    fp8_dtype = parse_dtype(args.fp8_dtype)
    quant_kwargs = {}
    if args.impl.startswith("triton_per_block_"):
        default_block_m = 1 if args.impl.endswith("_1d") else 128
        quant_kwargs = {
            "block_m": (
                args.block_m if args.block_m is not None else default_block_m
            ),
            "block_n": args.block_n if args.block_n is not None else 128,
        }

    def run_quant() -> None:
        impl.fn(
            x,
            fp8_dtype=fp8_dtype,
            profile=False,
            **quant_kwargs,
        )

    for _ in range(args.warmup):
        run_quant()
    torch.cuda.synchronize()
    _profile_once(run_quant)


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
    if impl.a_layout == "m":
        a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
    if impl.layout == "k":
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
    quant_a_kwargs, quant_b_kwargs = _bmm_quant_kwargs(impl, args)
    qa = impl.prepare_a(
        quant_impl.fn(
            a,
            fp8_dtype=fp8_dtype,
            profile=True,
            **quant_a_kwargs,
        )
    )
    qb = impl.prepare_b(
        quant_impl.fn(
            b,
            fp8_dtype=fp8_dtype,
            profile=True,
            **quant_b_kwargs,
        )
    )
    call_kwargs = impl.prepare_call_kwargs(qa, qb)
    out = torch.empty((batch, m, n), device="cuda", dtype=out_dtype)

    def run_bmm() -> None:
        impl.fn(
            qa,
            qb,
            out_dtype=out_dtype,
            out=out,
            profile=False,
            use_tma=args.backend == "tma",
            **call_kwargs,
        )

    for _ in range(args.warmup):
        run_bmm()
    torch.cuda.synchronize()
    _profile_once(run_bmm)


def main() -> None:
    load_builtin_impls()
    parser = argparse.ArgumentParser(
        description="Launch exactly one post-warmup kernel for Nsight Compute."
    )
    parser.add_argument("--op", choices=["quant", "bmm"], required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument(
        "--backend",
        choices=("block-ptr", "tma"),
        default="block-ptr",
        help="BMM profiling backend. Ignored for quant profiling.",
    )
    parser.add_argument("--input-dtype", default="bf16")
    parser.add_argument("--out-dtype", default="bf16")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--quant-block-m", type=int)
    parser.add_argument("--quant-block-k", type=int)
    parser.add_argument("--quant-block-n", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.warmup < 1:
        raise ValueError(
            "--warmup must be at least 1 so autotuning stays outside NCU"
        )

    choices = QUANT_IMPLS if args.op == "quant" else BMM_IMPLS
    if args.impl not in choices:
        valid_names = sorted(choices) if args.op == "quant" else bmm_impl_names()
        raise ValueError(
            f"unknown {args.op} impl {args.impl}; choices={valid_names}"
        )
    seed_everything(args.seed)
    if args.op == "quant":
        profile_quant(args)
    else:
        profile_bmm(args)
    print(f"profile target completed: op={args.op} case={args.case} impl={args.impl}")


if __name__ == "__main__":
    main()
