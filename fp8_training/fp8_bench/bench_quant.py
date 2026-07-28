from __future__ import annotations

import argparse
from typing import Iterable

import torch

from fp8_bench.cases import QUANT_SUITES, QuantCase
from fp8_bench.registry import QUANT_IMPLS, get_quant, load_builtin_impls
from fp8_bench.utils import (
    accuracy_metrics,
    append_jsonl,
    benchmark_cuda,
    parse_dtype,
    print_perf,
    result_record,
    seed_everything,
)


def _cases(suite: str, case_name: str | None) -> Iterable[QuantCase]:
    cases = QUANT_SUITES[suite]
    if case_name is None:
        return cases
    selected = [case for case in cases if case.name == case_name]
    if not selected:
        raise ValueError(f"case {case_name!r} is not in suite {suite!r}")
    return selected


def main() -> None:
    load_builtin_impls()
    parser = argparse.ArgumentParser(description="Benchmark standalone FP8 quantization.")
    parser.add_argument("--suite", choices=sorted(QUANT_SUITES), default="smoke")
    parser.add_argument("--case", help="Run one named case from the selected suite.")
    parser.add_argument(
        "--impl",
        action="append",
        choices=sorted(QUANT_IMPLS),
        help="Implementation to run; repeat the flag to select several. Default: all.",
    )
    parser.add_argument("--mode", choices=["perf", "accuracy", "both"], default="both")
    parser.add_argument("--input-dtype", default="fp32")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default="results/quant.jsonl")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    input_dtype = parse_dtype(args.input_dtype)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    impl_names = args.impl or sorted(QUANT_IMPLS)

    for case in _cases(args.suite, args.case):
        x = torch.randn(case.shape, device="cuda", dtype=input_dtype)
        for impl_name in impl_names:
            impl = get_quant(impl_name)
            values = {
                "suite": args.suite,
                "case": case.name,
                "shape": case.shape,
                "impl": impl_name,
                "input_dtype": str(input_dtype),
                "fp8_dtype": str(fp8_dtype),
                "seed": args.seed,
            }

            if args.mode in {"perf", "both"}:
                perf = benchmark_cuda(
                    lambda: impl.fn(x, fp8_dtype=fp8_dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                # Read input + write FP8 output. Scale traffic is negligible.
                traffic_bytes = x.numel() * (x.element_size() + 1)
                perf["bandwidth_gbps"] = traffic_bytes / perf["median_ms"] / 1e6
                values["performance"] = perf
                print_perf(
                    f"quant {case.name} [{impl_name}]",
                    perf,
                    extra=f"  {perf['bandwidth_gbps']:.1f} GB/s",
                )

            if args.mode in {"accuracy", "both"}:
                result = impl.fn(x, fp8_dtype=fp8_dtype)
                dequant = result.dequantize()
                metrics = accuracy_metrics(dequant, x)
                fp8_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
                metrics["saturation_ratio"] = float(
                    (result.tensor.float().abs() == fp8_max).float().mean().item()
                )
                metrics["inv_scale"] = float(result.inv_scale.item())
                values["accuracy"] = metrics
                print(
                    f"accuracy {case.name} [{impl_name}]"
                    f"  rel_l2={metrics['rel_l2']:.6g}"
                    f" cosine={metrics['cosine']:.6g}"
                    f" mse={metrics['mse']:.6g}"
                    f" max_abs={metrics['max_abs']:.6g}"
                )

            append_jsonl(args.results, result_record("quant", **values))


if __name__ == "__main__":
    main()
