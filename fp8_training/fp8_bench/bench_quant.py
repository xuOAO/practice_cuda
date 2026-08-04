from __future__ import annotations

import argparse
from typing import Iterable

import torch

from fp8_bench.cases import QUANT_SUITES, QuantCase
from fp8_bench.reporting import (
    format_triton_config,
    print_progress,
    print_records,
    triton_config_dict,
)
from fp8_bench.registry import QUANT_IMPLS, get_quant, load_builtin_impls
from fp8_bench.utils import (
    accuracy_metrics,
    append_jsonl,
    benchmark_cuda,
    parse_dtype,
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
    parser.add_argument(
        "--channel-axis",
        type=int,
        choices=[-2, -1],
        default=-1,
        help="Channel axis for per-channel implementations (default: -1).",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        help="Override the row block size for per-block quantization.",
    )
    parser.add_argument(
        "--block-n",
        type=int,
        help="Override the column block size for per-block quantization.",
    )
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
    cases = list(_cases(args.suite, args.case))
    total_runs = len(cases) * len(impl_names)
    completed_runs = 0
    records: list[dict[str, object]] = []

    for case in cases:
        x = torch.randn(case.shape, device="cuda", dtype=input_dtype)
        for impl_name in impl_names:
            impl = get_quant(impl_name)
            quant_kwargs = {}
            if "per_channel" in impl_name:
                quant_kwargs["channel_axis"] = args.channel_axis
            if impl_name.startswith("triton_per_block_"):
                default_block_m = 1 if impl_name.endswith("_1d") else 128
                quant_kwargs = {
                    "block_m": (
                        args.block_m
                        if args.block_m is not None
                        else default_block_m
                    ),
                    "block_n": args.block_n if args.block_n is not None else 128,
                }
            values = {
                "suite": args.suite,
                "case": case.name,
                "shape": case.shape,
                "impl": impl_name,
                "input_dtype": str(input_dtype),
                "fp8_dtype": str(fp8_dtype),
                "quant_kwargs": quant_kwargs,
                "seed": args.seed,
            }

            if args.mode in {"perf", "both"}:
                perf = benchmark_cuda(
                    lambda: impl.fn(
                        x,
                        fp8_dtype=fp8_dtype,
                        **quant_kwargs,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                # Read input + write FP8 output. Scale traffic is negligible.
                traffic_bytes = x.numel() * (x.element_size() + 1)
                perf["bandwidth_gbps"] = traffic_bytes / perf["median_ms"] / 1e6
                values["performance"] = perf

            if args.mode in {"accuracy", "both"}:
                result = impl.fn(
                    x,
                    fp8_dtype=fp8_dtype,
                    **quant_kwargs,
                )
                dequant = result.dequantize()
                metrics = accuracy_metrics(dequant, x)
                fp8_max = torch.finfo(fp8_dtype).max
                metrics["saturation_ratio"] = float(
                    (result.tensor.float().abs() == fp8_max).float().mean().item()
                )
                scale = result.dequant_scale.float()
                if scale.numel() == 1:
                    metrics["dequant_scale"] = float(scale.item())
                else:
                    metrics["dequant_scale_min"] = float(scale.min().item())
                    metrics["dequant_scale_max"] = float(scale.max().item())
                    metrics["dequant_scale_mean"] = float(scale.mean().item())
                values["accuracy"] = metrics

            best_config = impl.get_best_config()
            if best_config is not None:
                values["best_config"] = triton_config_dict(best_config)

            record = result_record("quant", **values)
            append_jsonl(args.results, record)
            records.append(record)
            completed_runs += 1
            performance_parts = []
            if perf := values.get("performance"):
                performance_parts.append(
                    f"duration={perf['median_ms']:.4f} ms"
                )
            if config_text := format_triton_config(best_config):
                performance_parts.append(f"config={config_text}")
            print_progress(
                completed_runs,
                total_runs,
                f"{case.name} [{impl_name}]",
                performance="  ".join(performance_parts) or None,
            )

    print_records(records, title="Quant summary")


if __name__ == "__main__":
    main()
