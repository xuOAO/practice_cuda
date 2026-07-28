from __future__ import annotations

import argparse
from typing import Iterable

import torch

from fp8_bench.cases import BMM_SUITES, BMMCase
from fp8_bench.registry import BMM_IMPLS, get_bmm, get_quant, load_builtin_impls
from fp8_bench.utils import (
    accuracy_metrics,
    append_jsonl,
    benchmark_cuda,
    parse_dtype,
    print_perf,
    result_record,
    seed_everything,
)


def _cases(suite: str, case_name: str | None) -> Iterable[BMMCase]:
    cases = BMM_SUITES[suite]
    if case_name is None:
        return cases
    selected = [case for case in cases if case.name == case_name]
    if not selected:
        raise ValueError(f"case {case_name!r} is not in suite {suite!r}")
    return selected


def main() -> None:
    load_builtin_impls()
    parser = argparse.ArgumentParser(description="Benchmark FP8 BMM kernels and pipelines.")
    parser.add_argument("--suite", choices=sorted(BMM_SUITES), default="smoke")
    parser.add_argument("--case", help="Run one named case from the selected suite.")
    parser.add_argument(
        "--impl",
        action="append",
        choices=sorted(BMM_IMPLS),
        help="BMM implementation; repeat to select several. Default: all.",
    )
    parser.add_argument("--mode", choices=["perf", "accuracy", "both"], default="both")
    parser.add_argument("--input-dtype", default="bf16")
    parser.add_argument("--out-dtype", default="bf16")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default="results/bmm.jsonl")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    input_dtype = parse_dtype(args.input_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    impl_names = args.impl or sorted(BMM_IMPLS)

    for case in _cases(args.suite, args.case):
        batch, m, n, k = case.shape
        a = torch.randn((batch, m, k), device="cuda", dtype=input_dtype)
        b = torch.randn((batch, k, n), device="cuda", dtype=input_dtype)
        bias = (
            torch.randn((batch, m, n), device="cuda", dtype=out_dtype)
            if args.bias
            else None
        )

        for impl_name in impl_names:
            bmm_impl = get_bmm(impl_name)
            quant_impl = get_quant(bmm_impl.quant_impl)
            qa = quant_impl.fn(
                a,
                fp8_dtype=fp8_dtype,
                **bmm_impl.quant_a_kwargs,
            )
            qb = bmm_impl.prepare_b(
                quant_impl.fn(
                    b,
                    fp8_dtype=fp8_dtype,
                    **bmm_impl.quant_b_kwargs,
                )
            )
            call_kwargs = bmm_impl.prepare_call_kwargs(qa, qb)
            out = torch.empty((batch, m, n), device="cuda", dtype=out_dtype)

            values = {
                "suite": args.suite,
                "case": case.name,
                "shape": case.shape,
                "impl": impl_name,
                "quant_impl": bmm_impl.quant_impl,
                "quant_a_kwargs": bmm_impl.quant_a_kwargs,
                "quant_b_kwargs": bmm_impl.quant_b_kwargs,
                "layout": bmm_impl.layout,
                "input_dtype": str(input_dtype),
                "out_dtype": str(out_dtype),
                "fp8_dtype": str(fp8_dtype),
                "bias": args.bias,
                "seed": args.seed,
            }
            flops = 2 * batch * m * n * k

            if args.mode in {"perf", "both"}:
                kernel_perf = benchmark_cuda(
                    lambda: bmm_impl.fn(
                        qa,
                        qb,
                        out_dtype=out_dtype,
                        bias=bias,
                        out=out,
                        **call_kwargs,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                kernel_perf["tflops"] = flops / kernel_perf["median_ms"] / 1e9

                def run_pipeline() -> torch.Tensor:
                    run_qa = quant_impl.fn(
                        a,
                        fp8_dtype=fp8_dtype,
                        **bmm_impl.quant_a_kwargs,
                    )
                    run_qb = bmm_impl.prepare_b(
                        quant_impl.fn(
                            b,
                            fp8_dtype=fp8_dtype,
                            **bmm_impl.quant_b_kwargs,
                        )
                    )
                    return bmm_impl.fn(
                        run_qa,
                        run_qb,
                        out_dtype=out_dtype,
                        bias=bias,
                        out=out,
                    )

                pipeline_perf = benchmark_cuda(
                    run_pipeline,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                pipeline_perf["tflops"] = flops / pipeline_perf["median_ms"] / 1e9
                values["bmm_performance"] = kernel_perf
                values["pipeline_performance"] = pipeline_perf
                # Keep these at the top level as well so JSONL consumers do
                # not have to understand the timing payload structure.
                values["bmm_tflops"] = kernel_perf["tflops"]
                values["pipeline_tflops"] = pipeline_perf["tflops"]
                print_perf(
                    f"bmm-only {case.name} [{impl_name}]",
                    kernel_perf,
                    extra=f"  {kernel_perf['tflops']:.2f} TFLOPS",
                )
                print_perf(
                    f"pipeline {case.name} [{impl_name}]",
                    pipeline_perf,
                    extra=f"  {pipeline_perf['tflops']:.2f} TFLOPS",
                )

            if args.mode in {"accuracy", "both"}:
                actual = bmm_impl.fn(
                    qa,
                    qb,
                    out_dtype=out_dtype,
                    bias=bias,
                    out=out,
                    **call_kwargs,
                )
                dequant_a = qa.dequantize()
                kernel_reference = torch.bmm(dequant_a, qb.dequantize())
                pipeline_reference = torch.bmm(a.float(), b.float())
                if bias is not None:
                    kernel_reference += bias.float()
                    pipeline_reference += bias.float()

                kernel_metrics = accuracy_metrics(actual, kernel_reference)
                pipeline_metrics = accuracy_metrics(actual, pipeline_reference)
                values["kernel_accuracy"] = kernel_metrics
                values["pipeline_accuracy"] = pipeline_metrics
                print(
                    f"accuracy {case.name} [{impl_name}]"
                    f"  kernel_rel_l2={kernel_metrics['rel_l2']:.6g}"
                    f" pipeline_rel_l2={pipeline_metrics['rel_l2']:.6g}"
                    f" pipeline_mse={pipeline_metrics['mse']:.6g}"
                    f" cosine={pipeline_metrics['cosine']:.6g}"
                )

            append_jsonl(args.results, result_record("bmm", **values))


if __name__ == "__main__":
    main()
