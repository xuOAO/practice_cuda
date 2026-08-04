from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from fp8_bench.cases import BMM_SUITES, BMMCase
from fp8_bench.reporting import (
    format_triton_config,
    print_progress,
    print_records,
    triton_config_dict,
)
from fp8_bench.registry import bmm_impl_names, get_bmm, get_quant, load_builtin_impls
from fp8_bench.utils import (
    accuracy_metrics,
    append_jsonl,
    benchmark_cuda,
    parse_dtype,
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


def _quant_kwargs(
    impl,
    *,
    quant_block_m: int | None,
    quant_block_k: int | None,
    quant_block_n: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    a_kwargs = dict(impl.quant_a_kwargs)
    b_kwargs = dict(impl.quant_b_kwargs)
    if not impl.quant_impl.startswith("triton_per_block_"):
        return a_kwargs, b_kwargs

    if quant_block_m is not None:
        a_kwargs["block_m"] = quant_block_m
    if quant_block_k is not None:
        a_kwargs["block_n"] = quant_block_k
        b_kwargs["block_m"] = quant_block_k
    if quant_block_n is not None:
        b_kwargs["block_n"] = quant_block_n
    return a_kwargs, b_kwargs


def main() -> None:
    load_builtin_impls()
    parser = argparse.ArgumentParser(description="Benchmark FP8 BMM kernels and pipelines.")
    parser.add_argument("--suite", choices=sorted(BMM_SUITES), default="smoke")
    parser.add_argument("--case", help="Run one named case from the selected suite.")
    parser.add_argument(
        "--impl",
        action="append",
        choices=bmm_impl_names(),
        help="BMM implementation; repeat to select several. Default: all.",
    )
    parser.add_argument("--mode", choices=["perf", "accuracy", "both"], default="both")
    parser.add_argument(
        "--perf-scope",
        choices=("both", "bmm", "pipeline"),
        default="both",
        help=(
            "Performance phases to benchmark when --mode includes perf. "
            "'bmm' measures only the pre-quantized BMM kernel; "
            "'pipeline' includes quantization and BMM. Source layouts are "
            "materialized before timing; explicitly named in-call transpose "
            "variants still include their packing cost."
        ),
    )
    parser.add_argument("--input-dtype", default="bf16")
    parser.add_argument("--out-dtype", default="bf16")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--quant-block-m", type=int)
    parser.add_argument("--quant-block-k", type=int)
    parser.add_argument("--quant-block-n", type=int)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--fp16-baseline",
        action="store_true",
        help=(
            "Also benchmark pure torch.bmm FP16 baselines for a_k_b_k, "
            "a_k_b_n, and a_m_b_n per case. Input conversion, layout "
            "preparation, and output allocation stay outside the timed region."
        ),
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--backend",
        choices=("block-ptr", "tma", "both"),
        help=(
            "Kernel backend to benchmark. 'both' runs block-pointer and TMA "
            "kernels against the same prepared inputs. Default: block-ptr."
        ),
    )
    backend_group.add_argument(
        "--use-tma",
        action="store_true",
        help=(
            "Compatibility alias for --backend tma. TMA requires 16-byte "
            "aligned M/N/K dimensions for fp8."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results",
        default="results/bmm.jsonl",
        help=(
            "Output JSONL path. The file is replaced at the start of each run, "
            "then completed records are flushed incrementally."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    input_dtype = parse_dtype(args.input_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    impl_names = args.impl or bmm_impl_names()
    cases = list(_cases(args.suite, args.case))
    backend_mode = args.backend or ("tma" if args.use_tma else "block-ptr")
    backend_variants = {
        "block-ptr": [("block-ptr", False)],
        "tma": [("tma", True)],
        "both": [("block-ptr", False), ("tma", True)],
    }[backend_mode]
    total_runs = len(cases) * (
        len(impl_names) * len(backend_variants)
        + 3 * int(args.fp16_baseline)
    )
    completed_runs = 0
    records: list[dict[str, object]] = []
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    for case in cases:
        batch, m, n, k = case.shape
        a = torch.randn((batch, m, k), device="cuda", dtype=input_dtype)
        b = torch.randn((batch, k, n), device="cuda", dtype=input_dtype)
        bias = (
            torch.randn((batch, m, n), device="cuda", dtype=out_dtype)
            if args.bias
            else None
        )
        flops = 2 * batch * m * n * k

        # The non-default layouts model tensors produced by a preceding op
        # (for example, backward). Materialize those layouts before timing;
        # quantization preserves the input layout.
        a_inputs = {"k": a}
        b_inputs = {"n": b}

        if args.fp16_baseline:
            baseline_a_k = a.to(torch.float16)
            baseline_b_n = b.to(torch.float16)
            baseline_a_m = baseline_a_k.transpose(-1, -2).contiguous().transpose(
                -1,
                -2,
            )
            baseline_b_k = baseline_b_n.transpose(-1, -2).contiguous().transpose(
                -1,
                -2,
            )
            baseline_bias = bias.to(torch.float16) if bias is not None else None
            baseline_out = torch.empty(
                (batch, m, n),
                device="cuda",
                dtype=torch.float16,
            )
            baseline_variants = (
                ("fp16_a_k_b_k", "k", "k", baseline_a_k, baseline_b_k),
                ("fp16_a_k_b_n", "k", "n", baseline_a_k, baseline_b_n),
                ("fp16_a_m_b_n", "m", "n", baseline_a_m, baseline_b_n),
            )

            for baseline_name, a_layout, b_layout, baseline_a, baseline_b in (
                baseline_variants
            ):
                def run_fp16_baseline() -> torch.Tensor:
                    if baseline_bias is None:
                        return torch.bmm(baseline_a, baseline_b, out=baseline_out)
                    return torch.baddbmm(
                        baseline_bias,
                        baseline_a,
                        baseline_b,
                        out=baseline_out,
                    )

                baseline_values: dict[str, object] = {
                    "suite": args.suite,
                    "case": case.name,
                    "shape": case.shape,
                    "impl": baseline_name,
                    "quant_impl": None,
                    "a_layout": a_layout,
                    "b_layout": b_layout,
                    "layout": b_layout,
                    "backend": "torch",
                    "input_dtype": str(torch.float16),
                    "out_dtype": str(torch.float16),
                    "fp8_dtype": None,
                    "bias": args.bias,
                    "use_tma": False,
                    "perf_scope": "bmm",
                    "seed": args.seed,
                }
                if args.mode in {"perf", "both"}:
                    baseline_perf = benchmark_cuda(
                        run_fp16_baseline,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    baseline_perf["tflops"] = (
                        flops / baseline_perf["median_ms"] / 1e9
                    )
                    baseline_values["bmm_performance"] = baseline_perf
                    baseline_values["bmm_tflops"] = baseline_perf["tflops"]

                if args.mode in {"accuracy", "both"}:
                    baseline_actual = run_fp16_baseline()
                    baseline_kernel_reference = torch.bmm(
                        baseline_a.float(),
                        baseline_b.float(),
                    )
                    baseline_pipeline_reference = torch.bmm(a.float(), b.float())
                    if baseline_bias is not None:
                        baseline_kernel_reference += baseline_bias.float()
                    if bias is not None:
                        baseline_pipeline_reference += bias.float()
                    baseline_values["kernel_accuracy"] = accuracy_metrics(
                        baseline_actual,
                        baseline_kernel_reference,
                    )
                    baseline_values["pipeline_accuracy"] = accuracy_metrics(
                        baseline_actual,
                        baseline_pipeline_reference,
                    )

                baseline_record = result_record("bmm", **baseline_values)
                append_jsonl(results_path, baseline_record)
                records.append(baseline_record)
                completed_runs += 1
                baseline_performance = baseline_values.get("bmm_performance")
                performance_text = None
                if isinstance(baseline_performance, dict):
                    performance_text = (
                        f"bmm={baseline_performance['median_ms']:.4f} ms "
                        f"{baseline_values['bmm_tflops']:.2f} TFLOPS"
                    )
                print_progress(
                    completed_runs,
                    total_runs,
                    f"{case.name} [{baseline_name}] backend=torch",
                    performance=performance_text,
                )

        for impl_name in impl_names:
            bmm_impl = get_bmm(impl_name)
            quant_impl = get_quant(bmm_impl.quant_impl)
            if bmm_impl.a_layout not in a_inputs:
                a_inputs["m"] = a.transpose(-1, -2).contiguous().transpose(
                    -1,
                    -2,
                )
            if bmm_impl.layout not in b_inputs:
                b_inputs["k"] = b.transpose(-1, -2).contiguous().transpose(
                    -1,
                    -2,
                )
            quant_a = a_inputs[bmm_impl.a_layout]
            quant_b = b_inputs[bmm_impl.layout]
            quant_a_kwargs, quant_b_kwargs = _quant_kwargs(
                bmm_impl,
                quant_block_m=args.quant_block_m,
                quant_block_k=args.quant_block_k,
                quant_block_n=args.quant_block_n,
            )
            qa = bmm_impl.prepare_a(
                quant_impl.fn(
                    quant_a,
                    fp8_dtype=fp8_dtype,
                    **quant_a_kwargs,
                )
            )
            qb = bmm_impl.prepare_b(
                quant_impl.fn(
                    quant_b,
                    fp8_dtype=fp8_dtype,
                    **quant_b_kwargs,
                )
            )
            call_kwargs = bmm_impl.prepare_call_kwargs(qa, qb)
            out = torch.empty((batch, m, n), device="cuda", dtype=out_dtype)

            for backend_name, use_tma in backend_variants:
                values = {
                    "suite": args.suite,
                    "case": case.name,
                    "shape": case.shape,
                    "impl": impl_name,
                    "quant_impl": bmm_impl.quant_impl,
                    "quant_a_kwargs": quant_a_kwargs,
                    "quant_b_kwargs": quant_b_kwargs,
                    "a_layout": bmm_impl.a_layout,
                    "b_layout": bmm_impl.layout,
                    "layout": bmm_impl.layout,
                    "backend": backend_name,
                    "input_dtype": str(input_dtype),
                    "out_dtype": str(out_dtype),
                    "fp8_dtype": str(fp8_dtype),
                    "bias": args.bias,
                    "use_tma": use_tma,
                    "perf_scope": args.perf_scope,
                    "seed": args.seed,
                }

                if (
                    args.mode in {"perf", "both"}
                    and args.perf_scope in {"both", "bmm"}
                ):
                    kernel_perf = benchmark_cuda(
                        lambda: bmm_impl.fn(
                            qa,
                            qb,
                            out_dtype=out_dtype,
                            bias=bias,
                            out=out,
                            use_tma=use_tma,
                            **call_kwargs,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    kernel_perf["tflops"] = flops / kernel_perf["median_ms"] / 1e9

                    values["bmm_performance"] = kernel_perf
                    values["bmm_tflops"] = kernel_perf["tflops"]

                if (
                    args.mode in {"perf", "both"}
                    and args.perf_scope in {"both", "pipeline"}
                ):
                    def run_pipeline() -> torch.Tensor:
                        run_qa = bmm_impl.prepare_a(
                            quant_impl.fn(
                                quant_a,
                                fp8_dtype=fp8_dtype,
                                **quant_a_kwargs,
                            )
                        )
                        run_qb = bmm_impl.prepare_b(
                            quant_impl.fn(
                                quant_b,
                                fp8_dtype=fp8_dtype,
                                **quant_b_kwargs,
                            )
                        )
                        run_call_kwargs = bmm_impl.prepare_call_kwargs(
                            run_qa,
                            run_qb,
                        )
                        return bmm_impl.fn(
                            run_qa,
                            run_qb,
                            out_dtype=out_dtype,
                            bias=bias,
                            out=out,
                            use_tma=use_tma,
                            **run_call_kwargs,
                        )

                    pipeline_perf = benchmark_cuda(
                        run_pipeline,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    pipeline_perf["tflops"] = (
                        flops / pipeline_perf["median_ms"] / 1e9
                    )
                    values["pipeline_performance"] = pipeline_perf
                    values["pipeline_tflops"] = pipeline_perf["tflops"]

                if args.mode in {"accuracy", "both"}:
                    actual = bmm_impl.fn(
                        qa,
                        qb,
                        out_dtype=out_dtype,
                        bias=bias,
                        out=out,
                        use_tma=use_tma,
                        **call_kwargs,
                    )
                    dequant_a = qa.dequantize()
                    kernel_reference = torch.bmm(dequant_a, qb.dequantize())
                    pipeline_reference = torch.bmm(a.float(), b.float())
                    if bias is not None:
                        kernel_reference += bias.float()
                        pipeline_reference += bias.float()

                    kernel_metrics = accuracy_metrics(actual, kernel_reference)
                    pipeline_metrics = accuracy_metrics(
                        actual,
                        pipeline_reference,
                    )
                    values["kernel_accuracy"] = kernel_metrics
                    values["pipeline_accuracy"] = pipeline_metrics

                best_config = bmm_impl.get_best_config(use_tma)
                if best_config is not None:
                    values["best_config"] = triton_config_dict(best_config)

                record = result_record("bmm", **values)
                append_jsonl(results_path, record)
                records.append(record)
                completed_runs += 1
                performance_parts = []
                if kernel_perf := values.get("bmm_performance"):
                    performance_parts.append(
                        "bmm="
                        f"{kernel_perf['median_ms']:.4f} ms "
                        f"{values['bmm_tflops']:.2f} TFLOPS"
                    )
                if pipeline_perf := values.get("pipeline_performance"):
                    performance_parts.append(
                        "pipeline="
                        f"{pipeline_perf['median_ms']:.4f} ms "
                        f"{values['pipeline_tflops']:.2f} TFLOPS"
                    )
                if config_text := format_triton_config(best_config):
                    performance_parts.append(f"config={config_text}")
                print_progress(
                    completed_runs,
                    total_runs,
                    f"{case.name} [{impl_name}] backend={backend_name}",
                    performance="  ".join(performance_parts) or None,
                )

    print_records(records, title="BMM summary")


if __name__ == "__main__":
    main()
