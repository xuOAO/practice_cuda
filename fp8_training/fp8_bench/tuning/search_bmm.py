from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from fp8_bench.cases import BMMCase, BMM_SUITES, find_bmm_case
from fp8_bench.tuning.adapters import ADAPTERS, TuningSpec, get_adapter
from fp8_bench.tuning.compile_filter import (
    CompileResult,
    DeviceLimits,
    compile_filter,
    tile_efficiency,
    wave_metrics,
)
from fp8_bench.tuning.space import KernelConfig, SpacePolicy, generate_configs
from fp8_bench.utils import (
    append_jsonl,
    benchmark_cuda,
    environment_info,
    parse_dtype,
)


@dataclass
class TimedConfig:
    compile_result: CompileResult
    perf: dict[str, Any]
    tflops: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.compile_result.config.as_dict(),
            "compile": self.compile_result.as_dict(),
            "perf": self.perf,
            "tflops": self.tflops,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, compile-filter and benchmark Triton FP8 BMM configs."
        )
    )
    parser.add_argument("--impl", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--suite", default="legacy", choices=sorted(BMM_SUITES))
    parser.add_argument(
        "--case",
        action="append",
        help="Run one named case; repeat this flag to select several cases.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fp8-dtype", default="e4m3", choices=("e4m3", "e5m2"))
    parser.add_argument(
        "--out-dtype",
        default="bf16",
        choices=("fp16", "bf16", "fp32"),
    )
    parser.add_argument("--bias", action="store_true")

    parser.add_argument("--quant-block-m", type=int, default=128)
    parser.add_argument("--quant-block-k", type=int, default=128)
    parser.add_argument("--quant-block-n", type=int, default=128)

    parser.add_argument("--block-m-min", type=int, default=64)
    parser.add_argument("--block-n-min", type=int, default=8)
    parser.add_argument("--block-k-min", type=int, default=32)
    parser.add_argument("--block-m-cap", type=int, default=256)
    parser.add_argument("--block-n-cap", type=int, default=256)
    parser.add_argument("--block-k-cap", type=int, default=256)
    parser.add_argument("--group-m", type=int, nargs="+", default=[8])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--num-stages", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument(
        "--static-resource-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply conservative register/shared-memory lower-bound filters.",
    )
    parser.add_argument(
        "--reject-local-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject kernels whose loaded cubin reports spills/local memory.",
    )
    parser.add_argument(
        "--require-wgmma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only kernels whose generated assembly contains WGMMA/HGMMA.",
    )
    parser.add_argument(
        "--prebench-top-k",
        type=int,
        default=64,
        help=(
            "Benchmark only this many resource-valid configs per shape after "
            "tile/wave heuristic ranking; use 0 to benchmark all."
        ),
    )
    parser.add_argument("--quick-top-k", type=int, default=8)
    parser.add_argument("--quick-warmup", type=int, default=5)
    parser.add_argument("--quick-iters", type=int, default=20)
    parser.add_argument("--quick-repeats", type=int, default=2)
    parser.add_argument("--final-warmup", type=int, default=20)
    parser.add_argument("--final-iters", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        help=(
            "JSONL path. Defaults to "
            "results/tuning/<impl>_<suite>.jsonl."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing JSONL instead of replacing it.",
    )
    return parser.parse_args()


def _selected_cases(args: argparse.Namespace) -> list[BMMCase]:
    if args.case:
        return [find_bmm_case(name) for name in args.case]
    return BMM_SUITES[args.suite]


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "quant_block_m",
        "quant_block_k",
        "quant_block_n",
        "prebench_top_k",
        "quick_top_k",
        "quick_warmup",
        "quick_iters",
        "quick_repeats",
        "final_warmup",
        "final_iters",
        "final_repeats",
    )
    for name in positive_names:
        value = getattr(args, name)
        allow_zero = {"prebench_top_k", "quick_warmup", "final_warmup"}
        minimum = 0 if name in allow_zero else 1
        if value < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")


def _policy(args: argparse.Namespace, limits: DeviceLimits) -> SpacePolicy:
    return SpacePolicy(
        block_m_min=args.block_m_min,
        block_n_min=args.block_n_min,
        block_k_min=args.block_k_min,
        block_m_cap=args.block_m_cap,
        block_n_cap=args.block_n_cap,
        block_k_cap=args.block_k_cap,
        group_ms=tuple(args.group_m),
        num_warps=tuple(args.num_warps),
        num_stages=tuple(args.num_stages),
        warp_size=limits.warp_size,
        register_limit_per_thread=limits.register_limit_per_thread,
    )


def _spec(
    args: argparse.Namespace,
    case: BMMCase,
    *,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype,
    per_block: bool,
) -> TuningSpec:
    return TuningSpec(
        impl=args.impl,
        case=case.name,
        batch=case.batch,
        m=case.m,
        n=case.n,
        k=case.k,
        fp8_dtype=fp8_dtype,
        out_dtype=out_dtype,
        use_bias=args.bias,
        quant_block_m=args.quant_block_m if per_block else None,
        quant_block_k=args.quant_block_k if per_block else None,
        quant_block_n=args.quant_block_n if per_block else None,
    )


def _heuristic_rank(
    results: list[CompileResult],
    spec: TuningSpec,
    limits: DeviceLimits,
) -> list[tuple[CompileResult, dict[str, float | None]]]:
    ranked: list[tuple[CompileResult, dict[str, float | None]]] = []
    for result in results:
        sm_fill, wave_efficiency = wave_metrics(result, spec, limits)
        tile = tile_efficiency(result.config, spec)
        score = tile * (wave_efficiency if wave_efficiency is not None else 1.0)
        ranked.append(
            (
                result,
                {
                    "tile_efficiency": tile,
                    "sm_fill": sm_fill,
                    "wave_efficiency": wave_efficiency,
                    "heuristic_score": score,
                },
            )
        )
    ranked.sort(key=lambda item: float(item[1]["heuristic_score"]), reverse=True)
    return ranked


def _time_config(
    function: Callable[[], None],
    result: CompileResult,
    spec: TuningSpec,
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> TimedConfig:
    perf = benchmark_cuda(
        function,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    operations = 2 * spec.batch * spec.m * spec.n * spec.k
    tflops = operations / perf["median_ms"] / 1e9
    return TimedConfig(result, perf, tflops)


def _write_record(path: Path, kind: str, **values: Any) -> None:
    append_jsonl(path, {"kind": kind, **values})


def _failure_counts(results: list[CompileResult]) -> dict[str, int]:
    counts = {
        "compile_failed": 0,
        "resource_or_instruction_rejected": 0,
        "accepted": 0,
    }
    for result in results:
        if not result.compile_ok:
            counts["compile_failed"] += 1
        elif not result.accepted:
            counts["resource_or_instruction_rejected"] += 1
        else:
            counts["accepted"] += 1
    return counts


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    cases = _selected_cases(args)
    torch.cuda.set_device(args.device)
    limits = DeviceLimits.current(args.device)
    adapter = get_adapter(args.impl)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    per_block = adapter.family == "per_block"
    quant_block_k = args.quant_block_k if per_block else None

    policy = _policy(args, limits)
    configs = generate_configs(
        cases,
        policy,
        quant_block_k=quant_block_k,
        max_num_regs=limits.max_num_regs,
        max_shared_mem=limits.max_shared_mem,
        static_resource_filter=args.static_resource_filter,
    )
    if not configs:
        raise RuntimeError("the requested search space contains no configs")

    output = Path(
        args.output
        or f"results/tuning/{args.impl}_{args.suite}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        output.write_text("", encoding="utf-8")
    best_output = output.with_suffix(".best.json")

    environment = environment_info()
    _write_record(
        output,
        "search_start",
        environment=environment,
        device_limits=limits.as_dict(),
        implementation=args.impl,
        suite=args.suite,
        cases=[case.name for case in cases],
        policy={
            **policy.__dict__,
            "group_ms": list(policy.group_ms),
            "num_warps": list(policy.num_warps),
            "num_stages": list(policy.num_stages),
        },
        candidate_count=len(configs),
        require_wgmma=args.require_wgmma,
        reject_local_memory=args.reject_local_memory,
    )

    print(
        f"device={limits.name} cc={limits.compute_capability} "
        f"impl={args.impl} static_candidates={len(configs)}"
    )
    if args.impl.endswith("_n_transpose"):
        print(
            "note: autotuning measures the post-transpose K-major BMM kernel; "
            "the transpose itself is intentionally not part of config search"
        )

    best_by_case: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        spec = _spec(
            args,
            case,
            fp8_dtype=fp8_dtype,
            out_dtype=out_dtype,
            per_block=per_block,
        )
        survivors, compile_results = compile_filter(
            adapter=adapter,
            spec=spec,
            configs=configs,
            limits=limits,
            reject_local_memory=args.reject_local_memory,
            require_wgmma=args.require_wgmma,
        )
        counts = _failure_counts(compile_results)
        print(
            f"[{case_index}/{len(cases)}] {case.name} shape={case.shape}: "
            f"compile_failed={counts['compile_failed']} "
            f"rejected={counts['resource_or_instruction_rejected']} "
            f"survived={counts['accepted']}"
        )
        for result in compile_results:
            _write_record(
                output,
                "compile",
                spec=spec.as_dict(),
                **result.as_dict(),
            )
        if not survivors:
            _write_record(
                output,
                "case_failed",
                spec=spec.as_dict(),
                reason="no config survived compilation/resource filtering",
                counts=counts,
            )
            continue

        ranked = _heuristic_rank(survivors, spec, limits)
        for result, metrics in ranked:
            _write_record(
                output,
                "heuristic",
                spec=spec.as_dict(),
                config=result.config.as_dict(),
                **metrics,
            )
        if args.prebench_top_k:
            ranked = ranked[: args.prebench_top_k]

        try:
            runtime = adapter.create_runtime(spec)
        except Exception as exc:
            _write_record(
                output,
                "case_failed",
                spec=spec.as_dict(),
                reason=f"runtime allocation failed: {type(exc).__name__}: {exc}",
            )
            torch.cuda.empty_cache()
            continue
        quick_results: list[TimedConfig] = []
        for rank, (result, metrics) in enumerate(ranked, start=1):
            launcher = adapter.make_launcher(spec, result.config, runtime)
            try:
                timed = _time_config(
                    launcher,
                    result,
                    spec,
                    warmup=args.quick_warmup,
                    iters=args.quick_iters,
                    repeats=args.quick_repeats,
                )
                quick_results.append(timed)
                _write_record(
                    output,
                    "quick_bench",
                    spec=spec.as_dict(),
                    heuristic_rank=rank,
                    heuristic=metrics,
                    **timed.as_dict(),
                )
            except Exception as exc:
                _write_record(
                    output,
                    "quick_bench_failed",
                    spec=spec.as_dict(),
                    config=result.config.as_dict(),
                    reason=f"{type(exc).__name__}: {exc}",
                )
        if not quick_results:
            _write_record(
                output,
                "case_failed",
                spec=spec.as_dict(),
                reason="all resource-valid configs failed during quick benchmark",
            )
            del runtime
            torch.cuda.empty_cache()
            continue

        quick_results.sort(key=lambda item: item.perf["median_ms"])
        finalists = quick_results[: args.quick_top_k]
        final_results: list[TimedConfig] = []
        for quick_rank, quick in enumerate(finalists, start=1):
            launcher = adapter.make_launcher(
                spec,
                quick.compile_result.config,
                runtime,
            )
            try:
                timed = _time_config(
                    launcher,
                    quick.compile_result,
                    spec,
                    warmup=args.final_warmup,
                    iters=args.final_iters,
                    repeats=args.final_repeats,
                )
                final_results.append(timed)
                _write_record(
                    output,
                    "final_bench",
                    spec=spec.as_dict(),
                    quick_rank=quick_rank,
                    quick_median_ms=quick.perf["median_ms"],
                    **timed.as_dict(),
                )
            except Exception as exc:
                _write_record(
                    output,
                    "final_bench_failed",
                    spec=spec.as_dict(),
                    config=quick.compile_result.config.as_dict(),
                    reason=f"{type(exc).__name__}: {exc}",
                )
        if not final_results:
            _write_record(
                output,
                "case_failed",
                spec=spec.as_dict(),
                reason="all finalists failed during final benchmark",
            )
            del runtime
            torch.cuda.empty_cache()
            continue

        final_results.sort(key=lambda item: item.perf["median_ms"])
        best = final_results[0]
        best_record = {
            "case": case.name,
            "shape": list(case.shape),
            **best.as_dict(),
        }
        best_by_case.append(best_record)
        print(
            f"  best {best.compile_result.config.format_triton()} "
            f"{best.perf['median_ms']:.4f} ms {best.tflops:.2f} TFLOPS"
        )
        del runtime
        torch.cuda.empty_cache()

    unique_configs = sorted(
        {
            KernelConfig(**record["config"])
            for record in best_by_case
        }
    )
    summary = {
        "environment": environment,
        "device_limits": limits.as_dict(),
        "implementation": args.impl,
        "suite": args.suite,
        "best_by_case": best_by_case,
        "unique_configs": [config.as_dict() for config in unique_configs],
        "triton_configs": [
            config.format_triton()
            for config in unique_configs
        ],
    }
    best_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_record(
        output,
        "search_end",
        successful_cases=len(best_by_case),
        requested_cases=len(cases),
        best_output=str(best_output),
    )
    median_tflops = (
        statistics.median(record["tflops"] for record in best_by_case)
        if best_by_case
        else 0.0
    )
    print(
        f"done successful_cases={len(best_by_case)}/{len(cases)} "
        f"median_best={median_tflops:.2f} TFLOPS"
    )
    print(f"records: {output}")
    print(f"best configs: {best_output}")


if __name__ == "__main__":
    main()
