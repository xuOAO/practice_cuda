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
    parser.add_argument(
        "--impl",
        action="append",
        required=True,
        choices=sorted(ADAPTERS),
        help=(
            "BMM implementation to search; repeat to search several, with each "
            "implementation written to its own <impl>_<suite>.jsonl file "
            "(e.g. --impl triton_per_tensor_a_k_b_n --impl "
            "triton_per_tensor_a_k_b_n_tma)."
        ),
    )
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

    parser.add_argument("--quant-block-m", type=int)
    parser.add_argument("--quant-block-k", type=int)
    parser.add_argument("--quant-block-n", type=int)

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
            "JSONL path for a single implementation. Defaults to "
            "results/tuning/<impl>_<suite>.jsonl; when --impl is repeated, "
            "each implementation is written to its own default path."
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
    if args.output is not None and len(args.impl) != 1:
        raise ValueError(
            "--output requires exactly one --impl; omit --output to write "
            "repeated --impl values to separate <impl>_<suite>.jsonl files"
        )
    positive_names = (
        "quant_block_m",
        "quant_block_k",
        "quant_block_n",
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
        if value is None:
            continue
        allow_zero = {"quick_warmup", "final_warmup"}
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


def _quant_blocks(
    args: argparse.Namespace,
    impl: str,
) -> tuple[int, int, int]:
    one_dimensional = impl.startswith("triton_per_block_1d_")
    defaults = (1, 128, 1) if one_dimensional else (128, 128, 128)
    values = (args.quant_block_m, args.quant_block_k, args.quant_block_n)
    return (
        values[0] if values[0] is not None else defaults[0],
        values[1] if values[1] is not None else defaults[1],
        values[2] if values[2] is not None else defaults[2],
    )


def _spec(
    args: argparse.Namespace,
    case: BMMCase,
    *,
    impl: str,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype,
    per_block: bool,
) -> TuningSpec:
    quant_blocks = _quant_blocks(args, impl) if per_block else (None, None, None)
    return TuningSpec(
        impl=impl,
        case=case.name,
        batch=case.batch,
        m=case.m,
        n=case.n,
        k=case.k,
        fp8_dtype=fp8_dtype,
        out_dtype=out_dtype,
        use_bias=args.bias,
        quant_block_m=quant_blocks[0],
        quant_block_k=quant_blocks[1],
        quant_block_n=quant_blocks[2],
    )


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


def _policy_dict(policy: SpacePolicy) -> dict[str, Any]:
    return {
        **policy.__dict__,
        "group_ms": list(policy.group_ms),
        "num_warps": list(policy.num_warps),
        "num_stages": list(policy.num_stages),
    }


def _search_one_impl(
    args: argparse.Namespace,
    impl: str,
    cases: list[BMMCase],
    limits: DeviceLimits,
    environment: dict[str, Any],
    output: Path,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    adapter = get_adapter(impl)
    per_block = adapter.family == "per_block"
    quant_block_k = _quant_blocks(args, impl)[1] if per_block else None

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
        raise RuntimeError(
            f"impl={impl}: the requested search space contains no configs"
        )

    _write_record(
        output,
        "search_start",
        environment=environment,
        device_limits=limits.as_dict(),
        implementation=impl,
        suite=args.suite,
        cases=[case.name for case in cases],
        policy=_policy_dict(policy),
        candidate_count=len(configs),
        require_wgmma=args.require_wgmma,
        reject_local_memory=args.reject_local_memory,
    )

    print(
        f"device={limits.name} cc={limits.compute_capability} "
        f"impl={impl} static_candidates={len(configs)}"
    )
    if impl.endswith("_n_transpose"):
        print(
            "note: autotuning measures the post-transpose K-major BMM kernel; "
            "the transpose itself is intentionally not part of config search"
        )

    best_by_case: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        spec = _spec(
            args,
            case,
            impl=impl,
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
        for result in survivors:
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
            "impl": impl,
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

    return best_by_case


def _compact_config(config: dict[str, Any]) -> str:
    return (
        f"BM={config['block_m']} BN={config['block_n']} "
        f"BK={config['block_k']} G={config['group_m']} "
        f"w={config['num_warps']} s={config['num_stages']}"
    )


def _print_summary(
    impls: list[str],
    by_impl: dict[str, dict[str, Any]],
    cases: list[BMMCase],
    output: Path,
    best_output: Path,
) -> None:
    width = 78
    print("\n" + "=" * width)
    print("best configs summary")
    print("=" * width)
    for impl in impls:
        payload = by_impl.get(impl, {})
        recs = payload.get("best_by_case", [])
        median_tflops = (
            statistics.median(record["tflops"] for record in recs)
            if recs
            else 0.0
        )
        print(
            f"\n[{impl}] successful_cases={len(recs)}/{len(cases)} "
            f"median_best={median_tflops:.2f} TFLOPS"
        )
        if not recs:
            print("  (no case survived)")
            continue
        case_w = max(len(record["case"]) for record in recs)
        for record in recs:
            cfg = _compact_config(record["config"])
            print(
                f"  {record['case']:<{case_w}}  shape={tuple(record['shape'])}  "
                f"{cfg}  {record['perf']['median_ms']:.4f} ms  "
                f"{record['tflops']:.2f} TFLOPS"
            )
        # Unique best configs across cases, ready to paste into _CONFIGS.
        unique = payload.get("triton_configs", [])
        print(f"  -- {len(unique)} unique config(s), paste into _CONFIGS --")
        print("  _CONFIGS = [")
        for line in unique:
            print(f"      {line},")
        print("  ]")
    print(f"\nrecords: {output}")
    print(f"best configs: {best_output}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    cases = _selected_cases(args)
    torch.cuda.set_device(args.device)
    limits = DeviceLimits.current(args.device)
    fp8_dtype = parse_dtype(args.fp8_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    impls = args.impl

    environment = environment_info()
    for impl in impls:
        output = Path(
            args.output or f"results/tuning/{impl}_{args.suite}.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not args.append:
            output.write_text("", encoding="utf-8")
        best_output = output.with_suffix(".best.json")

        best_by_case = _search_one_impl(
            args,
            impl,
            cases,
            limits,
            environment,
            output,
            fp8_dtype,
            out_dtype,
        )

        payload: dict[str, Any] = {
            "best_by_case": best_by_case,
            "unique_configs": [],
            "triton_configs": [],
        }
        unique_configs = sorted(
            {KernelConfig(**r["config"]) for r in payload["best_by_case"]}
        )
        payload["unique_configs"] = [config.as_dict() for config in unique_configs]
        payload["triton_configs"] = [
            config.format_triton() for config in unique_configs
        ]

        by_impl = {impl: payload}
        summary = {
            "environment": environment,
            "device_limits": limits.as_dict(),
            "implementations": [impl],
            "suite": args.suite,
            "best_by_case": best_by_case,
            "by_impl": by_impl,
        }
        best_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_record(
            output,
            "search_end",
            implementations=[impl],
            successful_cases=len(best_by_case),
            requested_cases=len(cases),
            best_output=str(best_output),
        )

        recs = payload["best_by_case"]
        median_tflops = (
            statistics.median(record["tflops"] for record in recs)
            if recs
            else 0.0
        )
        print(
            f"impl={impl} successful_cases={len(recs)}/{len(cases)} "
            f"median_best={median_tflops:.2f} TFLOPS"
        )
        print(f"records: {output}")
        print(f"best configs: {best_output}")

        _print_summary([impl], by_impl, cases, output, best_output)


if __name__ == "__main__":
    main()
