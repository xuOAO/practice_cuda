from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch


DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name}; choices={sorted(DTYPES)}") from exc


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def benchmark_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int = 20,
    iters: int = 100,
    repeats: int = 5,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / iters)

    mean = statistics.fmean(samples_ms)
    stdev = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "median_ms": statistics.median(samples_ms),
        "p20_ms": percentile(samples_ms, 0.2),
        "p80_ms": percentile(samples_ms, 0.8),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "cv": stdev / mean if mean else 0.0,
        "samples_ms": samples_ms,
    }


@torch.no_grad()
def accuracy_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    finite = torch.isfinite(actual_f)
    diff = actual_f - expected_f
    squared_diff = diff.square()
    mse = squared_diff.mean()
    expected_norm = torch.linalg.vector_norm(expected_f)
    diff_norm = torch.linalg.vector_norm(diff)
    denom = torch.clamp(expected_norm, min=1e-12)
    cosine_denom = torch.clamp(
        torch.linalg.vector_norm(actual_f) * expected_norm,
        min=1e-12,
    )
    return {
        "finite": bool(finite.all().item()),
        "nan_count": int(torch.isnan(actual_f).sum().item()),
        "inf_count": int(torch.isinf(actual_f).sum().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "mse": float(mse.item()),
        "rmse": float(mse.sqrt().item()),
        "rel_l2": float((diff_norm / denom).item()),
        "cosine": float((torch.sum(actual_f * expected_f) / cosine_denom).item()),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device": device,
                "gpu": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "gpu_memory_bytes": props.total_memory,
            }
        )
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def append_jsonl(path: str | os.PathLike[str], record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def result_record(kind: str, **values: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "environment": environment_info(),
        **values,
    }
