from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd



_DEFAULT_COLUMNS = {
    "bmm": (
        "case",
        "impl",
        "backend",
        "a_layout",
        "b_layout",
        "bmm_ms",
        "bmm_p20_ms",
        "bmm_p80_ms",
        "bmm_tflops",
        "best_config",
        "pipeline_ms",
        "pipeline_p20_ms",
        "pipeline_p80_ms",
        "pipeline_tflops",
        "kernel_rel_l2",
        "pipeline_rel_l2",
    ),
    "quant": (
        "case",
        "impl",
        "shape",
        "duration_ms",
        "p20_ms",
        "p80_ms",
        "bandwidth_gbps",
        "best_config",
        "rel_l2",
        "cosine",
        "mse",
        "max_abs",
    ),
    "fsdp2": (
        "impl",
        "world_size",
        "batch_per_rank",
        "seq",
        "duration_ms",
        "samples_per_second",
        "estimated_gemm_tflops",
        "peak_memory_gib",
        "first_measured_loss",
        "last_loss",
    ),
}


def _shape_text(shape: object) -> str:
    if isinstance(shape, (list, tuple)):
        return "x".join(str(value) for value in shape)
    return str(shape)


def _nested(
    record: Mapping[str, Any],
    section: str,
    field: str,
) -> Any:
    value = record.get(section)
    if not isinstance(value, Mapping):
        return None
    return value.get(field)


def triton_config_dict(config: object | None) -> dict[str, Any] | None:
    if config is None:
        return None
    values = dict(getattr(config, "kwargs", {}))
    for field in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
        value = getattr(config, field, None)
        if value is not None:
            values[field] = value
    return values


def format_triton_config(config: object | None) -> str | None:
    if config is None:
        return None
    values = (
        dict(config)
        if isinstance(config, Mapping)
        else triton_config_dict(config)
    )
    if not values:
        return None
    aliases = {
        "BLOCK_M": "BM",
        "BLOCK_N": "BN",
        "BLOCK_K": "BK",
        "GROUP_M": "G",
        "num_warps": "w",
        "num_stages": "s",
        "num_ctas": "ctas",
        "maxnreg": "maxnreg",
    }
    preferred = tuple(aliases)
    ordered = [key for key in preferred if key in values]
    ordered.extend(sorted(key for key in values if key not in aliases))
    return " ".join(f"{aliases.get(key, key)}={values[key]}" for key in ordered)


def record_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(record.get("kind", "unknown"))
    common = {
        "kind": kind,
        "suite": record.get("suite"),
        "case": record.get("case"),
        "shape": _shape_text(record.get("shape")),
        "impl": record.get("impl"),
    }

    if kind == "bmm":
        return {
            **common,
            "backend": record.get("backend"),
            "a_layout": record.get("a_layout"),
            "b_layout": record.get("b_layout", record.get("layout")),
            "bmm_ms": _nested(record, "bmm_performance", "median_ms"),
            "bmm_p20_ms": _nested(record, "bmm_performance", "p20_ms"),
            "bmm_p80_ms": _nested(record, "bmm_performance", "p80_ms"),
            "bmm_tflops": record.get("bmm_tflops"),
            "best_config": format_triton_config(record.get("best_config")),
            "pipeline_ms": _nested(
                record, "pipeline_performance", "median_ms"
            ),
            "pipeline_p20_ms": _nested(
                record, "pipeline_performance", "p20_ms"
            ),
            "pipeline_p80_ms": _nested(
                record, "pipeline_performance", "p80_ms"
            ),
            "pipeline_tflops": record.get("pipeline_tflops"),
            "kernel_rel_l2": _nested(
                record, "kernel_accuracy", "rel_l2"
            ),
            "pipeline_rel_l2": _nested(
                record, "pipeline_accuracy", "rel_l2"
            ),
            "pipeline_cosine": _nested(
                record, "pipeline_accuracy", "cosine"
            ),
            "pipeline_mse": _nested(record, "pipeline_accuracy", "mse"),
        }

    if kind == "quant":
        duration_ms = _nested(record, "performance", "median_ms")
        return {
            **common,
            "duration_ms": duration_ms,
            # Backward-compatible alias for existing report --column usage.
            "median_ms": duration_ms,
            "p20_ms": _nested(record, "performance", "p20_ms"),
            "p80_ms": _nested(record, "performance", "p80_ms"),
            "bandwidth_gbps": _nested(
                record, "performance", "bandwidth_gbps"
            ),
            "best_config": format_triton_config(record.get("best_config")),
            "rel_l2": _nested(record, "accuracy", "rel_l2"),
            "cosine": _nested(record, "accuracy", "cosine"),
            "mse": _nested(record, "accuracy", "mse"),
            "max_abs": _nested(record, "accuracy", "max_abs"),
            "saturation_ratio": _nested(
                record, "accuracy", "saturation_ratio"
            ),
        }

    if kind == "fsdp2":
        peak_bytes = record.get("max_peak_memory_bytes")
        duration_ms = record.get("median_step_ms")
        return {
            **common,
            "world_size": record.get("world_size"),
            "batch_per_rank": record.get("batch_per_rank"),
            "seq": record.get("seq"),
            "duration_ms": duration_ms,
            # Backward-compatible alias for existing report --column usage.
            "median_step_ms": duration_ms,
            "samples_per_second": record.get("samples_per_second"),
            "estimated_gemm_tflops": record.get("estimated_gemm_tflops"),
            "peak_memory_gib": (
                float(peak_bytes) / 1024**3 if peak_bytes is not None else None
            ),
            "first_measured_loss": record.get("first_measured_loss"),
            "last_loss": record.get("last_loss"),
        }

    # Unknown record kinds remain inspectable through pandas.json_normalize.
    return dict(pd.json_normalize(record, sep=".").iloc[0])


def records_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(record_to_row(record) for record in records)


def default_columns(frame: pd.DataFrame) -> list[str]:
    kinds = [str(value) for value in frame.get("kind", pd.Series(dtype=str)).dropna().unique()]
    if len(kinds) == 1 and kinds[0] in _DEFAULT_COLUMNS:
        candidates = _DEFAULT_COLUMNS[kinds[0]]
    else:
        candidates = tuple(frame.columns)
    return [
        column
        for column in candidates
        if column in frame.columns and not frame[column].isna().all()
    ]


def _formatter(column: str):
    if column.endswith("_ms"):
        return lambda value: f"{value:.4f}"
    if "tflops" in column:
        return lambda value: f"{value:.2f}"
    if column.endswith("_gbps"):
        return lambda value: f"{value:.1f}"
    if column in {"samples_per_second", "peak_memory_gib"}:
        return lambda value: f"{value:.2f}"
    if any(
        token in column
        for token in ("rel_l2", "cosine", "mse", "max_abs", "loss", "ratio")
    ):
        return lambda value: f"{value:.6g}"
    return None


def format_frame(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> str:
    selected = list(columns) if columns is not None else default_columns(frame)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"unknown report columns: {missing}")
    view = frame.loc[:, selected]
    formatters = {
        column: formatter
        for column in selected
        if (formatter := _formatter(column)) is not None
    }
    return view.to_string(
        index=False,
        na_rep="-",
        formatters=formatters,
    )


def print_records(
    records: Iterable[Mapping[str, Any]],
    *,
    title: str,
) -> pd.DataFrame:
    frame = records_frame(records)
    if frame.empty:
        return frame
    print(f"\n{title}")
    print(format_frame(frame))
    return frame


def print_progress(
    current: int,
    total: int,
    label: str,
    *,
    performance: str | None = None,
) -> None:
    width = len(str(total))
    prefix = f"[{current:>{width}}/{total}] {label}"
    line = (
        f"{prefix:<{100}}{performance}"
        if performance
        else prefix
    )
    print(line, flush=True)
