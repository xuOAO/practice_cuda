from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class QuantResult:
    tensor: torch.Tensor
    dequant_scale: torch.Tensor
    impl: str
    meta: dict[str, Any] = field(default_factory=dict)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        scale = self.dequant_scale.to(dtype)
        channel_axis = self.meta.get("channel_axis")
        if channel_axis in {-1, -2}:
            expected_ndim = self.tensor.ndim - 1
            if scale.ndim != expected_ndim:
                raise ValueError(
                    "per-channel dequant scale must have one fewer dimension "
                    f"than its tensor: tensor={tuple(self.tensor.shape)}, "
                    f"scale={tuple(scale.shape)}"
                )
            scale = scale.unsqueeze(-1 if channel_axis == -2 else -2)
        return self.tensor.to(dtype) * scale


@dataclass(frozen=True)
class QuantImpl:
    name: str
    fn: Callable[..., QuantResult]
    description: str = ""


@dataclass(frozen=True)
class BMMImpl:
    name: str
    fn: Callable[..., torch.Tensor]
    quant_impl: str
    layout: str
    prepare_b: Callable[[QuantResult], QuantResult]
    quant_a_kwargs: dict[str, Any]
    quant_b_kwargs: dict[str, Any]
    prepare_call_kwargs: Callable[[QuantResult, QuantResult], dict[str, Any]]
    description: str = ""


QUANT_IMPLS: dict[str, QuantImpl] = {}
BMM_IMPLS: dict[str, BMMImpl] = {}
_BUILTINS_LOADED = False


def register_quant(name: str, fn: Callable[..., QuantResult], description: str = "") -> None:
    if name in QUANT_IMPLS:
        raise ValueError(f"duplicate quant implementation: {name}")
    QUANT_IMPLS[name] = QuantImpl(name, fn, description)


def register_bmm(
    name: str,
    fn: Callable[..., torch.Tensor],
    *,
    quant_impl: str,
    layout: str,
    prepare_b: Callable[[QuantResult], QuantResult],
    quant_a_kwargs: dict[str, Any] | None = None,
    quant_b_kwargs: dict[str, Any] | None = None,
    prepare_call_kwargs: (
        Callable[[QuantResult, QuantResult], dict[str, Any]] | None
    ) = None,
    description: str = "",
) -> None:
    if name in BMM_IMPLS:
        raise ValueError(f"duplicate BMM implementation: {name}")
    if layout not in {"n", "k"}:
        raise ValueError(f"layout must be 'n' or 'k', got {layout}")
    BMM_IMPLS[name] = BMMImpl(
        name,
        fn,
        quant_impl,
        layout,
        prepare_b,
        dict(quant_a_kwargs or {}),
        dict(quant_b_kwargs or {}),
        prepare_call_kwargs or (lambda _a, _b: {}),
        description,
    )


def load_builtin_impls() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        importlib.import_module("fp8_bench.impls")
        _BUILTINS_LOADED = True


def get_quant(name: str) -> QuantImpl:
    load_builtin_impls()
    try:
        return QUANT_IMPLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown quant implementation {name}; choices={sorted(QUANT_IMPLS)}") from exc


def get_bmm(name: str) -> BMMImpl:
    load_builtin_impls()
    try:
        return BMM_IMPLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown BMM implementation {name}; choices={sorted(BMM_IMPLS)}") from exc
