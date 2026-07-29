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
    granularity: str
    meta: dict[str, Any] = field(default_factory=dict)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        scale = self.dequant_scale.to(dtype)
        if self.granularity == "tensor":
            return self.tensor.to(dtype) * scale
        if self.granularity == "channel":
            channel_axis = self.meta["channel_axis"]
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
            else:
                raise ValueError(
                    f"unknown channel_axis for triton_per_channel: {channel_axis}"
                )
        if self.granularity == "block":
            if "block_m" not in self.meta or "block_n" not in self.meta:
                raise ValueError(
                    "block_m and block_n must be specified in meta for triton_per_block"
                )
            block_m, block_n = self.meta["block_m"], self.meta["block_n"]
            m, n = self.tensor.shape[-2:]
            if block_m <= 0 or block_n <= 0:
                raise ValueError(
                    "block_m and block_n must be positive for block granularity"
                )

            num_blocks_m = (m + block_m - 1) // block_m
            num_blocks_n = (n + block_n - 1) // block_n

            expected_shape = tuple(self.tensor.shape[:-2]) + (
                num_blocks_m,
                num_blocks_n,
            )
            if tuple(scale.shape) != expected_shape:
                raise ValueError(
                    f"per-block dequant scale must have shape {expected_shape}, "
                    f"got {tuple(scale.shape)}"
                )

            row_block_ids = torch.arange(m, device=scale.device) // block_m
            col_block_ids = torch.arange(n, device=scale.device) // block_n
            scale = scale.index_select(-2, row_block_ids).index_select(
                -1, col_block_ids
            )
            dequant = self.tensor.to(dtype)
            return dequant.mul_(scale)
        raise ValueError(f"unknown quantization granularity: {self.granularity}")


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
