from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from fp8_bench.kernels.bmm.per_block import batch_fp8_per_block_bmm_kernel
from fp8_bench.kernels.bmm.per_channel import batch_fp8_per_channel_bmm_kernel
from fp8_bench.kernels.bmm.per_tensor import batch_fp8_per_tensor_bmm_kernel
from fp8_bench.tuning.space import KernelConfig, cdiv


@dataclass(frozen=True)
class TuningSpec:
    impl: str
    case: str
    batch: int
    m: int
    n: int
    k: int
    fp8_dtype: torch.dtype
    out_dtype: torch.dtype
    use_bias: bool
    quant_block_m: int | None = None
    quant_block_k: int | None = None
    quant_block_n: int | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.m, self.n, self.k

    def as_dict(self) -> dict[str, Any]:
        return {
            "impl": self.impl,
            "case": self.case,
            "shape": list(self.shape),
            "fp8_dtype": str(self.fp8_dtype),
            "out_dtype": str(self.out_dtype),
            "use_bias": self.use_bias,
            "quant_block_m": self.quant_block_m,
            "quant_block_k": self.quant_block_k,
            "quant_block_n": self.quant_block_n,
        }


@dataclass
class RuntimeTensors:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    bias_ptr: torch.Tensor
    bias_strides: tuple[int, int, int]
    scale_a: torch.Tensor
    scale_b: torch.Tensor | None = None


class BMMTuningAdapter(Protocol):
    name: str
    family: str
    b_n_major: bool

    def compile(self, spec: TuningSpec, config: KernelConfig) -> Any:
        ...

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        ...

    def make_launcher(
        self,
        spec: TuningSpec,
        config: KernelConfig,
        runtime: RuntimeTensors,
    ) -> Callable[[], None]:
        ...


def _matrix_strides(spec: TuningSpec, b_n_major: bool) -> dict[str, tuple[int, ...]]:
    if b_n_major:
        b_strides = (spec.k * spec.n, spec.n, 1)
    else:
        b_strides = (spec.k * spec.n, 1, spec.k)
    return {
        "a": (spec.m * spec.k, spec.k, 1),
        "b": b_strides,
        "c": (spec.m * spec.n, spec.n, 1),
        "bias": (spec.m * spec.n, spec.n, 1) if spec.use_bias else (0, 0, 0),
    }


def _make_matrices(
    spec: TuningSpec,
    b_n_major: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.empty(
        (spec.batch, spec.m, spec.k),
        device="cuda",
        dtype=spec.fp8_dtype,
    )
    if b_n_major:
        b = torch.empty(
            (spec.batch, spec.k, spec.n),
            device="cuda",
            dtype=spec.fp8_dtype,
        )
    else:
        b = torch.empty(
            (spec.batch, spec.n, spec.k),
            device="cuda",
            dtype=spec.fp8_dtype,
        ).transpose(-1, -2)
    c = torch.empty(
        (spec.batch, spec.m, spec.n),
        device="cuda",
        dtype=spec.out_dtype,
    )
    return a, b, c


def _make_bias(
    spec: TuningSpec,
    a: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if not spec.use_bias:
        return a, (0, 0, 0)
    bias = torch.zeros(
        (spec.batch, spec.m, spec.n),
        device="cuda",
        dtype=spec.out_dtype,
    )
    return bias, tuple(bias.stride())


class _BaseAdapter:
    family: str

    def __init__(self, name: str, kernel: Any, *, b_n_major: bool) -> None:
        self.name = name
        self.kernel = kernel
        self.b_n_major = b_n_major

    def _compile_common(
        self,
        spec: TuningSpec,
    ) -> tuple[Any, ...]:
        strides = _matrix_strides(spec, self.b_n_major)
        bias_dtype = spec.out_dtype if spec.use_bias else spec.fp8_dtype
        return (
            spec.fp8_dtype,
            spec.fp8_dtype,
            spec.out_dtype,
            bias_dtype,
            spec.m,
            spec.n,
            spec.k,
            *strides["a"],
            *strides["b"],
            *strides["c"],
            *strides["bias"],
        )

    def _launch_kwargs(
        self,
        spec: TuningSpec,
        config: KernelConfig,
    ) -> dict[str, Any]:
        return {
            "USE_BIAS": spec.use_bias,
            "B_N_MAJOR": self.b_n_major,
            **config.as_triton_kwargs(),
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }

    def _grid(
        self,
        spec: TuningSpec,
        config: KernelConfig,
    ) -> tuple[int, int]:
        return (
            cdiv(spec.m, config.block_m) * cdiv(spec.n, config.block_n),
            spec.batch,
        )


class PerTensorAdapter(_BaseAdapter):
    family = "per_tensor"

    def __init__(self, name: str, *, b_n_major: bool) -> None:
        super().__init__(
            name,
            batch_fp8_per_tensor_bmm_kernel,
            b_n_major=b_n_major,
        )

    def compile(self, spec: TuningSpec, config: KernelConfig) -> Any:
        common = self._compile_common(spec)
        args = (*common[:4], torch.float32, *common[4:])
        return self.kernel.warmup(
            *args,
            grid=(1, 1),
            **self._launch_kwargs(spec, config),
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        a, b, c = _make_matrices(spec, self.b_n_major)
        bias_ptr, bias_strides = _make_bias(spec, a)
        scale = torch.ones((), device="cuda", dtype=torch.float32)
        return RuntimeTensors(a, b, c, bias_ptr, bias_strides, scale)

    def make_launcher(
        self,
        spec: TuningSpec,
        config: KernelConfig,
        runtime: RuntimeTensors,
    ) -> Callable[[], None]:
        def run() -> None:
            self.kernel[self._grid(spec, config)](
                runtime.a,
                runtime.b,
                runtime.c,
                runtime.bias_ptr,
                runtime.scale_a,
                spec.m,
                spec.n,
                spec.k,
                *runtime.a.stride(),
                *runtime.b.stride(),
                *runtime.c.stride(),
                *runtime.bias_strides,
                **self._launch_kwargs(spec, config),
            )

        return run


class PerChannelAdapter(_BaseAdapter):
    family = "per_channel"

    def __init__(self, name: str, *, b_n_major: bool) -> None:
        super().__init__(
            name,
            batch_fp8_per_channel_bmm_kernel,
            b_n_major=b_n_major,
        )

    def compile(self, spec: TuningSpec, config: KernelConfig) -> Any:
        common = self._compile_common(spec)
        args = (
            *common[:4],
            torch.float32,
            torch.float32,
            *common[4:],
            spec.m,
            1,
            spec.n,
            1,
        )
        return self.kernel.warmup(
            *args,
            grid=(1, 1),
            **self._launch_kwargs(spec, config),
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        a, b, c = _make_matrices(spec, self.b_n_major)
        bias_ptr, bias_strides = _make_bias(spec, a)
        scale_a = torch.ones(
            (spec.batch, spec.m),
            device="cuda",
            dtype=torch.float32,
        )
        scale_b = torch.ones(
            (spec.batch, spec.n),
            device="cuda",
            dtype=torch.float32,
        )
        return RuntimeTensors(
            a,
            b,
            c,
            bias_ptr,
            bias_strides,
            scale_a,
            scale_b,
        )

    def make_launcher(
        self,
        spec: TuningSpec,
        config: KernelConfig,
        runtime: RuntimeTensors,
    ) -> Callable[[], None]:
        assert runtime.scale_b is not None

        def run() -> None:
            self.kernel[self._grid(spec, config)](
                runtime.a,
                runtime.b,
                runtime.c,
                runtime.bias_ptr,
                runtime.scale_a,
                runtime.scale_b,
                spec.m,
                spec.n,
                spec.k,
                *runtime.a.stride(),
                *runtime.b.stride(),
                *runtime.c.stride(),
                *runtime.bias_strides,
                *runtime.scale_a.stride(),
                *runtime.scale_b.stride(),
                **self._launch_kwargs(spec, config),
            )

        return run


class PerBlockAdapter(_BaseAdapter):
    family = "per_block"

    def __init__(self, name: str, *, b_n_major: bool) -> None:
        super().__init__(
            name,
            batch_fp8_per_block_bmm_kernel,
            b_n_major=b_n_major,
        )

    @staticmethod
    def _blocks(spec: TuningSpec) -> tuple[int, int, int, int, int, int]:
        if (
            spec.quant_block_m is None
            or spec.quant_block_k is None
            or spec.quant_block_n is None
        ):
            raise ValueError("per-block tuning requires QBM, QBK and QBN")
        qbm = spec.quant_block_m
        qbk = spec.quant_block_k
        qbn = spec.quant_block_n
        return (
            qbm,
            qbk,
            qbn,
            cdiv(spec.m, qbm),
            cdiv(spec.k, qbk),
            cdiv(spec.n, qbn),
        )

    def _block_launch_kwargs(
        self,
        spec: TuningSpec,
        config: KernelConfig,
    ) -> dict[str, Any]:
        qbm, qbk, qbn, _, num_qbk, _ = self._blocks(spec)
        if config.block_k > qbk or qbk % config.block_k != 0:
            raise ValueError(
                f"BLOCK_K={config.block_k} is incompatible with QBK={qbk}"
            )
        return {
            **self._launch_kwargs(spec, config),
            "QUANT_BLOCK_M": qbm,
            "QUANT_BLOCK_K": qbk,
            "QUANT_BLOCK_N": qbn,
            "NUM_QUANT_BLOCK_K": num_qbk,
        }

    def compile(self, spec: TuningSpec, config: KernelConfig) -> Any:
        common = self._compile_common(spec)
        _, _, _, num_qbm, num_qbk, num_qbn = self._blocks(spec)
        scale_a_strides = (num_qbm * num_qbk, num_qbk, 1)
        scale_b_strides = (num_qbk * num_qbn, num_qbn, 1)
        args = (
            *common[:4],
            torch.float32,
            torch.float32,
            *common[4:],
            *scale_a_strides,
            *scale_b_strides,
        )
        return self.kernel.warmup(
            *args,
            grid=(1, 1),
            **self._block_launch_kwargs(spec, config),
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        a, b, c = _make_matrices(spec, self.b_n_major)
        bias_ptr, bias_strides = _make_bias(spec, a)
        _, _, _, num_qbm, num_qbk, num_qbn = self._blocks(spec)
        scale_a = torch.ones(
            (spec.batch, num_qbm, num_qbk),
            device="cuda",
            dtype=torch.float32,
        )
        scale_b = torch.ones(
            (spec.batch, num_qbk, num_qbn),
            device="cuda",
            dtype=torch.float32,
        )
        return RuntimeTensors(
            a,
            b,
            c,
            bias_ptr,
            bias_strides,
            scale_a,
            scale_b,
        )

    def make_launcher(
        self,
        spec: TuningSpec,
        config: KernelConfig,
        runtime: RuntimeTensors,
    ) -> Callable[[], None]:
        assert runtime.scale_b is not None

        def run() -> None:
            self.kernel[self._grid(spec, config)](
                runtime.a,
                runtime.b,
                runtime.c,
                runtime.bias_ptr,
                runtime.scale_a,
                runtime.scale_b,
                spec.m,
                spec.n,
                spec.k,
                *runtime.a.stride(),
                *runtime.b.stride(),
                *runtime.c.stride(),
                *runtime.bias_strides,
                *runtime.scale_a.stride(),
                *runtime.scale_b.stride(),
                **self._block_launch_kwargs(spec, config),
            )

        return run


ADAPTERS: dict[str, BMMTuningAdapter] = {
    "triton_per_tensor_n": PerTensorAdapter(
        "triton_per_tensor_n",
        b_n_major=True,
    ),
    "triton_per_tensor_k": PerTensorAdapter(
        "triton_per_tensor_k",
        b_n_major=False,
    ),
    "triton_per_tensor_n_transpose": PerTensorAdapter(
        "triton_per_tensor_n_transpose",
        b_n_major=False,
    ),
    "triton_per_channel_n": PerChannelAdapter(
        "triton_per_channel_n",
        b_n_major=True,
    ),
    "triton_per_channel_k": PerChannelAdapter(
        "triton_per_channel_k",
        b_n_major=False,
    ),
    "triton_per_channel_n_transpose": PerChannelAdapter(
        "triton_per_channel_n_transpose",
        b_n_major=False,
    ),
    "triton_per_block_n": PerBlockAdapter(
        "triton_per_block_n",
        b_n_major=True,
    ),
    "triton_per_block_k": PerBlockAdapter(
        "triton_per_block_k",
        b_n_major=False,
    ),
    "triton_per_block_n_transpose": PerBlockAdapter(
        "triton_per_block_n_transpose",
        b_n_major=False,
    ),
}


def get_adapter(name: str) -> BMMTuningAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown tuning adapter {name}; choices={sorted(ADAPTERS)}") from exc
