from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

import triton

from fp8_bench.kernels.bmm.per_block import (
    batch_fp8_per_block_bmm_kernel,
    batch_fp8_per_block_bmm_tma_kernel,
)
from fp8_bench.kernels.bmm.per_channel import (
    batch_fp8_per_channel_bmm_kernel,
    batch_fp8_per_channel_bmm_tma_kernel,
)
from fp8_bench.kernels.bmm.per_tensor import (
    batch_fp8_per_tensor_bmm_kernel,
    batch_fp8_per_tensor_bmm_tma_kernel,
)
from fp8_bench.tuning.space import KernelConfig, cdiv


def _tma_alloc_fn(size: int, alignment: int, stream: Any) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_tma_allocator() -> None:
    # TMA tensor descriptors need a device-side allocator for descriptor memory.
    # It is global and idempotent; setting it does not affect non-TMA kernels.
    triton.set_allocator(_tma_alloc_fn)


def _assert_tma_aligned(
    spec: "TuningSpec",
    a_k_major: bool,
    b_n_major: bool,
) -> None:
    # TMA descriptors require 16-byte aligned leading strides. For fp8 (1 byte)
    # that means K must be a multiple of 16 (A's leading stride), and for
    # N-major B the leading stride is N, so N must also be a multiple of 16.
    a_leading_dim = spec.k if a_k_major else spec.m
    if a_leading_dim % 16 != 0:
        raise ValueError(
            "TMA requires A's leading stride to be 16-byte aligned, got "
            f"A layout={'K' if a_k_major else 'N'}-major and "
            f"leading_dim={a_leading_dim}"
        )
    b_leading_dim = spec.n if b_n_major else spec.k
    if b_leading_dim % 16 != 0:
        raise ValueError(
            "TMA requires B's leading stride to be 16-byte aligned, got "
            f"B layout={'N' if b_n_major else 'K'}-major and "
            f"leading_dim={b_leading_dim}"
        )
    output_stride_bytes = (
        spec.n * torch.empty((), dtype=spec.out_dtype).element_size()
    )
    if output_stride_bytes % 16 != 0:
        raise ValueError(
            "TMA requires C's leading stride to be 16-byte aligned, got "
            f"N={spec.n}, dtype={spec.out_dtype}, "
            f"stride_bytes={output_stride_bytes}"
        )


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
    a_k_major: bool
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


def _matrix_strides(
    spec: TuningSpec,
    a_k_major: bool,
    b_n_major: bool,
) -> dict[str, tuple[int, ...]]:
    if a_k_major:
        a_strides = (spec.m * spec.k, spec.k, 1)
    else:
        a_strides = (spec.m * spec.k, 1, spec.m)
    if b_n_major:
        b_strides = (spec.k * spec.n, spec.n, 1)
    else:
        b_strides = (spec.k * spec.n, 1, spec.k)
    return {
        "a": a_strides,
        "b": b_strides,
        "c": (spec.m * spec.n, spec.n, 1),
        "bias": (spec.m * spec.n, spec.n, 1) if spec.use_bias else (0, 0, 0),
    }


def _make_matrices(
    spec: TuningSpec,
    a_k_major: bool,
    b_n_major: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a_k_major:
        a = torch.empty(
            (spec.batch, spec.m, spec.k),
            device="cuda",
            dtype=spec.fp8_dtype,
        )
    else:
        a = torch.empty(
            (spec.batch, spec.k, spec.m),
            device="cuda",
            dtype=spec.fp8_dtype,
        ).transpose(-1, -2)
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

    def __init__(
        self,
        name: str,
        kernel: Any,
        *,
        a_k_major: bool,
        b_n_major: bool,
    ) -> None:
        self.name = name
        self.kernel = kernel
        self.a_k_major = a_k_major
        self.b_n_major = b_n_major

    def _compile_common(
        self,
        spec: TuningSpec,
    ) -> tuple[Any, ...]:
        strides = _matrix_strides(spec, self.a_k_major, self.b_n_major)
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
            "A_K_MAJOR": self.a_k_major,
            "B_N_MAJOR": self.b_n_major,
            "ACTIVATION": "none",
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

    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
        kernel: Any = batch_fp8_per_tensor_bmm_kernel,
    ) -> None:
        super().__init__(
            name,
            kernel,
            a_k_major=a_k_major,
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
        a, b, c = _make_matrices(spec, self.a_k_major, self.b_n_major)
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

    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
        kernel: Any = batch_fp8_per_channel_bmm_kernel,
    ) -> None:
        super().__init__(
            name,
            kernel,
            a_k_major=a_k_major,
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
        a, b, c = _make_matrices(spec, self.a_k_major, self.b_n_major)
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

    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
        kernel: Any = batch_fp8_per_block_bmm_kernel,
    ) -> None:
        super().__init__(
            name,
            kernel,
            a_k_major=a_k_major,
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
        a, b, c = _make_matrices(spec, self.a_k_major, self.b_n_major)
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


class PerTensorTmaAdapter(PerTensorAdapter):
    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
    ) -> None:
        _ensure_tma_allocator()
        super().__init__(
            name,
            b_n_major=b_n_major,
            a_k_major=a_k_major,
            kernel=batch_fp8_per_tensor_bmm_tma_kernel,
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        _assert_tma_aligned(spec, self.a_k_major, self.b_n_major)
        return super().create_runtime(spec)

    def _launch_kwargs(
        self,
        spec: TuningSpec,
        config: KernelConfig,
    ) -> dict[str, Any]:
        return {
            **super()._launch_kwargs(spec, config),
            "SCALES_ARE_QUANT": False,
        }

    def compile(self, spec: TuningSpec, config: KernelConfig) -> Any:
        common = self._compile_common(spec)
        args = (
            *common[:4],
            torch.float32,
            torch.float32,
            *common[4:],
        )
        return self.kernel.warmup(
            *args,
            grid=(1, 1),
            **self._launch_kwargs(spec, config),
        )

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


class PerChannelTmaAdapter(PerChannelAdapter):
    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
    ) -> None:
        _ensure_tma_allocator()
        super().__init__(
            name,
            b_n_major=b_n_major,
            a_k_major=a_k_major,
            kernel=batch_fp8_per_channel_bmm_tma_kernel,
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        _assert_tma_aligned(spec, self.a_k_major, self.b_n_major)
        return super().create_runtime(spec)


class PerBlockTmaAdapter(PerBlockAdapter):
    def __init__(
        self,
        name: str,
        *,
        b_n_major: bool,
        a_k_major: bool = True,
    ) -> None:
        _ensure_tma_allocator()
        super().__init__(
            name,
            b_n_major=b_n_major,
            a_k_major=a_k_major,
            kernel=batch_fp8_per_block_bmm_tma_kernel,
        )

    def create_runtime(self, spec: TuningSpec) -> RuntimeTensors:
        _assert_tma_aligned(spec, self.a_k_major, self.b_n_major)
        return super().create_runtime(spec)


ADAPTERS: dict[str, BMMTuningAdapter] = {
    "triton_per_tensor_a_k_b_n": PerTensorAdapter(
        "triton_per_tensor_a_k_b_n",
        b_n_major=True,
    ),
    "triton_per_tensor_a_k_b_k": PerTensorAdapter(
        "triton_per_tensor_a_k_b_k",
        b_n_major=False,
    ),
    "triton_per_tensor_a_k_b_n_transpose": PerTensorAdapter(
        "triton_per_tensor_a_k_b_n_transpose",
        b_n_major=False,
    ),
    "triton_per_tensor_a_m_b_n": PerTensorAdapter(
        "triton_per_tensor_a_m_b_n",
        a_k_major=False,
        b_n_major=True,
    ),
    "triton_per_tensor_a_m_transpose_b_n_transpose": PerTensorAdapter(
        "triton_per_tensor_a_m_transpose_b_n_transpose",
        a_k_major=True,
        b_n_major=False,
    ),
    "triton_per_tensor_a_k_b_n_tma": PerTensorTmaAdapter(
        "triton_per_tensor_a_k_b_n_tma",
        b_n_major=True,
    ),
    "triton_per_tensor_a_k_b_k_tma": PerTensorTmaAdapter(
        "triton_per_tensor_a_k_b_k_tma",
        b_n_major=False,
    ),
    "triton_per_tensor_a_k_b_n_transpose_tma": PerTensorTmaAdapter(
        "triton_per_tensor_a_k_b_n_transpose_tma",
        b_n_major=False,
    ),
    "triton_per_tensor_a_m_b_n_tma": PerTensorTmaAdapter(
        "triton_per_tensor_a_m_b_n_tma",
        a_k_major=False,
        b_n_major=True,
    ),
    "triton_per_tensor_a_m_transpose_b_n_transpose_tma": PerTensorTmaAdapter(
        "triton_per_tensor_a_m_transpose_b_n_transpose_tma",
        a_k_major=True,
        b_n_major=False,
    ),
    "triton_per_channel_a_k_b_n": PerChannelAdapter(
        "triton_per_channel_a_k_b_n",
        b_n_major=True,
    ),
    "triton_per_channel_a_k_b_k": PerChannelAdapter(
        "triton_per_channel_a_k_b_k",
        b_n_major=False,
    ),
    "triton_per_channel_a_k_b_n_transpose": PerChannelAdapter(
        "triton_per_channel_a_k_b_n_transpose",
        b_n_major=False,
    ),
    "triton_per_channel_a_m_b_n": PerChannelAdapter(
        "triton_per_channel_a_m_b_n",
        a_k_major=False,
        b_n_major=True,
    ),
    "triton_per_channel_a_m_transpose_b_n_transpose": PerChannelAdapter(
        "triton_per_channel_a_m_transpose_b_n_transpose",
        a_k_major=True,
        b_n_major=False,
    ),
    "triton_per_channel_a_k_b_n_tma": PerChannelTmaAdapter(
        "triton_per_channel_a_k_b_n_tma",
        b_n_major=True,
    ),
    "triton_per_channel_a_k_b_k_tma": PerChannelTmaAdapter(
        "triton_per_channel_a_k_b_k_tma",
        b_n_major=False,
    ),
    "triton_per_channel_a_k_b_n_transpose_tma": PerChannelTmaAdapter(
        "triton_per_channel_a_k_b_n_transpose_tma",
        b_n_major=False,
    ),
    "triton_per_channel_a_m_b_n_tma": PerChannelTmaAdapter(
        "triton_per_channel_a_m_b_n_tma",
        a_k_major=False,
        b_n_major=True,
    ),
    "triton_per_channel_a_m_transpose_b_n_transpose_tma": PerChannelTmaAdapter(
        "triton_per_channel_a_m_transpose_b_n_transpose_tma",
        a_k_major=True,
        b_n_major=False,
    ),
    **{
        f"triton_per_block_{scheme}_a_k_b_n": PerBlockAdapter(
            f"triton_per_block_{scheme}_a_k_b_n",
            b_n_major=True,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_k_b_k": PerBlockAdapter(
            f"triton_per_block_{scheme}_a_k_b_k",
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_k_b_n_transpose": PerBlockAdapter(
            f"triton_per_block_{scheme}_a_k_b_n_transpose",
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_m_b_n": PerBlockAdapter(
            f"triton_per_block_{scheme}_a_m_b_n",
            a_k_major=False,
            b_n_major=True,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_m_transpose_b_n_transpose": PerBlockAdapter(
            f"triton_per_block_{scheme}_a_m_transpose_b_n_transpose",
            a_k_major=True,
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_k_b_n_tma": PerBlockTmaAdapter(
            f"triton_per_block_{scheme}_a_k_b_n_tma",
            b_n_major=True,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_k_b_k_tma": PerBlockTmaAdapter(
            f"triton_per_block_{scheme}_a_k_b_k_tma",
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_k_b_n_transpose_tma": PerBlockTmaAdapter(
            f"triton_per_block_{scheme}_a_k_b_n_transpose_tma",
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_m_b_n_tma": PerBlockTmaAdapter(
            f"triton_per_block_{scheme}_a_m_b_n_tma",
            a_k_major=False,
            b_n_major=True,
        )
        for scheme in ("1d", "2d")
    },
    **{
        f"triton_per_block_{scheme}_a_m_transpose_b_n_transpose_tma": PerBlockTmaAdapter(
            f"triton_per_block_{scheme}_a_m_transpose_b_n_transpose_tma",
            a_k_major=True,
            b_n_major=False,
        )
        for scheme in ("1d", "2d")
    },
}


def get_adapter(name: str) -> BMMTuningAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown tuning adapter {name}; choices={sorted(ADAPTERS)}") from exc
