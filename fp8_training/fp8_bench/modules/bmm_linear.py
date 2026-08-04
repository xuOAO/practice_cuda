from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import triton

from fp8_bench.kernels.bmm.per_tensor import (
    batch_fp8_per_tensor_bmm_tma_kernel_autotuned,
)
from fp8_bench.modules.float8.config import (
    Float8LinearConfig,
    ScalingGranularity,
    ScalingType,
)
from fp8_bench.modules.float8.distributed_utils import (
    tensor_already_casted_to_fp8,
)
from fp8_bench.modules.float8.float8_scaling_utils import (
    hp_tensor_to_float8_dynamic,
)
from fp8_bench.modules.float8.float8_training_tensor import (
    Float8TrainingTensor,
    GemmInputRole,
    LinearMMConfig,
)
from fp8_bench.modules.float8.float8_utils import EPS
from fp8_bench.modules.float8.fsdp_utils import (
    WeightWithDynamicFloat8CastTensor,
)


@dataclass(frozen=True)
class Float8BMMLinearConfig(Float8LinearConfig):
    """torchAO-compatible configuration for :class:`Float8BMMLinear`.

    The BMM TMA backend currently supports dynamic tensorwise scaling only.
    torchAO's default mixed-dtype recipe is retained: input and weight use
    E4M3, while grad-output uses E5M2.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        cast_configs = {
            "input": self.cast_config_input,
            "input_for_grad_weight": self.cast_config_input_for_grad_weight,
            "weight": self.cast_config_weight,
            "weight_for_grad_input": self.cast_config_weight_for_grad_input,
            "grad_output": self.cast_config_grad_output,
            "grad_output_for_grad_weight": (
                self.cast_config_grad_output_for_grad_weight
            ),
        }
        for name, cast_config in cast_configs.items():
            if cast_config.scaling_type is not ScalingType.DYNAMIC:
                raise ValueError(
                    f"Float8BMMLinear requires dynamic FP8 casting for {name}"
                )
            if (
                cast_config.scaling_granularity
                is not ScalingGranularity.TENSORWISE
            ):
                raise ValueError(
                    "Float8BMMLinear's TMA backend requires tensorwise scaling "
                    f"for {name}"
                )
        if self.emulate:
            raise ValueError("Float8BMMLinear does not support emulation")


def _tma_allocator(
    size: int,
    alignment: int,
    stream: Optional[int],
) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_tma_allocator)


def _a_is_k_major(tensor: torch.Tensor) -> bool:
    if tensor.stride(-1) == 1:
        return True
    if tensor.stride(-2) == 1:
        return False
    raise ValueError(
        "A must be contiguous along M or K; "
        f"shape={tuple(tensor.shape)}, strides={tensor.stride()}"
    )


def _b_is_n_major(tensor: torch.Tensor) -> bool:
    if tensor.stride(-1) == 1:
        return True
    if tensor.stride(-2) == 1:
        return False
    raise ValueError(
        "B must be contiguous along K or N; "
        f"shape={tuple(tensor.shape)}, strides={tensor.stride()}"
    )


@torch.library.triton_op(
    "fp8_bench::bmm_linear_tma",
    mutates_args=(),
)
def _bmm_linear_tma_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    use_bias: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Triton's allocator is a ContextVar. Backward may run on an autograd
    # worker thread, so install it in the actual launch context as well.
    triton.set_allocator(_tma_allocator)

    batch, m, k = a.shape
    n = b.shape[-1]
    output = torch.empty(
        (batch, m, n),
        device=a.device,
        dtype=output_dtype,
    )
    a_k_major = _a_is_k_major(a)
    b_n_major = _b_is_n_major(b)

    if use_bias:
        stride_biasb = bias.stride(0)
        stride_biasm = 0
        stride_biasn = bias.stride(1)
    else:
        stride_biasb = stride_biasm = stride_biasn = 0

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(n, meta["BLOCK_N"]),
            batch,
        )

    torch.library.wrap_triton(
        batch_fp8_per_tensor_bmm_tma_kernel_autotuned
    )[grid](
        a,
        b,
        output,
        bias,
        scale_a,
        scale_b,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *output.stride(),
        stride_biasb,
        stride_biasm,
        stride_biasn,
        USE_BIAS=use_bias,
        A_K_MAJOR=a_k_major,
        B_N_MAJOR=b_n_major,
        SCALES_ARE_QUANT=True,
        ACTIVATION="none",
    )
    return output


def _fp8_bmm_tma(
    a: Float8TrainingTensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a_data = a._data
    a_scale = a._scale
    if a_data.ndim != 3 or b_data.ndim != 3:
        raise ValueError(
            "Float8 BMM expects rank-3 operands, got "
            f"A={tuple(a_data.shape)}, B={tuple(b_data.shape)}"
        )
    if a_data.shape[0] != b_data.shape[0] or a_data.shape[2] != b_data.shape[1]:
        raise ValueError(
            f"incompatible BMM shapes: A={tuple(a_data.shape)}, "
            f"B={tuple(b_data.shape)}"
        )
    if a_data.device != b_data.device or a_data.device.type != "cuda":
        raise ValueError(
            "Float8 BMM operands must be CUDA tensors on the same device"
        )
    if a_scale.numel() != 1 or b_scale.numel() != 1:
        raise ValueError("Float8BMMLinear only supports per-tensor scales")

    batch = a_data.shape[0]
    n = b_data.shape[2]
    if bias is None:
        bias_arg = a_data
        use_bias = False
    else:
        if bias.shape != (batch, n):
            raise ValueError(
                f"bias must have shape {(batch, n)}, got {tuple(bias.shape)}"
            )
        bias_arg = bias
        use_bias = True

    return _bmm_linear_tma_op(
        a_data,
        b_data,
        a_scale,
        b_scale,
        bias_arg,
        use_bias,
        output_dtype,
    )


def _to_float8(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    role: GemmInputRole,
    round_scales_to_power_of_2: bool,
) -> Float8TrainingTensor:
    if tensor_already_casted_to_fp8(tensor):
        if not isinstance(tensor, Float8TrainingTensor):
            raise TypeError(
                "Float8BMMLinear currently expects the FSDP2 all-gather result "
                "to be a local Float8TrainingTensor"
            )
        return tensor
    result = hp_tensor_to_float8_dynamic(
        tensor,
        target_dtype,
        linear_mm_config,
        gemm_input_role=role,
        scaling_granularity=ScalingGranularity.TENSORWISE,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    if not isinstance(result, Float8TrainingTensor):
        raise TypeError(f"expected Float8TrainingTensor, got {type(result)}")
    return result


@torch._dynamo.allow_in_graph
class _Float8BMMLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        config: Float8BMMLinearConfig,
        linear_mm_config: LinearMMConfig,
    ) -> torch.Tensor:
        input_fp8 = _to_float8(
            input,
            config.cast_config_input.target_dtype,
            linear_mm_config,
            GemmInputRole.INPUT,
            config.round_scales_to_power_of_2,
        )
        weight_fp8 = _to_float8(
            weight,
            config.cast_config_weight.target_dtype,
            linear_mm_config,
            GemmInputRole.WEIGHT,
            config.round_scales_to_power_of_2,
        )

        ctx.save_for_backward(input_fp8, weight_fp8)
        ctx.config = config
        ctx.linear_mm_config = linear_mm_config
        ctx.has_bias = bias is not None

        bias_for_kernel = None if bias is None else bias.to(input.dtype)
        return _fp8_bmm_tma(
            input_fp8,
            weight_fp8._data.transpose(-2, -1),
            weight_fp8._scale,
            input.dtype,
            bias_for_kernel,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_fp8, weight_fp8 = ctx.saved_tensors
        config = ctx.config
        grad_output_fp8 = _to_float8(
            grad_output,
            config.cast_config_grad_output.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
            config.round_scales_to_power_of_2,
        )

        grad_input = None
        if ctx.needs_input_grad[0]:
            # dX = dY @ W. W is [B,N,K], i.e. N-major as B's [K,N].
            grad_input = _fp8_bmm_tma(
                grad_output_fp8,
                weight_fp8._data,
                weight_fp8._scale,
                grad_output.dtype,
            )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            # dW = dY.T @ X. Both are views into already-quantized tensors;
            # TMA consumes A_M_MAJOR/B_N_MAJOR directly without packing.
            grad_output_t = Float8TrainingTensor(
                grad_output_fp8._data.transpose(-2, -1),
                grad_output_fp8._scale,
                grad_output_fp8._orig_dtype,
                ctx.linear_mm_config,
                GemmInputRole.GRAD_OUTPUT,
            )
            grad_weight = _fp8_bmm_tma(
                grad_output_t,
                input_fp8._data,
                input_fp8._scale,
                grad_output.dtype,
            )

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=1)
        return grad_input, grad_weight, grad_bias, None, None


class BMMLinear(nn.Module):
    """High-precision batched dense layer used as the conversion baseline."""

    def __init__(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or in_features <= 0 or out_features <= 0:
            raise ValueError("batch_size, in_features and out_features must be positive")
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                (batch_size, out_features, in_features),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((batch_size, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(f"BMMLinear expects [B,M,K], got {tuple(input.shape)}")
        if input.shape[0] != self.batch_size or input.shape[2] != self.in_features:
            raise ValueError(
                "input shape mismatch: expected "
                f"[{self.batch_size}, M, {self.in_features}], got {tuple(input.shape)}"
            )
        output = torch.bmm(input, self.weight.transpose(-2, -1))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)
        return output

    def extra_repr(self) -> str:
        return (
            f"batch_size={self.batch_size}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class Float8BMMLinear(nn.Module):
    """Batched dense layer implemented with FP8 TMA BMM.

    ``input`` has shape ``[batch, M, in_features]`` and the learnable weight
    has shape ``[batch, out_features, in_features]``. This is a standalone
    module and does not inherit from or wrap ``torch.nn.Linear``.
    """

    def __init__(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        config: Optional[Float8BMMLinearConfig] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or in_features <= 0 or out_features <= 0:
            raise ValueError("batch_size, in_features and out_features must be positive")

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or Float8BMMLinearConfig()
        self.linear_mm_config = LinearMMConfig()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                (batch_size, out_features, in_features),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((batch_size, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if self.config.enable_fsdp_float8_all_gather:
            self.weight = nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    self.weight,
                    self.linear_mm_config,
                    self.config.cast_config_weight.target_dtype,
                ),
                requires_grad=self.weight.requires_grad,
            )

    def reset_parameters(self) -> None:
        # Match nn.Linear's fan-in bound without applying Conv-style fan-in
        # inference to the leading BMM batch dimension.
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(
                f"Float8BMMLinear expects [B,M,K], got {tuple(input.shape)}"
            )
        if input.shape[0] != self.batch_size or input.shape[2] != self.in_features:
            raise ValueError(
                "input shape mismatch: expected "
                f"[{self.batch_size}, M, {self.in_features}], got {tuple(input.shape)}"
            )
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())
        return _Float8BMMLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.config,
            self.linear_mm_config,
        )

    def extra_repr(self) -> str:
        return (
            f"batch_size={self.batch_size}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}, "
            "backend=triton_per_tensor_tma, "
            f"fp8_all_gather={self.config.enable_fsdp_float8_all_gather}"
        )

    @classmethod
    def from_float(
        cls,
        module: BMMLinear,
        config: Optional[Float8BMMLinearConfig] = None,
    ) -> "Float8BMMLinear":
        if not isinstance(module, BMMLinear):
            raise TypeError(f"expected BMMLinear, got {type(module)}")
        config = config or Float8BMMLinearConfig()
        # Build metadata without allocating another real parameter tensor.
        with torch.device("meta"):
            converted = cls(
                module.batch_size,
                module.in_features,
                module.out_features,
                bias=False,
                config=Float8BMMLinearConfig(),
            )
        converted.config = config
        converted.weight = module.weight
        converted.bias = module.bias
        if config.enable_fsdp_float8_all_gather:
            converted.weight = nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    converted.weight,
                    converted.linear_mm_config,
                    config.cast_config_weight.target_dtype,
                ),
                requires_grad=converted.weight.requires_grad,
            )
        return converted


def convert_to_float8_bmm_training(
    module: nn.Module,
    config: Optional[Float8BMMLinearConfig] = None,
) -> nn.Module:
    """Recursively replace every :class:`BMMLinear` with its FP8 variant."""
    config = config or Float8BMMLinearConfig()
    if isinstance(module, BMMLinear):
        return Float8BMMLinear.from_float(module, config)
    for name, child in list(module.named_children()):
        converted_child = convert_to_float8_bmm_training(child, config)
        if converted_child is not child:
            module.register_module(name, converted_child)
    return module


@torch.no_grad()
def precompute_bmm_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None:
    """Precompute all BMM weight scales with one FSDP2 all-reduce.

    Call this after ``optimizer.step()``. It mirrors torchAO's float8-linear
    precompute path but scans :class:`Float8BMMLinear` modules.
    """
    from torch.distributed._tensor import DTensor

    bmm_linears = [
        child
        for child in module.modules()
        if isinstance(child, Float8BMMLinear)
        and isinstance(child.weight, DTensor)
        and isinstance(
            child.weight._local_tensor,
            WeightWithDynamicFloat8CastTensor,
        )
    ]
    if not bmm_linears:
        return

    target_dtypes = {
        child.config.cast_config_weight.target_dtype for child in bmm_linears
    }
    if len(target_dtypes) != 1:
        raise ValueError(
            "precomputed FP8 weight scales require one shared target dtype"
        )
    (target_dtype,) = target_dtypes
    weights = [child.weight for child in bmm_linears]

    max_weights = torch._foreach_norm(weights, ord=math.inf)
    amax_tensor = torch.clamp(torch.stack(max_weights), min=EPS)
    original_dtype = amax_tensor.dtype
    scale_tensor = torch.finfo(target_dtype).max / amax_tensor.to(torch.float64)
    if original_dtype is torch.float16:
        scale_tensor = torch.clamp(
            scale_tensor,
            max=torch.finfo(torch.float16).max,
        )
    local_scales = scale_tensor.to_local().to(torch.float32)
    for index, child in enumerate(bmm_linears):
        child.weight._local_tensor._precomputed_scale = local_scales[index]


__all__ = [
    "BMMLinear",
    "Float8BMMLinear",
    "Float8BMMLinearConfig",
    "convert_to_float8_bmm_training",
    "precompute_bmm_float8_dynamic_scale_for_fsdp",
]
