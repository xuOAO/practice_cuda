from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


def cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def powers_of_two(begin: int, end: int) -> tuple[int, ...]:
    if begin <= 0 or end <= 0:
        raise ValueError(f"bounds must be positive, got begin={begin}, end={end}")
    value = next_power_of_two(begin)
    values: list[int] = []
    while value <= end:
        values.append(value)
        value *= 2
    return tuple(values)


@dataclass(frozen=True, order=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def as_triton_kwargs(self) -> dict[str, int]:
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
            "GROUP_M": self.group_m,
        }

    def format_triton(self) -> str:
        return (
            "triton.Config("
            f'{{"BLOCK_M": {self.block_m}, "BLOCK_N": {self.block_n}, '
            f'"BLOCK_K": {self.block_k}, "GROUP_M": {self.group_m}}}, '
            f"num_warps={self.num_warps}, num_stages={self.num_stages})"
        )


@dataclass(frozen=True)
class SpacePolicy:
    block_m_min: int = 64
    block_n_min: int = 8
    block_k_min: int = 32
    block_m_cap: int = 256
    block_n_cap: int = 256
    block_k_cap: int = 256
    group_ms: tuple[int, ...] = (8,)
    num_warps: tuple[int, ...] = (4, 8)
    num_stages: tuple[int, ...] = (2, 3, 4, 5)
    warp_size: int = 32
    register_limit_per_thread: int = 255

    def __post_init__(self) -> None:
        positive_values = {
            "block_m_min": self.block_m_min,
            "block_n_min": self.block_n_min,
            "block_k_min": self.block_k_min,
            "block_m_cap": self.block_m_cap,
            "block_n_cap": self.block_n_cap,
            "block_k_cap": self.block_k_cap,
            "warp_size": self.warp_size,
            "register_limit_per_thread": self.register_limit_per_thread,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.block_m_min > self.block_m_cap:
            raise ValueError("block_m_min must not exceed block_m_cap")
        if self.block_n_min > self.block_n_cap:
            raise ValueError("block_n_min must not exceed block_n_cap")
        if self.block_k_min > self.block_k_cap:
            raise ValueError("block_k_min must not exceed block_k_cap")
        if not self.group_ms or not self.num_warps or not self.num_stages:
            raise ValueError("group_ms, num_warps and num_stages must be non-empty")
        sequence_values = {
            "group_ms": self.group_ms,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }
        for name, values in sequence_values.items():
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must all be positive, got {values}")


def _shape_tuple(shape: object) -> tuple[int, int, int, int]:
    if hasattr(shape, "shape"):
        shape = getattr(shape, "shape")
    values = tuple(int(value) for value in shape)  # type: ignore[arg-type]
    if len(values) != 4 or any(value <= 0 for value in values):
        raise ValueError(f"expected a positive (B,M,N,K) shape, got {values}")
    return values  # type: ignore[return-value]


def generate_configs(
    shapes: Iterable[object],
    policy: SpacePolicy,
    *,
    quant_block_k: int | None = None,
    max_num_regs: int | None = None,
    max_shared_mem: int | None = None,
    fp8_itemsize: int = 1,
    static_resource_filter: bool = True,
) -> list[KernelConfig]:
    shape_values = [_shape_tuple(shape) for shape in shapes]
    if not shape_values:
        raise ValueError("at least one shape is required")
    if quant_block_k is not None and quant_block_k <= 0:
        raise ValueError(f"quant_block_k must be positive, got {quant_block_k}")

    max_m = max(policy.block_m_min, next_power_of_two(max(x[1] for x in shape_values)))
    max_n = max(policy.block_n_min, next_power_of_two(max(x[2] for x in shape_values)))
    max_k = max(policy.block_k_min, next_power_of_two(max(x[3] for x in shape_values)))
    block_ms = powers_of_two(policy.block_m_min, min(policy.block_m_cap, max_m))
    block_ns = powers_of_two(policy.block_n_min, min(policy.block_n_cap, max_n))
    block_ks = powers_of_two(policy.block_k_min, min(policy.block_k_cap, max_k))

    configs: list[KernelConfig] = []
    values = product(
        block_ms,
        block_ns,
        block_ks,
        policy.group_ms,
        policy.num_warps,
        policy.num_stages,
    )
    for block_m, block_n, block_k, group_m, num_warps, num_stages in values:
        if quant_block_k is not None:
            if block_k > quant_block_k or quant_block_k % block_k != 0:
                continue

        if static_resource_filter:
            threads = policy.warp_size * num_warps
            accumulator_regs_lb = cdiv(block_m * block_n, threads)
            if accumulator_regs_lb > policy.register_limit_per_thread:
                continue
            if (
                max_num_regs is not None
                and accumulator_regs_lb * threads > max_num_regs
            ):
                continue

            pipeline_smem_estimate = (
                num_stages * block_k * (block_m + block_n) * fp8_itemsize
            )
            if (
                max_shared_mem is not None
                and pipeline_smem_estimate > max_shared_mem
            ):
                continue

        configs.append(
            KernelConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=group_m,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs
