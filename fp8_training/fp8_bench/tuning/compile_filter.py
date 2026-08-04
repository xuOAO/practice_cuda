from __future__ import annotations

import ctypes
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from triton.runtime import driver

from fp8_bench.tuning.adapters import BMMTuningAdapter, TuningSpec
from fp8_bench.tuning.space import KernelConfig


def _metadata_get(metadata: Any, name: str, default: Any = None) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _initialize_compiled_kernel(compiled: Any) -> None:
    init_handles = getattr(compiled, "_init_handles", None)
    if init_handles is not None:
        init_handles()
    else:
        # Some Triton versions lazily load the cubin when ``run`` is accessed.
        _ = compiled.run


@dataclass(frozen=True)
class DeviceLimits:
    device: int
    name: str
    compute_capability: str
    sms: int
    max_num_regs: int
    max_shared_mem: int
    warp_size: int
    register_limit_per_thread: int = 255

    @classmethod
    def current(cls, device: int | None = None) -> DeviceLimits:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for kernel compilation and filtering")
        device = torch.cuda.current_device() if device is None else device
        properties = driver.active.utils.get_device_properties(device)
        torch_properties = torch.cuda.get_device_properties(device)
        target = driver.active.get_current_target()
        return cls(
            device=device,
            name=torch_properties.name,
            compute_capability=(
                f"{torch_properties.major}.{torch_properties.minor}"
            ),
            sms=int(
                properties.get(
                    "multiprocessor_count",
                    torch_properties.multi_processor_count,
                )
            ),
            max_num_regs=int(properties["max_num_regs"]),
            max_shared_mem=int(properties["max_shared_mem"]),
            warp_size=int(getattr(target, "warp_size", 32)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompileResult:
    config: KernelConfig
    compile_ok: bool
    accepted: bool
    uses_wgmma: bool | None = None
    n_regs: int = 0
    regs_per_cta: int = 0
    local_words_per_thread: int = 0
    local_bytes_per_thread: int = 0
    shared_bytes: int = 0
    compiled_num_warps: int = 0
    threads_per_cta: int = 0
    active_ctas_per_sm: int | None = None
    compile_and_load_ms: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["config"] = self.config.as_dict()
        return values


_OCCUPANCY_FUNCTION: Any = None


def _occupancy_function() -> Any:
    global _OCCUPANCY_FUNCTION
    if _OCCUPANCY_FUNCTION is None:
        cuda = ctypes.CDLL("libcuda.so.1")
        function = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
        function.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        _OCCUPANCY_FUNCTION = function
    return _OCCUPANCY_FUNCTION


def _active_ctas_per_sm(
    compiled: Any,
    *,
    threads_per_cta: int,
    shared_bytes: int,
) -> int:
    value = ctypes.c_int()
    status = _occupancy_function()(
        ctypes.byref(value),
        ctypes.c_void_p(int(compiled.function)),
        threads_per_cta,
        shared_bytes,
    )
    if status != 0:
        raise RuntimeError(f"CUDA occupancy query failed with status {status}")
    return value.value


def _assembly_uses_wgmma(compiled: Any) -> bool | None:
    assembly = getattr(compiled, "asm", None)
    if not assembly:
        return None
    if isinstance(assembly, dict):
        parts = assembly.values()
    else:
        parts = (assembly,)
    inspected = False
    for part in parts:
        if isinstance(part, bytes):
            continue
        text = str(part).lower()
        inspected = True
        if "wgmma" in text or "hgmma" in text:
            return True
    return False if inspected else None


def compile_one(
    *,
    adapter: BMMTuningAdapter,
    spec: TuningSpec,
    config: KernelConfig,
    limits: DeviceLimits,
    reject_local_memory: bool = True,
    require_wgmma: bool = True,
) -> CompileResult:
    started = time.perf_counter()
    try:
        compiled = adapter.compile(spec, config)
        _initialize_compiled_kernel(compiled)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        metadata = compiled.metadata
        compiled_num_warps = int(
            _metadata_get(metadata, "num_warps", config.num_warps)
        )
        threads_per_cta = compiled_num_warps * limits.warp_size
        n_regs = int(compiled.n_regs)
        local_words = int(getattr(compiled, "n_spills", 0) or 0)
        local_bytes = local_words * 4
        shared_bytes = int(_metadata_get(metadata, "shared", 0))
        regs_per_cta = n_regs * threads_per_cta
        uses_wgmma = _assembly_uses_wgmma(compiled)

        reasons: list[str] = []
        if reject_local_memory and local_words:
            reasons.append(f"local memory={local_bytes} bytes/thread")
        if shared_bytes > limits.max_shared_mem:
            reasons.append(
                f"shared={shared_bytes} > limit={limits.max_shared_mem}"
            )
        if n_regs > limits.register_limit_per_thread:
            reasons.append(
                f"registers={n_regs}/thread > "
                f"limit={limits.register_limit_per_thread}"
            )
        if regs_per_cta > limits.max_num_regs:
            reasons.append(
                f"registers/CTA={regs_per_cta} > limit={limits.max_num_regs}"
            )
        if require_wgmma and uses_wgmma is not True:
            reason = "not found" if uses_wgmma is False else "assembly unavailable"
            reasons.append(f"WGMMA/HGMMA {reason}")

        active_ctas: int | None
        try:
            active_ctas = _active_ctas_per_sm(
                compiled,
                threads_per_cta=threads_per_cta,
                shared_bytes=shared_bytes,
            )
            if active_ctas <= 0:
                reasons.append(f"active CTAs/SM={active_ctas}")
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            active_ctas = None
            reasons.append(f"occupancy unavailable: {type(exc).__name__}: {exc}")

        return CompileResult(
            config=config,
            compile_ok=True,
            accepted=not reasons,
            uses_wgmma=uses_wgmma,
            n_regs=n_regs,
            regs_per_cta=regs_per_cta,
            local_words_per_thread=local_words,
            local_bytes_per_thread=local_bytes,
            shared_bytes=shared_bytes,
            compiled_num_warps=compiled_num_warps,
            threads_per_cta=threads_per_cta,
            active_ctas_per_sm=active_ctas,
            compile_and_load_ms=elapsed_ms,
            reason="; ".join(reasons),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        message = f"{type(exc).__name__}: {exc}"
        return CompileResult(
            config=config,
            compile_ok=False,
            accepted=False,
            compile_and_load_ms=elapsed_ms,
            reason=message[-2000:],
        )


def compile_filter(
    *,
    adapter: BMMTuningAdapter,
    spec: TuningSpec,
    configs: Iterable[KernelConfig],
    limits: DeviceLimits,
    reject_local_memory: bool = True,
    require_wgmma: bool = True,
) -> tuple[list[CompileResult], list[CompileResult]]:
    all_results = [
        compile_one(
            adapter=adapter,
            spec=spec,
            config=config,
            limits=limits,
            reject_local_memory=reject_local_memory,
            require_wgmma=require_wgmma,
        )
        for config in configs
    ]
    return [result for result in all_results if result.accepted], all_results
