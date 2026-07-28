import os
import sys
import time
from dataclasses import dataclass

import torch
import triton
from triton.runtime import driver

# 让本文件无论被当脚本跑还是当模块跑都能找到 fp8_bmm 下的同级包
# （kernel/ 等）。fp8_bmm 是 namespace package，没有 __init__.py，
# 所以把它自身目录加进 sys.path，使 `from kernel...` 可解析。
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel_base
import occupancy_utils as utils


dev_id = 0
torch.cuda.set_device(dev_id)

prop = driver.active.utils.get_device_properties(dev_id)
sms = prop["multiprocessor_count"]
regs = prop["max_num_regs"]
smems = prop["max_shared_mem"]


# data_shapes = [
#     # B, M, N, K
#     (16, 512, 960, 1280),
#     (16, 2048, 640, 1280),
#     (16, 2048, 1280, 1280),
#     (16, 2048, 1280, 960),
#     (32, 2048, 640, 1280),
#     (32, 2048, 960, 1600),
#     (32, 2048, 1280, 1600),
#     (32, 2048, 1600, 1600),
#     (32, 2048, 1280, 960),
#     (80, 512, 640, 1280),
#     (80, 2048, 640, 640),
#     (80, 2048, 960, 640),
#     (80, 2048, 1280, 640),
#     (80, 2048, 1280, 960),
#     (80, 2048, 640, 1280),
# ]

# 参考 fp8_bmm_benchmark 的 data_meta 构造的 shape 集合（B, M, N, K），
# 用于和他同 shape 公平对比 TFLOPS。
data_shapes = [
    # B, M, N, K
    (80, 512, 1280, 960),
    (80, 512, 1280, 640),
    (80, 512, 960, 640),
    (32, 512, 1280, 960),
    (32, 512, 1280, 1600),
    (32, 512, 960, 1600),
    (16, 512, 1280, 160),
    (16, 512, 1280, 960),
    (16, 512, 1280, 1280),
    (16, 512, 960, 1280),
    (80, 512, 80, 640),
    (16, 512, 160, 1280),
    (80, 512, 640, 1280),
    (32, 512, 640, 1280),
    (16, 512, 640, 1280),
    (512, 512, 512, 512),
]


def cdiv(a, b):
    return (a + b - 1) // b


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def powers_of_2(begin, end):
    values = []
    x = next_power_of_2(begin)
    end = next_power_of_2(end)

    while x <= end:
        values.append(x)
        x *= 2

    return values


@dataclass(frozen=True)
class Config:
    Bm: int
    Bn: int
    Bk: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class CompileSignature:
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    c_dtype: torch.dtype
    bias_dtype: torch.dtype
    scale_dtype: torch.dtype
    use_bias: bool


@dataclass
class CompileResult:
    config: Config
    signature: CompileSignature
    sample_shape: tuple[int, int, int, int]

    accepted: bool

    n_regs: int = 0
    regs_per_cta_unrounded: int =0
    local_words_per_thread: int = 0
    local_bytes_per_thread: int = 0
    shared_bytes: int = 0

    compiled_num_warps: int = 0
    threads_per_cta: int = 0
    active_ctas_per_sm: int = 0

    compile_and_load_ms: float = 0.0
    reason: str = ""

    score: float = 0.0


def get_configs(data_shapes, prop):
    max_m = next_power_of_2(max(x[1] for x in data_shapes))
    max_n = next_power_of_2(max(x[2] for x in data_shapes))
    max_k = next_power_of_2(max(x[3] for x in data_shapes))

    Bms = powers_of_2(64, max_m)
    Bns = powers_of_2(8, max_n)
    Bks = powers_of_2(32, max_k)

    group_size_ms = (8,)
    warp_values = (4, 8)
    stage_values = (2, 3, 4, 5)

    configs = []

    for Bm in Bms:
        for Bn in Bns:
            for Bk in Bks:
                for num_warps in warp_values:
                    for num_stages in stage_values:
                        num_threads = 32 * num_warps

                        # 每线程 FP32 accumulator 寄存器数下界。
                        acc_regs_per_thread_lb = cdiv(
                            Bm * Bn,
                            num_threads,
                        )

                        # 整个 CTA 的寄存器需求下界。
                        regs_per_cta_lb = (
                            acc_regs_per_thread_lb
                            * num_threads
                        )

                        # 当前 Hopper pipeline family 中，
                        # 每个 stage 至少保存一份 FP8 A/B tile。
                        smem_pipeline_lb = (
                            num_stages
                            * Bk
                            * (Bm + Bn)
                        )

                        # accumulator 本身已经超过单线程寄存器上限。
                        if acc_regs_per_thread_lb > 255:
                            continue

                        # 只算 accumulator 都放不下。
                        if regs_per_cta_lb > prop["max_num_regs"]:
                            continue

                        # 只算 A/B pipeline buffer 都放不下。
                        if smem_pipeline_lb > prop["max_shared_mem"]:
                            continue

                        for group_size_m in group_size_ms:
                            configs.append(
                                Config(
                                    Bm=Bm,
                                    Bn=Bn,
                                    Bk=Bk,
                                    GROUP_SIZE_M=group_size_m,
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )

    return configs

def make_kernel_args(
    shape: tuple[int, int, int, int],
    signature: CompileSignature,
):
    batch, M, N, K = shape

    # 假设生产环境中的 A/B/C/bias 都是 contiguous：
    #
    # A:    [batch, M, K]
    # B:    [batch, K, N]
    # C:    [batch, M, N]
    # bias: [batch, M, N]
    return (
        signature.a_dtype,
        signature.b_dtype,
        signature.c_dtype,
        signature.bias_dtype,
        signature.scale_dtype,

        M,
        N,
        K,

        # A strides
        M * K,
        K,
        1,

        # B strides
        K * N,
        N,
        1,

        # C strides
        M * N,
        N,
        1,

        # bias strides
        M * N,
        N,
        1,
    )

def metadata_get(metadata, name, default=None):
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def initialize_compiled_kernel(compiled):
    """
    warmup 只编译。

    加载 cubin 后才会填充：
      compiled.n_regs
      compiled.n_spills

    注意：Triton 3.3 的 CompiledKernel 不再暴露 n_max_threads，
    该字段在更低版本才存在。
    """
    init_handles = getattr(compiled, "_init_handles", None)

    if init_handles is not None:
        init_handles()
    else:
        # 部分 Triton 版本访问 run 时触发 lazy load。
        _ = compiled.run

def compile_one(
    config: Config,
    shape: tuple[int, int, int, int],
    signature: CompileSignature,
    prop,
    reject_local_memory: bool = True,
) -> CompileResult:
    kernel_args = make_kernel_args(shape, signature)

    begin = time.perf_counter()

    compiled = batch_quant_fp8_mm_kernel_base.warmup(
        *kernel_args,
        grid=(1, 1),

        USE_BIASE=signature.use_bias,
        BLOCK_SIZE_M=config.Bm,
        BLOCK_SIZE_N=config.Bn,
        BLOCK_SIZE_K=config.Bk,
        GROUP_SIZE_M=8,

        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    initialize_compiled_kernel(compiled)

    elapsed_ms = (time.perf_counter() - begin) * 1000

    metadata = compiled.metadata

    # Warp specialization 后，编译后的 warp 数可能与输入值不同，
    # 所以这里读取 metadata 中的最终结果。
    compiled_num_warps = int(
        metadata_get(
            metadata,
            "num_warps",
            config.num_warps,
        )
    )

    warp_size = driver.active.get_current_target().warp_size
    threads_per_cta = compiled_num_warps * warp_size

    n_regs = int(compiled.n_regs)

    # Triton 3.4 中它是：
    # CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4
    local_words = int(compiled.n_spills)
    local_bytes = local_words * 4

    shared_bytes = int(
        metadata_get(metadata, "shared", 0)
    )

    active_ctas_per_sm = utils.get_active_ctas_per_sm(
        compiled,
        threads_per_cta=threads_per_cta,
        shared_bytes=shared_bytes,
    )

    reasons = []

    if reject_local_memory and local_words != 0:
        reasons.append(
            f"local memory = {local_bytes} bytes/thread"
        )

    if shared_bytes > prop["max_shared_mem"]:
        reasons.append(
            f"shared = {shared_bytes} > "
            f"{prop['max_shared_mem']}"
        )

    if n_regs > 255:
        reasons.append(
            f"registers = {n_regs}/thread > 255"
        )

    # CTA 级寄存器占用：单个 CTA 的总寄存器需求不能超过
    # 单个 SM 的寄存器文件总量（prop["max_num_regs"]）。
    # Hopper 上一个 CTA 最多可独占整个 SM 的寄存器文件，
    # 因此这是 per-CTA 的硬上限，超过则该 CTA 无法驻留。
    regs_per_cta = n_regs * threads_per_cta
    if regs_per_cta > prop["max_num_regs"]:
        reasons.append(
            f"regs/cta = {n_regs}/thread * "
            f"{threads_per_cta} threads = {regs_per_cta} > "
            f"{prop['max_num_regs']}"
        )

    return CompileResult(
        config=config,
        signature=signature,
        sample_shape=shape,
        accepted=not reasons,
        n_regs=n_regs,
        regs_per_cta_unrounded=regs_per_cta,
        local_words_per_thread=local_words,
        local_bytes_per_thread=local_bytes,
        shared_bytes=shared_bytes,
        compiled_num_warps=compiled_num_warps,
        threads_per_cta=threads_per_cta,
        active_ctas_per_sm=active_ctas_per_sm,
        compile_and_load_ms=elapsed_ms,
        reason="; ".join(reasons),
    )

def serial_compile_filter(
    configs: list[Config],
    data_shape,
    signature,
    prop,
    reject_local_memory: bool = True,
):
    survivors = []
    all_results = []

    for config in configs:
        try:
            result = compile_one(
                config=config,
                shape=data_shape,
                signature=signature,
                prop=prop,
                reject_local_memory=reject_local_memory,
            )
        except Exception as exc:
            result = CompileResult(
                config=config,
                signature=signature,
                sample_shape=data_shape,
                accepted=False,
                reason=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if result.accepted:
            survivors.append(result)

        all_results.append(result)

    return survivors, all_results

def to_triton_config(config: Config):
    return triton.Config(
        {
            "BLOCK_SIZE_M": config.Bm,
            "BLOCK_SIZE_N": config.Bn,
            "BLOCK_SIZE_K": config.Bk,
            "GROUP_SIZE_M": 8,
        },
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

def compute_efficiency(config, shape):
    _, M, N, K = shape
    padded_M = cdiv(M, config.Bm) * config.Bm
    padded_N = cdiv(N, config.Bn) * config.Bn
    padded_K = cdiv(K, config.Bk) * config.Bk

    return (
        M * N * K
        / (padded_M * padded_N * padded_K)
    )

def wave_metrics(result: CompileResult, shape):
    config = result.config
    B, M, N, _ = shape
    total_ctas = B * cdiv(M, config.Bm) * cdiv(N, config.Bn)
    capacity = sms * result.active_ctas_per_sm
    num_waves = cdiv(total_ctas, capacity)
    wave_eff = (
        total_ctas / (num_waves * capacity)
    )
    sm_fill = min(1.0, total_ctas / sms)
    return sm_fill, wave_eff

def filter_low_eff(survivor_results: list[CompileResult], data_shape):
    results = []
    for result in survivor_results:
        compute_eff = compute_efficiency(result.config, data_shape)
        sm_fill, wave_eff = wave_metrics(result, data_shape)
        if sm_fill < 0.5:
            continue
        result.score = compute_eff * wave_eff # 粗略打分
        results.append(result)
    final_size = min(64, len(results))
    results = sorted(results, key=lambda x: x.score, reverse=True)[:final_size]

def get_valid_configs():
    candidate_configs = get_configs(
        data_shapes,
        prop,
    )

    print(
        f"static candidates: "
        f"{len(candidate_configs)}"
    )

    # 必须换成生产环境的真实 dtype。
    signature = CompileSignature(
            a_dtype=torch.float8_e4m3fn,
            b_dtype=torch.float8_e4m3fn,
            c_dtype=torch.float16,
            bias_dtype=torch.float16,
            scale_dtype=torch.float32,
            use_bias=True,
        )
    
    survivor_results, compile_results = (
        serial_compile_filter(
            configs=candidate_configs,
            data_shape=data_shapes[0],
            signature=signature,
            prop=prop,
            reject_local_memory=True,
        )
    )

    print("after compiled: ", len(survivor_results))
    print("compiled total: ", len(compile_results))
    print("compiled failed: ", len(compile_results) - len(survivor_results))

    # for data_shape in data_shapes:
    #     filter_low_eff(survivor_results, data_shape)

    return [to_triton_config(result.config) for result in survivor_results]