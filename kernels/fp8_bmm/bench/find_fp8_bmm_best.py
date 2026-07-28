import os
import sys
import json
import time

import torch
import triton

# 让本文件无论被当脚本跑还是当模块跑都能找到 fp8_bmm 下的同级包
# （kernel/ 等）。fp8_bmm 是 namespace package，没有 __init__.py，
# 所以把它自身目录加进 sys.path，使 `from kernel...` 可解析。
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

# bench/ 自身在脚本运行时位于 sys.path[0]，可直接 import 同级模块。
from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel_base
import filter_shape as fs


dev_id = 0
torch.cuda.set_device(dev_id)

# 手动切换 B 的主序用于搜索最佳 config:
#   True  -> N 主序: B 物理布局 (B,K,N) 连续, kernel 用 B_N_ORDER=True
#   False -> K 主序: B 物理布局 (B,N,K) 连续, kernel 用 B_N_ORDER=False
B_N_ORDER = False


def cdiv(a, b):
    return (a + b - 1) // b


def make_tensors(shape, signature):
    """按生产环境布局构造一份输入：A/B/bias contiguous，quant_scale 标量。"""
    B, M, N, K = shape

    A = torch.randn(B, M, K, device="cuda", dtype=torch.float32).to(
        signature.a_dtype
    )
    # B 用正态分布量化后值域受限，这里直接 randn 再转 fp8 即可（只用于计时）。
    B_mat = torch.randn(B, K, N, device="cuda", dtype=torch.float32).to(
        signature.b_dtype
    )
    bias = torch.randn(B, M, N, device="cuda", dtype=signature.bias_dtype)
    C = torch.empty(B, M, N, device="cuda", dtype=signature.c_dtype)

    # kernel 内部 `tl.load(quant_scale)` 取一个标量，传单元素 tensor。
    quant_scale = torch.tensor(
        [1.0], device="cuda", dtype=signature.scale_dtype
    )

    return A, B_mat, bias, C, quant_scale


def make_launcher(config, A, B, C, bias, quant_scale, shape, b_n_order=True):
    """返回一个无参 callable，固定用某个 config 启动 kernel，供 do_bench 计时。"""
    Bs, M, N, K = shape

    BM = config.kwargs["BLOCK_SIZE_M"]
    BN = config.kwargs["BLOCK_SIZE_N"]
    BK = config.kwargs["BLOCK_SIZE_K"]
    GPM = config.kwargs["GROUP_SIZE_M"]

    grid = (cdiv(M, BM) * cdiv(N, BN), Bs)

    if b_n_order:
        # N 主序: B 为 (B,K,N) 连续, kernel 视作 (K,N)、N 连续
        B_ptr = B
        stride_bb, stride_bk, stride_bn = B.stride(0), B.stride(1), B.stride(2)
    else:
        # K 主序: B 物理布局改为 (B,N,K) 连续, kernel 视作 (K,N)、K 连续
        B_ptr = B.transpose(1, 2).contiguous()
        stride_bb = B_ptr.stride(0)
        stride_bk = B_ptr.stride(2)  # K
        stride_bn = B_ptr.stride(1)  # N

    def run():
        batch_quant_fp8_mm_kernel_base[grid](
            A,
            B_ptr,
            C,
            bias,
            quant_scale,
            M,
            N,
            K,
            # A strides
            A.stride(0),
            A.stride(1),
            A.stride(2),
            # B strides
            stride_bb,
            stride_bk,
            stride_bn,
            # C strides
            C.stride(0),
            C.stride(1),
            C.stride(2),
            # bias strides
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            # constexpr / launch params
            USE_BIASE=1,
            B_N_ORDER=b_n_order,
            BLOCK_SIZE_M=BM,
            BLOCK_SIZE_N=BN,
            BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GPM,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    return run


def config_to_dict(config):
    return {
        "BLOCK_SIZE_M": config.kwargs["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config.kwargs["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config.kwargs["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config.kwargs["GROUP_SIZE_M"],
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def bench_one_shape(shape, configs, signature, b_n_order=True, warmup=25, rep=100, topk=5):
    """对单个 data_shape 跑全部 config，返回 (best_config, best_ms, ranked_list)。"""
    A, B, bias, C, quant_scale = make_tensors(shape, signature)

    ranked = []  # (ms, config_dict)
    n = len(configs)

    for i, config in enumerate(configs):
        run = make_launcher(config, A, B, C, bias, quant_scale, shape, b_n_order=b_n_order)

        # 某些 config 在该 shape 下可能仍会因显存/编译边界抛错，跳过。
        try:
            # 第一次调用触发 JIT 编译；do_bench 内部还会再做 warmup。
            run()
            ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
        except Exception as exc:
            print(
                f"  [skip] config #{i} {config_to_dict(config)}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        ranked.append((ms, config_to_dict(config)))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n} configs timed")

    ranked.sort(key=lambda x: x[0])

    if not ranked:
        raise RuntimeError(f"no config survived for shape {shape}")

    best_ms, best_cfg = ranked[0]
    return best_cfg, best_ms, ranked[:topk]


def main():
    signature = fs.CompileSignature(
        a_dtype=torch.float8_e4m3fn,
        b_dtype=torch.float8_e4m3fn,
        c_dtype=torch.float16,
        bias_dtype=torch.float16,
        scale_dtype=torch.float32,
        use_bias=True,
    )

    print("collecting valid configs via filter_shape ...")
    configs = fs.get_valid_configs()
    print(f"got {len(configs)} configs")

    b_n_order = B_N_ORDER
    order_tag = "n_order" if b_n_order else "k_order"
    print(f"B layout: {order_tag} (B_N_ORDER={b_n_order})")

    data_shapes = fs.data_shapes

    results = {}
    for shape in data_shapes:
        B, M, N, K = shape
        flops = 2 * B * M * N * K  # 2 ops per MAC
        print(f"\n=== shape (B={B}, M={M}, N={N}, K={K}) [{order_tag}] ===")

        begin = time.perf_counter()
        best_cfg, best_ms, top = bench_one_shape(shape, configs, signature, b_n_order=b_n_order)
        elapsed = time.perf_counter() - begin

        tflops = (flops / best_ms / 1e9) if best_ms > 0 else float("nan")

        print(f"best: {best_ms:.4f} ms  ({tflops:.1f} TFLOPS)")
        print(f"  cfg: {best_cfg}")
        print(f"  top{len(top)}:")
        for ms, cfg in top:
            print(f"    {ms:.4f} ms  {cfg}")
        print(f"  (sweep took {elapsed:.1f}s)")

        results[str(shape)] = {
            "shape": list(shape),
            "best_ms": best_ms,
            "best_config": best_cfg,
            "topk": [
                {"ms": ms, "config": cfg} for ms, cfg in top
            ],
        }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"fp8_quant_bmm_best_{order_tag}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {out_path}")

    print(f"\n========== summary [{order_tag}] ==========")
    print(f"{'shape':<28}{'best_ms':>10}    config")
    for shape in data_shapes:
        r = results[str(shape)]
        cfg = r["best_config"]
        cfg_str = (
            f"BM={cfg['BLOCK_SIZE_M']},BN={cfg['BLOCK_SIZE_N']},"
            f"BK={cfg['BLOCK_SIZE_K']},w={cfg['num_warps']},"
            f"s={cfg['num_stages']}"
        )
        print(f"{str(shape):<28}{r['best_ms']:>10.4f}    {cfg_str}")


if __name__ == "__main__":
    main()
