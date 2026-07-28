"""Profile the fp8 batch bmm kernel: dump PTX/TTGIR/LLIR and run for ncu.

用法:
    1. 编辑下方 ===== 手动编辑区 ===== 选 shape / config / B 主序。
    2. 直接跑一次，dump PTX 并做 sanity check:
           python3 profile_fp8_bmm.py
       PTX 等会写到 ./asm/<tag>.{ptx,ttgir,llir,ttir}。
    3. 用 ncu 抓报告 (.ncu-rep):
           ncu --set full -k batch_quant_fp8_mm_kernel_base \
               --target-processes all --launch-skip 0 --launch-count 1 \
               -o fp8_bmm_profile python3 profile_fp8_bmm.py
       生成 fp8_bmm_profile.ncu-rep，可用 ncu-ui 打开。
"""
import os
import sys

import torch
import triton

# fp8_bmm 是 namespace package，把它自身目录加进 sys.path 才能 `from kernel...`
_FP8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FP8_DIR not in sys.path:
    sys.path.insert(0, _FP8_DIR)

from kernel.fp8_quant_bmm_kernel import batch_quant_fp8_mm_kernel_base

dev_id = 0
torch.cuda.set_device(dev_id)


# ====================== 手动编辑区 ======================
SHAPE = (80, 2048, 640, 1280)   # B, M, N, K
B_N_ORDER = True               # True=N 主序(B 为 (B,K,N) 连续); False=K 主序(B 为 (B,N,K) 连续)
USE_BIAS = 1                   # 1=带 bias, 0=不带

BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 128
GROUP_SIZE_M = 8
NUM_WARPS = 8
NUM_STAGES = 3

PROFILE_ITERS = 5              # 给 ncu 抓取的 kernel launch 次数
# ========================================================


def cdiv(a, b):
    return (a + b - 1) // b


def make_tensors(shape):
    B, M, N, K = shape
    A = torch.randn(B, M, K, device="cuda").to(torch.float8_e4m3fn)
    B_mat = torch.randn(B, K, N, device="cuda").to(torch.float8_e4m3fn)
    if not B_N_ORDER:
        # K 主序: 物理布局改成 (B,N,K) 连续
        B_mat = B_mat.transpose(1, 2).contiguous()
    bias = torch.randn(B, M, N, device="cuda", dtype=torch.float16)
    C = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
    quant_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    return A, B_mat, bias, C, quant_scale


def build_launch(A, B, C, bias, quant_scale, shape):
    Bsz, M, N, K = shape
    BM, BN, BK, GPM = BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    grid = (cdiv(M, BM) * cdiv(N, BN), Bsz)

    # B 视作 (K,N): N 主序取 (b,k,n) stride; K 主序取 (b,n,k) 的转置 stride
    if B_N_ORDER:
        stride_bb, stride_bk, stride_bn = B.stride(0), B.stride(1), B.stride(2)
    else:
        stride_bb, stride_bk, stride_bn = B.stride(0), B.stride(2), B.stride(1)

    if USE_BIAS:
        bias_ptr = bias
        stride_biasb, stride_biasm, stride_biasn = bias.stride(0), bias.stride(1), bias.stride(2)
    else:
        bias_ptr = None
        stride_biasb = stride_biasm = stride_biasn = 0

    args = (
        A, B, C, bias_ptr, quant_scale, M, N, K,
        # A strides
        A.stride(0), A.stride(1), A.stride(2),
        # B strides
        stride_bb, stride_bk, stride_bn,
        # C strides
        C.stride(0), C.stride(1), C.stride(2),
        # bias strides
        stride_biasb, stride_biasm, stride_biasn,
    )
    consts = dict(
        USE_BIASE=USE_BIAS,
        B_N_ORDER=B_N_ORDER,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_SIZE_M=GPM,
    )

    # ---- dump PTX / TTGIR / LLIR / TTIR ----
    ck = batch_quant_fp8_mm_kernel_base.warmup(
        *args, grid=grid, num_warps=NUM_WARPS, num_stages=NUM_STAGES, **consts
    )
    asm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asm")
    os.makedirs(asm_dir, exist_ok=True)
    tag = f"{'n' if B_N_ORDER else 'k'}order_B{Bsz}_M{M}_N{N}_K{K}"
    for ext in ("ptx", "ttgir", "llir", "ttir"):
        content = getattr(ck, "asm", {}).get(ext)
        if content:
            path = os.path.join(asm_dir, f"{tag}.{ext}")
            with open(path, "w") as f:
                f.write(content)
            print(f"  dumped {ext:5s} -> {path}  ({len(content)} bytes)")

    def run():
        batch_quant_fp8_mm_kernel_base[grid](
            *args, num_warps=NUM_WARPS, num_stages=NUM_STAGES, **consts
        )

    return run


def main():
    B, M, N, K = SHAPE
    print(f"shape={SHAPE}  B_N_ORDER={B_N_ORDER}  USE_BIAS={USE_BIAS}")
    print(f"cfg: BM={BLOCK_SIZE_M},BN={BLOCK_SIZE_N},BK={BLOCK_SIZE_K},"
          f"GPM={GROUP_SIZE_M},warps={NUM_WARPS},stages={NUM_STAGES}")

    A, B_mat, bias, C, quant_scale = make_tensors(SHAPE)
    run = build_launch(A, B_mat, C, bias, quant_scale, SHAPE)

    # warmup 已由 build_launch 里的 warmup() 触发编译；这里再做一次 GPU 热身。
    run()
    torch.cuda.synchronize()

    for _ in range(PROFILE_ITERS):
        run()
    torch.cuda.synchronize()
    print(f"ran {PROFILE_ITERS} profiled launches")

    flops = 2 * B * M * N * K
    # 粗略计时（非 ncu 场景下看一眼）
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(20):
        run()
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / 20
    print(f"approx: {ms:.4f} ms  ({flops / ms / 1e9:.1f} TFLOPS)")

    print("\n========== ncu 命令 ==========")
    script = os.path.abspath(__file__)
    print(
        "ncu --set full -k batch_quant_fp8_mm_kernel_base "
        "--target-processes all --launch-skip 0 --launch-count 1 "
        f"-o fp8_bmm_profile python3 {script}"
    )
    print("(生成 fp8_bmm_profile.ncu-rep)")


if __name__ == "__main__":
    main()
