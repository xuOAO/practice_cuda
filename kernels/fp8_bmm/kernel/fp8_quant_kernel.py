import triton
import triton.language as tl

BLOCK_SIZE_M_SET = [32, 64]
BLOCK_SIZE_N_SET = [32, 64, 128, 256]
NUM_STAGES_SET = [1, 2, 3, 4, 7]
NUM_WARPS_SET = [4, 8]

configs = [
    triton.Config(
        {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN},
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )
    for BM in BLOCK_SIZE_M_SET
    for BN in BLOCK_SIZE_N_SET
    for NUM_STAGES in NUM_STAGES_SET
    for NUM_WARPS in NUM_WARPS_SET
]

@triton.jit
def batch_per_matrix_fp8_quant_kernel_base(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_yb,
    stride_ym,
    stride_yn,
    scale,
    fp8_range : tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    DIM : tl.constexpr,
):
    if DIM == 2:
        pid = tl.program_id(axis=0)
    if DIM == 3:
        bid = tl.program_id(axis=1)
        pid = tl.program_id(axis=0)

        x_ptr = x_ptr + bid * stride_xb
        y_ptr = y_ptr + bid * stride_yb

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_off = off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    y_off = off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
    mask = (off_m < M)[:, None] & (off_n < N)[None, :]
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    quant_factor = tl.load(scale)
    value = tl.clamp(x.to(tl.float32) * quant_factor, min=-fp8_range, max=fp8_range)
    value = value.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + y_off, value, mask=mask)

batch_per_matrix_fp8_quant_kernel = triton.autotune(configs=configs, key=["M", "N"])(
    batch_per_matrix_fp8_quant_kernel_base
)

batch_per_matrix_fp8_quant_kernel_test = triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32})],
    key=["M", "N"],
)(
    batch_per_matrix_fp8_quant_kernel_base
)