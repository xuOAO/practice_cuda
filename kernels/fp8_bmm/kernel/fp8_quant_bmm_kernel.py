import triton
import triton.language as tl

# BLOCK_SIZE_M_SET = [64, 128]
# BLOCK_SIZE_N_SET = [128, 256]
# BLOCK_SIZE_K_SET = [16, 32]
# GROUP_SIZE_M_SET = [8]
# NUM_STAGES_SET = [3, 4]
# NUM_WARPS_SET = [4, 8]

# configs = [
#     triton.Config(
#         {
#             "BLOCK_SIZE_M": BM,
#             "BLOCK_SIZE_N": BN,
#             "BLOCK_SIZE_K": BK,
#             "GROUP_SIZE_M": GPM,
#         },
#         num_stages=NUM_STAGES,
#         num_warps=NUM_WARPS,
#     )
#     for BM in BLOCK_SIZE_M_SET
#     for BN in BLOCK_SIZE_N_SET
#     for BK in BLOCK_SIZE_K_SET
#     for GPM in GROUP_SIZE_M_SET
#     for NUM_STAGES in NUM_STAGES_SET
#     for NUM_WARPS in NUM_WARPS_SET
# ]


@triton.jit
def batch_quant_fp8_mm_kernel_base(
    A,
    B,
    C,
    bias,
    quant_scale,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    stride_biasm,
    stride_biasn,
    USE_BIASE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    B_N_ORDER: tl.constexpr=True,
):
    bid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_group) % group_size_m)
    pid_n = (pid % num_pid_group) // group_size_m

    a_ptr = tl.make_block_ptr(
        A + bid * stride_ab,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_ptr = tl.make_block_ptr(
        B + bid * stride_bb,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0) if B_N_ORDER else (0, 1),
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in tl.range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        b = tl.load(
            b_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
        b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))
    
    quant_factor = tl.load(quant_scale)
    accumulator = quant_factor * accumulator

    if USE_BIASE:
        bias_ptr = tl.make_block_ptr(
            bias + bid * stride_biasb,
            shape=(M, N),
            strides=(stride_biasm, stride_biasn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0)
        )
        bbias = tl.load(
            bias_ptr,
            boundary_check=(1, 0),
            padding_option="zero",
        )
        accumulator += bbias
    
    c = accumulator.to(C.dtype.element_ty)

    c_ptr = tl.make_block_ptr(
        C + bid * stride_cb,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_ptr, c, boundary_check=(1, 0))



# batch_quant_fp8_mm_kernel = triton.autotune(configs=configs, key=["M", "N", "K"])(
#     batch_quant_fp8_mm_kernel_base
# )
configs = [
triton.Config(
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
    },
    num_warps=8,
    num_stages=3,
), 
triton.Config(
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
    },
    num_warps=4,
    num_stages=4,
),
triton.Config(
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
    },
    num_warps=4,
    num_stages=3,
),
triton.Config(
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
    },
    num_warps=4,
    num_stages=4,
),
]

# 从 find_fp8_bmm_best 的 sweep 结果里去重出的 per-shape 最佳 config 集合
# (GROUP_SIZE_M 固定 8)。autotune 会按 key 在这 9 个里选最快的。
# 注意 key 里带了 B_N_ORDER 和 USE_BIASE：这两种 constexpr 会让 kernel 编译成不同
# 二进制，最优 config 也不同，必须分开 autotune，否则 n/k 主序会共用 cache 互相污染。
configs = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
]

batch_quant_fp8_mm_kernel = triton.autotune(
    configs=configs, key=["M", "N", "K", "B_N_ORDER", "USE_BIASE"],
)(
    batch_quant_fp8_mm_kernel_base
)

batch_quant_fp8_mm_kernel_test = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=8,
        )
    ],
    key=["M", "N", "K"],
)(
    batch_quant_fp8_mm_kernel_base
)
    