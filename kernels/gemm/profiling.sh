#!/bin/bash
# Usage: profiling.sh exec_file bench_label [ncu_kernel_regex]
# Example: profiling.sh ./gemm opt0_64_64_8
#          profiling.sh ./gemm cublas_notc
#   -> 设置 PROFILE_KERNEL=bench_label, 程序只对选中的 kernel 走 profiling_pipe 单次启动,
#      其余 kernel 跳过; ncu 抓取单次 profile 到 profiling/<label>.ncu-rep
#
# bench_label 是程序里的 bench 名 (BT_CALL 里的 suffix_Bm_Bn_8, 或 cublas_notc/cublas_tc).
# 默认不传 -k: profile 模式下程序只启动选中的那一个 kernel, ncu 直接抓它.
# 第 3 参数可选, 传正则手动过滤 (ncu --kernel-name 是整串匹配, 写全名如 gemm_blockTile_opt0).

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 exec_file bench_label [ncu_kernel_regex]"
    exit 1
fi

EXEC_FILE=$1
LABEL=$2
# profile 模式下程序只启动选中的那一个 kernel, 所以默认不传 -k, ncu 直接抓唯一一次启动.
# 若要手动过滤可用第 3 参数传正则 (注意 ncu --kernel-name 是整串匹配, 需写全名如 gemm_blockTile_opt0).
KERNEL_REGEX=${3:-}

mkdir -p profiling

if [ -n "$KERNEL_REGEX" ]; then
    PROFILE_KERNEL="$LABEL" ncu --set full --import-source yes -f \
        -k "$KERNEL_REGEX" \
        -o "./profiling/${LABEL}" \
        -- "$EXEC_FILE"
else
    PROFILE_KERNEL="$LABEL" ncu --set full --import-source yes -f \
        -o "./profiling/${LABEL}" \
        -- "$EXEC_FILE"
fi
