#!/bin/bash
# Usage: profiling.sh exec_file kernel_name [suffix]
# Example: profiling.sh ./mat_transpose f32x4_shared_swizzle_16x16_row2col
#          profiling.sh ./reduce reduce_warp b32   # -> profiling/reduce_warp_b32.ncu-rep

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 exec_file kernel_name [suffix]"
    exit 1
fi

EXEC_FILE=$1
KERNEL_NAME=$2
SUFFIX=${3:-}

mkdir -p profiling

OUTPUT_NAME="${KERNEL_NAME}"
if [ -n "${SUFFIX}" ]; then
    OUTPUT_NAME="${KERNEL_NAME}_${SUFFIX}"
fi

ncu --set full --import-source yes -f \
    -o "./profiling/${OUTPUT_NAME}" \
    -k "${KERNEL_NAME}" \
    -- "${EXEC_FILE}"
