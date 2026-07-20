#include "sgemv_framework.h"
#include <vector>

// BENCH_PIPE(kernel_name, template_args...)
// bench_pipe_##kernel_name(bench, grid, block)
// profiling_pipe_##kernel_name(bench, grid, block)

const int N = 8192;
const int M = 8192;

#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]
#define WARP_SIZE 32

__global__ void sgemv_naive(float *a, float *x, float *y, int N, int M) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = gx; r < N; r += gridDim.x * blockDim.x) {
        y[r] = 0.0f;
        for (int c = 0; c < M; c++) {
            y[r] += a[r * M + c] * x[c];
        }
    }
}

__global__ void sgemv_fp32x4_naive(float *a, float *x, float *y, int N, int M) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = gx; r < N; r += gridDim.x * blockDim.x) {
        y[r] = 0.0f;
        for (int c = 0; c < M; c += 4) {
            float4 reg_a, reg_x;
            reg_a = FLOAT4(a[r * M + c]);
            reg_x = FLOAT4(x[c]);
            y[r] += reg_a.x * reg_x.x; 
            y[r] += reg_a.y * reg_x.y; 
            y[r] += reg_a.z * reg_x.z; 
            y[r] += reg_a.w * reg_x.w; 
        }
    }
}

template<const int reduce_size>
__device__ __forceinline__ float warp_reduce(float x) {
#pragma unroll
    for (int mask = reduce_size >> 1; mask >= 1; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

template<const int block_size>
__global__ void sgemv_fp32x4_block_reduce(float *a, float *x, float *y, int N, int M) {
    const int NUM_WARPS = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float shmem[block_size];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float acc = 0.0f;
#pragma unroll 4
    for (int c = tid * 4; c < M; c += block_size * 4) {
        float4 reg_a, reg_x;
        reg_a = FLOAT4(a[r * M + c]);
        reg_x = FLOAT4(x[c]);
        acc += reg_a.x * reg_x.x;
        acc += reg_a.y * reg_x.y;
        acc += reg_a.z * reg_x.z;
        acc += reg_a.w * reg_x.w;
    }

    // block reduce
    acc = warp_reduce<WARP_SIZE>(acc);
    if (lane_id == 0) {
        shmem[warp_id] = acc;
    }
    __syncthreads();
        
    if (warp_id == 0) {
        acc = (lane_id < NUM_WARPS) ? shmem[lane_id] : 0.0f; 
        acc = warp_reduce<NUM_WARPS>(acc);
        if (lane_id == 0) {
            y[r] = acc;
        }
    }
}

BENCH_PIPE(sgemv_naive)
BENCH_PIPE(sgemv_fp32x4_naive)
BENCH_PIPE(sgemv_fp32x4_block_reduce, 128, 128)
BENCH_PIPE(sgemv_fp32x4_block_reduce, 256, 256)
BENCH_PIPE(sgemv_fp32x4_block_reduce, 512, 512)
BENCH_PIPE(sgemv_fp32x4_block_reduce, 1024, 1024)

int main() {
    SgemvBenchBase bench = SgemvBenchBase(N, M);
    // bench_pipe_sgemv_naive(bench, N, 256);
    // bench_pipe_sgemv_fp32x4_naive(bench, N, 256);
    bench_pipe_sgemv_fp32x4_block_reduce_128(bench, N, 128);
    bench_pipe_sgemv_fp32x4_block_reduce_256(bench, N, 256);
    bench_pipe_sgemv_fp32x4_block_reduce_512(bench, N, 512);
    bench_pipe_sgemv_fp32x4_block_reduce_1024(bench, N, 1024);
    // profiling_pipe_sgemv_fp32x4_block_reduce_256(bench, N, 256);
    // profiling_pipe_sgemv_fp32x4_block_reduce_512(bench, N, 512);
    // profiling_pipe_sgemv_fp32x4_block_reduce_1024(bench, N, 1024);
    return 0;
}