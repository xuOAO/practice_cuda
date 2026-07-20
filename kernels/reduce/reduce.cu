#include "reduce_framework.h"
#include "common.h"

#include <cassert>

// BENCH_PIPE(kernel_name, optional(suffix), template_args...)
// bench_pipe_##kernel_name_suffix(bench, grid, block)
// profiling_pipe_##kernel_name_suffix(bench, grid, block)

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]

const int dev_id = 0;
const uint SMs = GetGpuInfo(std::string("sm_count"), dev_id);
const uint max_threads_per_sm = GetGpuInfo(std::string("max_threads_per_sm"), dev_id);

template<const int REDUCE_SIZE>
__device__ __forceinline__ float warp_reduce(float x) {
#pragma unroll
    for (int mask = REDUCE_SIZE >> 1; mask >= 1; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

__global__ void reduce_thread(float *x, float *y, int N, int M) {
    //TODO
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = gx; r < N; r += gridDim.x * blockDim.x) {
        float4 reg_x;
        float sum = 0.0f;
        for (int c = 0; c < M; c += 4) {
            reg_x = FLOAT4(x[r * M + c]);
            sum += reg_x.x;
            sum += reg_x.y;
            sum += reg_x.z;
            sum += reg_x.w;
        }
        y[r] = sum;
    }
}

__global__ void reduce_warp(float *x, float *y, int N, int M) {
    //TODO
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_warps = gridDim.x * (blockDim.x) / WARP_SIZE;
    int warp_id = gx / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

#pragma unroll 1
    for (int r = warp_id; r < N; r += nb_warps) {
        float sum = 0.0f;
        // float4 reg_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 reg_tmp;
        for (int c = lane_id * 4; c < M; c += WARP_SIZE * 4) {
            reg_tmp = FLOAT4(x[r * M + c]);
            sum += reg_tmp.x;
            sum += reg_tmp.y;
            sum += reg_tmp.z;
            sum += reg_tmp.w;
            // reg_x.x += reg_tmp.x;
            // reg_x.y += reg_tmp.y;
            // reg_x.z += reg_tmp.z;
            // reg_x.w += reg_tmp.w;
        }
        // float sum = 0.0f;
        // sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
        sum = warp_reduce<WARP_SIZE>(sum);
        y[r] = sum;
    }
}

template<const int BLOCK_SIZE>
__global__ void reduce_block(float *x, float *y, int N, int M) {
    //TODO
    const int nb_warps_per_block = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shmem[nb_warps_per_block];
    int bid = blockIdx.x;
    int nb_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE; 
    int lane_id = threadIdx.x % WARP_SIZE;

    for (int r = bid; r < N; r += nb_blocks) {
        // float4 reg_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float sum = 0.0;
        float4 reg_tmp;
#pragma unroll 2
        for (int c = threadIdx.x * 4; c < M; c += BLOCK_SIZE * 4) {
            reg_tmp = FLOAT4(x[r * M + c]);
            sum += reg_tmp.x;
            sum += reg_tmp.y;
            sum += reg_tmp.z;
            sum += reg_tmp.w;
            // reg_x.x += reg_tmp.x;
            // reg_x.y += reg_tmp.y;
            // reg_x.z += reg_tmp.z;
            // reg_x.w += reg_tmp.w;
        }
        // float sum = 0.0f;
        // sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w; 
        sum = warp_reduce<32>(sum);
        shmem[warp_id] = sum;
        __syncthreads();

        if (warp_id == 0) {
            sum = (lane_id < nb_warps_per_block) ? shmem[lane_id] : 0.0f;
            sum = warp_reduce<nb_warps_per_block>(sum);
            if (lane_id == 0) {
                y[r] = sum;
            }
        }
    }
}

// 严格要求cudaMemset(y, 0, N * sizeof(float))
// 无法测速，所以不参与测试了
template<const int BLOCK_SIZE, const int REDUCE_SIZE>
__global__ void reduce_grid(float *x, float *y, int N, int M) {
    //TODO
    const int nb_warps_per_block = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shmem[BLOCK_SIZE / WARP_SIZE];
    assert(gridDim.x % REDUCE_SIZE == 0);
    int nb_workers = gridDim.x / REDUCE_SIZE;
    int global_worker_id = blockIdx.x / REDUCE_SIZE;
    int local_worker_id = blockIdx.x % REDUCE_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    for (int r = global_worker_id; r < N; r += nb_workers) {
        float4 reg_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 reg_tmp;
        int off_c = local_worker_id * BLOCK_SIZE * 4;
        int stride = REDUCE_SIZE * BLOCK_SIZE * 4;
        for (int c = threadIdx.x * 4 + off_c; c < M; c += stride) {
            reg_tmp = FLOAT4(x[r * M + c]);
            reg_x.x += reg_tmp.x;
            reg_x.y += reg_tmp.y;
            reg_x.z += reg_tmp.z;
            reg_x.w += reg_tmp.w;
        }
        float sum = 0.0f;
        sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w; 
        sum = warp_reduce<32>(sum);
        shmem[warp_id] = sum;
        __syncthreads();

        if (warp_id == 0) {
            sum = (lane_id < nb_warps_per_block) ? shmem[lane_id] : 0.0f;
            sum = warp_reduce<nb_warps_per_block>(sum);
            if (lane_id == 0) {
                atomicAdd(y + r, sum);
            }
        }
    }
}

BENCH_PIPE(reduce_thread)
BENCH_PIPE(reduce_warp)
BENCH_PIPE(reduce_block, block128, 128)
BENCH_PIPE(reduce_block, block256, 256)
BENCH_PIPE(reduce_block, block512, 512)
BENCH_PIPE(reduce_block, block1024, 1024)

std::vector<int> sizes_1d = {
    128, 256, 512,              
    1024, 2048, 4096,           
    8192, 16384, 32768,         
};

void test_vec_reduce(bool write_csv) {
    BenchBase::OpenCsv("figures/reduce_one_line.csv");
    dim3 grid = dim3(1);
    for (auto& sz : sizes_1d) {
        ReduceBenchBase bench = ReduceBenchBase(1, sz, false);
        // 性能在任何数据规模都打不过
        // bench_pipe_reduce_thread(bench, grid, 32);
        bench_pipe_reduce_warp(bench, grid, 32);
        bench_pipe_reduce_block_block128(bench, grid, 128);
        bench_pipe_reduce_block_block256(bench, grid, 256);
        bench_pipe_reduce_block_block512(bench, grid, 512);
        bench_pipe_reduce_block_block1024(bench, grid, 1024);
    }
    BenchBase::CloseCsv();
}

void test_best_grid() {
    // const int s = 64 * SMs;
    const int s = 8192;
    const int k = 8192;
    std::vector<dim3> grids = {
        {SMs}, {SMs * 2}, {SMs * 4},
        {SMs * 8}, {SMs * 16}, {SMs * 32},
        {SMs * 64},
    };
    ReduceBenchBase bench = ReduceBenchBase(s, k, false);
    for (auto& grid : grids) {
        //慢中慢，不需要测，访存都不合并的
        // bench_pipe_reduce_thread(bench, grid, 512);
        // -> occ高的配置即为好的配置
        // bench_pipe_reduce_warp(bench, grid, 512);
        bench_pipe_reduce_block_block512(bench, grid, 512);
        bench_pipe_reduce_block_block1024(bench, grid, 1024);
        // profiling_pipe_reduce_warp(bench, grid, 512);
        // profiling_pipe_reduce_block_block512(bench, grid, 512);
    }
}

void test_grid() {
    // const int s = 64 * SMs;
    const int s = 1024;
    const int k = 8192;
    ReduceBenchBase bench = ReduceBenchBase(s, k, false);
    bench_pipe_reduce_warp(bench, (s + 3) / 4, 128);
    bench_pipe_reduce_block_block512(bench, 1024, 512);
    bench_pipe_reduce_block_block1024(bench, 1024, 1024);
    // profiling_pipe_reduce_warp(bench, 1024, 128);
    // profiling_pipe_reduce_block_block512(bench, grid, 512);
}

void test_variable_sizes(bool write_csv) {
    BenchBase::OpenCsv("figures/reduce_variable_sizes_results.csv");
    const int k = 8192;
    std::vector<int> Ss = {
        1, 2, 4, 8, 16, 32, 64, 128, 256,
        512, 1024, 2048, 4096, 8192, 16384, 
    };
    dim3 warp_block = 32;
    for (auto& s: Ss) {
        int min_warp_grid = s;
        int min_block_grid = s;
        ReduceBenchBase bench = ReduceBenchBase(s, k, false);
        bench_pipe_reduce_warp(bench, min_warp_grid, warp_block);
        bench_pipe_reduce_block_block128(bench, min_block_grid, 128);
        bench_pipe_reduce_block_block256(bench, min_block_grid, 256);
        bench_pipe_reduce_block_block512(bench, min_block_grid, 512);
        bench_pipe_reduce_block_block1024(bench, min_block_grid, 1024);
    }
    BenchBase::CloseCsv();
}

int main() {
    PrintGpuInfo();
    // test_vec_reduce(true);
    // test_best_grid();
    test_grid();
    // test_variable_sizes(true);
}