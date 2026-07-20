#include "gemm_framework.h"
#include "common.h"
#include <vector>

// BENCH_PIPE(kernel_name, optional(suffix), template_args...)
// bench_pipe_##kernel_name[_suffix](bench, grid, block)
// profiling_pipe_##kernel_name[_suffix](bench, grid, block)

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]

const int dev_id = 0;
const uint SMs = GetGpuInfo(std::string("sm_count"), dev_id);
const uint max_threads_per_sm = GetGpuInfo(std::string("max_threads_per_sm"), dev_id);

// C[M, N] = A[M, K] @ B[K, N], row-major.
const int M = 4096;
const int N = 4096;
const int K = 4096;
const bool check_correctness = false;

// ----------------------------------------------------------------------------
// 基线: 每个线程计算 C 中一个元素. 无共享内存, A 的行被反复读, B 的列每次跨步访存.
// grid = (ceildiv(N, BX), ceildiv(M, BY)), block = (BX, BY)
// ----------------------------------------------------------------------------
__global__ void gemm_naive(float *a, float *b, float *c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        acc = acc + a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = acc;
}

template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16>
__global__ void gemm_blockTile_opt0(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a[Bm][Bk];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    const int c_thread_x = threadIdx.x % C_BLOCK_X;
    const int c_thread_y = threadIdx.x / C_BLOCK_X;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                tile_a[i][j] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    int a_row = i * C_BLOCK_Y + c_thread_y;
                    int b_col = j * C_BLOCK_X + c_thread_x;
                    acc[i][j] += tile_a[a_row][pi] * tile_b[pi][b_col];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + i * C_BLOCK_Y + c_thread_y;
            int col = start_col + j * C_BLOCK_X + c_thread_x;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt1: mma的operator使用寄存器缓存，减少smem读取次数
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16>
__global__ void gemm_blockTile_opt1(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a[Bm][Bk];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    const int c_thread_x = threadIdx.x % C_BLOCK_X;
    const int c_thread_y = threadIdx.x / C_BLOCK_X;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                tile_a[i][j] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            // 减少smem的读取，reg稍微快于smem，但是可能增大寄存器压力
            float cache_a[m_elems];
            float cache_b[n_elems];

            // 每个warp读取的smem次数 = 2 * m_elems + 16 * n_elems
            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                // n-broadcast，n取决于warp大小，当前n = 16
                int a_row = i * C_BLOCK_Y + c_thread_y;
                cache_a[i] = tile_a[a_row][pi];
            }

            #pragma unroll
            for (int i = 0; i < n_elems; i++) {
                // n-broadcast，n取决于warp大小，当前n = 2 
                int b_col = i * C_BLOCK_X + c_thread_x;
                cache_b[i] = tile_b[pi][b_col];
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + i * C_BLOCK_Y + c_thread_y;
            int col = start_col + j * C_BLOCK_X + c_thread_x;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt2: 重排warp，减少smem读取次数
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt2(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a[Bm][Bk];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_x = lane_id % C_WARP_X;
    const int lane_y = lane_id / C_WARP_X;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                tile_a[i][j] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            // 减少smem的读取，reg稍微快于smem，但是可能增大寄存器压力
            float cache_a[m_elems];
            float cache_b[n_elems];

            // 重排block内的warp，可以降低smem读取次数
            // 每个warp读取的smem次数 = C_WARP_Y * m_elems + C_WARP_X * n_elems 
            // 假设 Bm / C_BLOCK_Y == Bn / C_BLOCK_X
            //  -> (C_WARP_X, C_WARP_Y) = (8, 4) or (4, 8) 时，smem读取次数最少
            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                int a_row = i * C_BLOCK_Y + c_thread_y;
                cache_a[i] = tile_a[a_row][pi];
            }

            #pragma unroll
            for (int i = 0; i < n_elems; i++) {
                int b_col = i * C_BLOCK_X + c_thread_x;
                cache_b[i] = tile_b[pi][b_col];
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + i * C_BLOCK_Y + c_thread_y;
            int col = start_col + j * C_BLOCK_X + c_thread_x;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt3: 使用LD128读取smem，减少指令使用
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt3(float *a, float *b, float *c, int M, int N, int K) {
    // 修改成transpose情况，更适合smem使用LD128
    __shared__ float tile_a_mT[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_x = lane_id % C_WARP_X;
    const int lane_y = lane_id / C_WARP_X;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                // i 和 j 互换
                // 但是会加大bank-conflict，由4-way变为8-way
                tile_a_mT[j][i] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            float cache_a[m_elems];
            float cache_b[n_elems];

            static_assert(m_elems % 4 == 0);
            static_assert(n_elems % 4 == 0);
            const int m_4elems = m_elems / 4;
            const int n_4elems = n_elems / 4;

            // 使用LD128指令
            // 实际上改变了每个thread负责的元素分布
            #pragma unroll
            for (int i = 0; i < m_4elems; i++) {
                int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
                FLOAT4(cache_a[i * 4]) = FLOAT4(tile_a_mT[pi][a_row]);
            }

            #pragma unroll
            for (int i = 0; i < n_4elems; i++) {
                int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
                FLOAT4(cache_b[i * 4]) = FLOAT4(tile_b[pi][b_col]);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt4: 使用siwwzle解决tile_a_mT读取时的bank-conflict
__device__ __forceinline__ int swizzle_tile_a_opt4(int lr, int lc) {
    return lr ^ (lc << 2);
}

template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt4(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a_mT[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_x = lane_id % C_WARP_X;
    const int lane_y = lane_id / C_WARP_X;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                // swizzle
                tile_a_mT[j][swizzle_tile_a_opt4(i, j)] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                // 无冲突，不需要swizzle
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            float cache_a[m_elems];
            float cache_b[n_elems];

            static_assert(m_elems % 4 == 0);
            static_assert(n_elems % 4 == 0);
            const int m_4elems = m_elems / 4;
            const int n_4elems = n_elems / 4;

            #pragma unroll
            for (int i = 0; i < m_4elems; i++) {
                int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
                FLOAT4(cache_a[i * 4]) = FLOAT4(tile_a_mT[pi][swizzle_tile_a_opt4(a_row, pi)]);
            }

            #pragma unroll
            for (int i = 0; i < n_4elems; i++) {
                int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
                FLOAT4(cache_b[i * 4]) = FLOAT4(tile_b[pi][b_col]);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

/* opt5: warp内线程z-order重排，更好的利用smem的广播机制
    对于LD64和LD128出发smem的广播机制参考：
    https://code.hitori.moe/post/cuda-shared-memory-access-mechanism-with-vectorized-instructions/
    总的来说，1.warp内所有i号线程和i ^ 1号线程访问的数据相同则可以广播 或者 2.warp内所有i号线程和i ^ 1号线程访问的数据相同则可以广播
    c_warp 线程排布如下：
         0,  1,  2,  3, ...,  7
         8,  9, 10, 11, ..., 15
        16, 17, 18, 19, ..., 23
        24, 25, 26, 27, ..., 31

    对于加载 a_cache，每一行访问相同的数据，满足条件 1. 触发广播，实际smem transaction = 2次
    对于加载 b_cache，每一列访问相同的数据，但每一列各线程跨步8，不触发广播，实际smem transaction = 4次

    考虑重排z-order：
    令每行bit0相同，每列bit1相同，5bit剩下3bit可变，低2bit可组成4组
        00, 01
        10, 11 (4组，每组8个元素，可知组内形状为 2x4)
    z-order后，线程拍不如下：
         0,  2,  4,  8, ..., 14
         1,  3,  5,  7, ..., 15
        16, 18, 20, 22, ..., 30
        17, 19, 21, 23, ..., 31 

    对于加载 a_cache，每一行访问相同的数据，满足条件 2. 触发广播，实际smem transaction = 2次
    对于加载 b_cache，每一列访问相同的数据，满足条件 1. 触发广播，实际smem transaction = 2次
*/
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt5(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a_mT[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // z-order重排
    const int lane_x = lane_id % 16 / 2;
    const int lane_y = lane_id / 16 * 2 + lane_id % 2;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    for (int ki = 0; ki < K; ki += Bk) {

        // load
        #pragma unroll
        for (int i = a_thread_y; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = a_thread_x; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i;
                int col = ki + j;
                // swizzle
                tile_a_mT[j][swizzle_tile_a_opt4(i, j)] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = b_thread_y; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = b_thread_x; j < Bn; j += B_BLOCK_X) {
                int row = ki + i;
                int col = start_col + j;
                // 无冲突，不需要swizzle
                tile_b[i][j] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            float cache_a[m_elems];
            float cache_b[n_elems];

            static_assert(m_elems % 4 == 0);
            static_assert(n_elems % 4 == 0);
            const int m_4elems = m_elems / 4;
            const int n_4elems = n_elems / 4;

            #pragma unroll
            for (int i = 0; i < m_4elems; i++) {
                int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
                FLOAT4(cache_a[i * 4]) = FLOAT4(tile_a_mT[pi][swizzle_tile_a_opt4(a_row, pi)]);
            }

            #pragma unroll
            for (int i = 0; i < n_4elems; i++) {
                int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
                FLOAT4(cache_b[i * 4]) = FLOAT4(tile_b[pi][b_col]);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt6: 优化代码书写
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt6(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a_mT[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // z-order重排
    const int lane_x = lane_id % 16 / 2;
    const int lane_y = lane_id / 16 * 2 + lane_id % 2;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    #pragma unroll
    for (int ki = 0; ki < K; ki += Bk) {

        // 更合适的写法，让展开循环更顺畅
        #pragma unroll
        for (int i = 0; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i + a_thread_y;
                int col = ki + j + a_thread_x;
                // swizzle
                tile_a_mT[j + a_thread_x][swizzle_tile_a_opt4(i + a_thread_y, j + a_thread_x)] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = 0; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bn; j += B_BLOCK_X) {
                int row = ki + i + b_thread_y;
                int col = start_col + j + b_thread_x;
                // 无冲突，不需要swizzle
                tile_b[i + b_thread_y][j + b_thread_x] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            float cache_a[m_elems];
            float cache_b[n_elems];

            static_assert(m_elems % 4 == 0);
            static_assert(n_elems % 4 == 0);
            const int m_4elems = m_elems / 4;
            const int n_4elems = n_elems / 4;

            #pragma unroll
            for (int i = 0; i < m_4elems; i++) {
                int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
                FLOAT4(cache_a[i * 4]) = FLOAT4(tile_a_mT[pi][swizzle_tile_a_opt4(a_row, pi)]);
            }

            #pragma unroll
            for (int i = 0; i < n_4elems; i++) {
                int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
                FLOAT4(cache_b[i * 4]) = FLOAT4(tile_b[pi][b_col]);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[i] * cache_b[j];
                }
            }
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt7: manual pipeline(douber buffer smem->reg)
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt7(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a_mT[Bk][Bm];
    __shared__ float tile_b[Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // z-order重排
    const int lane_x = lane_id % 16 / 2;
    const int lane_y = lane_id / 16 * 2 + lane_id % 2;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    float acc[m_elems][n_elems] = {0.0f};

    // k-loop
    #pragma unroll
    for (int ki = 0; ki < K; ki += Bk) {

        // 更合适的写法，让展开循环更顺畅
        #pragma unroll
        for (int i = 0; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i + a_thread_y;
                int col = ki + j + a_thread_x;
                // swizzle
                tile_a_mT[j + a_thread_x][swizzle_tile_a_opt4(i + a_thread_y, j + a_thread_x)] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        } 

        #pragma unroll
        for (int i = 0; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bn; j += B_BLOCK_X) {
                int row = ki + i + b_thread_y;
                int col = start_col + j + b_thread_x;
                // 无冲突，不需要swizzle
                tile_b[i + b_thread_y][j + b_thread_x] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
        __syncthreads();

        static_assert(m_elems % 4 == 0);
        static_assert(n_elems % 4 == 0);
        const int m_4elems = m_elems / 4;
        const int n_4elems = n_elems / 4;

        int pbuf_id = 0;
        float cache_a[2][m_elems];
        float cache_b[2][n_elems];

        auto load_ptile = [&](int pbuf_id, int pi) {
            #pragma unroll
            for (int i = 0; i < m_4elems; i++) {
                int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
                FLOAT4(cache_a[pbuf_id][i * 4]) = FLOAT4(tile_a_mT[pi][swizzle_tile_a_opt4(a_row, pi)]);
            } 

            #pragma unroll
            for (int i = 0; i < n_4elems; i++) {
                int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
                FLOAT4(cache_b[pbuf_id][i * 4]) = FLOAT4(tile_b[pi][b_col]);
            } 
        };

        load_ptile(pbuf_id, 0);

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            if (pi + 1 < Bk) {
                load_ptile(pbuf_id ^ 1, pi + 1);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[pbuf_id][i] * cache_b[pbuf_id][j];
                }
            }

            pbuf_id ^= 1;
        }

        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// opt8: manual pipeline(douber buffer hbm -> smem)
template<const int Bm = 128, const int Bn = 128, const int Bk = 8,
         const int BLOCK_SIZE = 256,
         const int A_BLOCK_X = 8, const int A_BLOCK_Y = 32, 
         const int B_BLOCK_X = 32, const int B_BLOCK_Y = 8,
         const int C_BLOCK_X = 16, const int C_BLOCK_Y = 16,
         const int C_WARP_X = 8, const int C_WARP_Y = 4>
__global__ void gemm_blockTile_opt8(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float tile_a_mT[2][Bk][Bm];
    __shared__ float tile_b[2][Bk][Bn];

    int tile_m_id = blockIdx.y;
    int tile_n_id = blockIdx.x;
    int start_row = tile_m_id * Bm;
    int start_col = tile_n_id * Bn;

    const int a_thread_x = threadIdx.x % A_BLOCK_X;
    const int a_thread_y = threadIdx.x / A_BLOCK_X;
    const int b_thread_x = threadIdx.x % B_BLOCK_X;
    const int b_thread_y = threadIdx.x / B_BLOCK_X;
    // 重排C_BLOCK中的warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int C_NUM_WARPS_X = C_BLOCK_X / C_WARP_X;
    const int warp_x = warp_id % C_NUM_WARPS_X;
    const int warp_y = warp_id / C_NUM_WARPS_X;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // z-order重排
    const int lane_x = lane_id % 16 / 2;
    const int lane_y = lane_id / 16 * 2 + lane_id % 2;

    const int c_thread_x = warp_x * C_WARP_X + lane_x;
    const int c_thread_y = warp_y * C_WARP_Y + lane_y;
    

    static_assert(A_BLOCK_X * A_BLOCK_Y == BLOCK_SIZE);
    static_assert(B_BLOCK_X * B_BLOCK_Y == BLOCK_SIZE);
    static_assert(C_BLOCK_X * C_BLOCK_Y == BLOCK_SIZE);
    static_assert(Bm % A_BLOCK_Y == 0);
    static_assert(Bk % A_BLOCK_X == 0);
    static_assert(Bk % B_BLOCK_Y == 0);
    static_assert(Bn % B_BLOCK_X == 0);
    static_assert(Bm % C_BLOCK_Y == 0);
    static_assert(Bn % C_BLOCK_X == 0);

    const int m_elems = Bm / C_BLOCK_Y;
    const int n_elems = Bn / C_BLOCK_X;
    static_assert(m_elems % 4 == 0);
    static_assert(n_elems % 4 == 0);
    const int m_4elems = m_elems / 4;
    const int n_4elems = n_elems / 4;

    float acc[m_elems][n_elems] = {0.0f};
    float cache_a[2][m_elems];
    float cache_b[2][n_elems];

    auto load_ktile = [&](int kbuf_id, int ki) {
        #pragma unroll
        for (int i = 0; i < Bm; i += A_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bk; j += A_BLOCK_X) {
                int row = start_row + i + a_thread_y;
                int col = ki + j + a_thread_x;
                // swizzle
                tile_a_mT[kbuf_id][j + a_thread_x][swizzle_tile_a_opt4(i + a_thread_y, j + a_thread_x)] = 
                    (row < M && col < K)
                    ? a[row * K + col]
                    : 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < Bk; i += B_BLOCK_Y) {
            #pragma unroll
            for (int j = 0; j < Bn; j += B_BLOCK_X) {
                int row = ki + i + b_thread_y;
                int col = start_col + j + b_thread_x;
                // 无冲突，不需要swizzle
                tile_b[kbuf_id][i + b_thread_y][j + b_thread_x] = 
                    (row < K && col < N)
                    ? b[row * N + col]
                    : 0.0f;
            }
        }
    };

    auto load_ptile = [&](int pbuf_id, int pi, int kbuf_id) {
        #pragma unroll
        for (int i = 0; i < m_4elems; i++) {
            int a_row = (i * C_BLOCK_Y + c_thread_y) * 4;
            FLOAT4(cache_a[pbuf_id][i * 4]) = FLOAT4(tile_a_mT[kbuf_id][pi][swizzle_tile_a_opt4(a_row, pi)]);
        }

        #pragma unroll
        for (int i = 0; i < n_4elems; i++) {
            int b_col = (i * C_BLOCK_X + c_thread_x) * 4;
            FLOAT4(cache_b[pbuf_id][i * 4]) = FLOAT4(tile_b[kbuf_id][pi][b_col]);
        }
    };

    int kbuf_id = 0;

    load_ktile(kbuf_id, 0);
    __syncthreads();

    // k-loop
    #pragma unroll
    for (int ki = 0; ki < K; ki += Bk) {
        if (ki + Bk < K) {
            load_ktile(kbuf_id ^ 1, ki + Bk);
        }

        int pbuf_id = 0;

        load_ptile(pbuf_id, 0, kbuf_id);

        // p-loop
        #pragma unroll
        for (int pi = 0; pi < Bk; pi++) {
            if (pi + 1 < Bk) {
                load_ptile(pbuf_id ^ 1, pi + 1, kbuf_id);
            }

            #pragma unroll
            for (int i = 0; i < m_elems; i++) {
                #pragma unroll
                for (int j = 0; j < n_elems; j++) {
                    acc[i][j] += cache_a[pbuf_id][i] * cache_b[pbuf_id][j];
                }
            }

            pbuf_id ^= 1;
        }

        kbuf_id ^= 1;
        __syncthreads();
    }
    // store
    #pragma unroll
    for (int i = 0; i < m_elems; i++) {
        #pragma unroll
        for (int j = 0; j < n_elems; j++) {
            int row = start_row + (i / 4 * C_BLOCK_Y * 4) + c_thread_y * 4 + i % 4;
            int col = start_col + (j / 4 * C_BLOCK_X * 4) + c_thread_x * 4 + j % 4;
            if (row < M && col < N)
                c[row * N + col] = acc[i][j];
        }
    }
}

// ----------------------------------------------------------------------------
// 练习路线 (由易到难, 仿照 sgemv/reduce 的写法注册):
//
// 注册模板 kernel 的写法:
//   BENCH_PIPE(gemm_shared_mem, b128x128x8, 128, 128, 8)
//   -> bench_pipe_gemm_shared_mem_b128x128x8(bench, grid, block)
//      kernel name 展开为 gemm_shared_mem<128,128,8>
// ----------------------------------------------------------------------------
#define WARP_BENCH_PIPE(kernel, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt0, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt1, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt2, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt3, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt4, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt5, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt6, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt7, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \
BENCH_PIPE(kernel##_opt8, Bm##_##Bn##_##Bk, Bm, Bn, Bk) \

BENCH_PIPE(gemm_naive)
// WARP_BENCH_PIPE(gemm_blockTile, 32, 32, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 32, 64, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 32, 96, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 32, 128, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 64, 32, 8)
WARP_BENCH_PIPE(gemm_blockTile, 64, 64, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 64, 96, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 64, 128, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 96, 32, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 96, 64, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 96, 96, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 96, 128, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 128, 32, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 128, 64, 8)
// WARP_BENCH_PIPE(gemm_blockTile, 128, 96, 8)
WARP_BENCH_PIPE(gemm_blockTile, 128, 128, 8)
CUBLAS_SGEMM_PIPE(cublas_tc)
CUBLAS_SGEMM_PIPE_NOTC(cublas_notc)

void test_naive() {
    GemmBenchBase bench = GemmBenchBase(M, N, K, 100, 100, check_correctness);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    bench_pipe_gemm_naive(bench, grid, block);
    // profiling_pipe_gemm_naive(bench, grid, block);
}

#define BT_CALL(suffix, Bm, Bn) \
    { \
        dim3 grid((N + Bn - 1) / Bn, (M + Bm - 1) / Bm); \
        const std::string _bn = #suffix "_" #Bm "_" #Bn "_8"; \
        if (BenchBase::ProfileSelected(_bn)) { \
            profiling_pipe_gemm_blockTile_##suffix##_##Bm##_##Bn##_8(bench, grid, BLOCK_SIZE); \
        } else if (!BenchBase::ProfileMode()) { \
            bench_pipe_gemm_blockTile_##suffix##_##Bm##_##Bn##_8(bench, grid, BLOCK_SIZE); \
        } \
    }

#define TEST_FUNC(suffix, baseline) \
void test_blockTile_##suffix() { \
    if (!BenchBase::ShouldRun(#suffix "_")) return; \
    GemmBenchBase bench = GemmBenchBase(M, N, K, 50, 100, check_correctness); \
    const int BLOCK_SIZE = 256; \
    bench.SetBaselineKernel(#baseline); \
    /* BT_CALL(suffix, 32, 32) */ \
    /* BT_CALL(suffix, 32, 64) */ \
    /* BT_CALL(suffix, 32, 96) */ \
    /* BT_CALL(suffix, 32, 128) */ \
    /* BT_CALL(suffix, 64, 32) */ \
    BT_CALL(suffix, 64, 64) \
    /* BT_CALL(suffix, 64, 96) */ \
    /* BT_CALL(suffix, 64, 128) */ \
    /* BT_CALL(suffix, 96, 32) */ \
    /* BT_CALL(suffix, 96, 64) */ \
    /* BT_CALL(suffix, 96, 96) */ \
    /* BT_CALL(suffix, 96, 128) */ \
    /* BT_CALL(suffix, 128, 32) */ \
    /* BT_CALL(suffix, 128, 64) */ \
    /* BT_CALL(suffix, 128, 96) */ \
     BT_CALL(suffix, 128, 128) \
    printf("============================================================\n"); \
}

TEST_FUNC(opt0, cublas_notc)
TEST_FUNC(opt1, cublas_notc)
TEST_FUNC(opt2, cublas_notc)
TEST_FUNC(opt3, cublas_notc)
TEST_FUNC(opt4, cublas_notc)
TEST_FUNC(opt5, cublas_notc)
TEST_FUNC(opt6, cublas_notc)
TEST_FUNC(opt7, cublas_notc)
TEST_FUNC(opt8, cublas_notc)

void test_cublas() {
    if (!BenchBase::ShouldRun("cublas")) return;
    GemmBenchBase bench = GemmBenchBase(M, N, K, check_correctness);
    if (BenchBase::ProfileSelected("cublas_notc"))
        profiling_pipe_cublas_notc(bench, dim3(), dim3());
    else if (!BenchBase::ProfileMode())
        bench_pipe_cublas_notc(bench, dim3(), dim3());
    if (BenchBase::ProfileSelected("cublas_tc"))
        profiling_pipe_cublas_tc(bench, dim3(), dim3());
    else if (!BenchBase::ProfileMode())
        bench_pipe_cublas_tc(bench, dim3(), dim3());
}

int main() {
    PrintGpuInfo();
    BenchBase::InitProfileEnv();
    // test_naive();
    test_cublas();
    printf("============================================================\n");
    test_blockTile_opt0();
    test_blockTile_opt1();
    test_blockTile_opt2();
    test_blockTile_opt3();
    test_blockTile_opt4();
    test_blockTile_opt5();
    test_blockTile_opt6();
    test_blockTile_opt7();
    test_blockTile_opt8();
    return 0;
}
