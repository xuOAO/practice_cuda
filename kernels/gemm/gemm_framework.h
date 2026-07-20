#include "bench_framework.h"
#include <cublas_v2.h>

// BENCH_PIPE(kernel_name, optional(suffix), template_args...)
// bench_pipe_##kernel_name[_suffix](bench, grid, block)
// profiling_pipe_##kernel_name[_suffix](bench, grid, block)

#define KERNEL Gemm

// C[M, N] = A[M, K] @ B[K, N]
//   A: row-major [M, K]
//   B: row-major [K, N]
//   C: row-major [M, N]
class GemmBenchBase: public BenchBase {
public:
    GemmBenchBase(int m, int n, int k, bool check_correctness=true)
        : BenchBase(), M(m), N(n), K(k), check_correctness(check_correctness){
        _BenchBase_init();
    }

    GemmBenchBase(int m, int n, int k,
        int warmup_iters, int iters, bool check_correctness=true)
        : BenchBase(warmup_iters, iters),
          M(m), N(n), K(k), check_correctness(check_correctness){
        _BenchBase_init();
    }

    void _BenchBase_init() {
        // [M, K] @ [K, N] -> [M, N]
        size_t size_a = M * K * sizeof(float);
        size_t size_b = K * N * sizeof(float);
        size_t size_c = M * N * sizeof(float);

        h_a = (float*)malloc(size_a);
        h_b = (float*)malloc(size_b);

        cudaMalloc(&d_a, size_a);
        cudaMalloc(&d_b, size_b);
        cudaMalloc(&d_c, size_c);

        // Initialize h_a and h_b with random values
        for (int i = 0; i < M * K; ++i) {
            h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        for (int i = 0; i < K * N; ++i) {
            h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

        // st/ed 由父类 BenchBase 构造函数创建, 这里不再重复

        ans_c = (float*)malloc(size_c);
        // prepare ans
        if (check_correctness == true) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < K; ++p) {
                        sum += h_a[i * K + p] * h_b[p * N + j];
                    }
                    ans_c[i * N + j] = sum;
                }
            } 
        }
    }

    ~GemmBenchBase() override {
        free(h_a);
        free(h_b);
        free(ans_c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    std::string BenchInfo() override {
        return std::string(std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K));
    }

    bool CheckCorrectness(const std::function<void()>& fn) override {
        if (check_correctness == false)
            return true;
        float *h_c = (float*)malloc(M * N * sizeof(float));
        fn();
        cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        // abs_err + rel_err: 适配数值量级跨度大的场景
        // tol = atol + rtol * |ans|
        // GEMM 沿 K 累加, 误差随 K 增大; 这里比 sgemv/reduce 放宽一档
        const float atol = 1e-3f;
        const float rtol = 1e-3f;
        for (int i = 0; i < M * N; ++i) {
            float diff = fabs(h_c[i] - ans_c[i]);
            float tol = atol + rtol * fabs(ans_c[i]);
            if (diff > tol) {
                printf("[CheckFailed] %s idx=%d: expected=%.6f actual=%.6f diff=%.6f tol=%.6f\n",
                       BenchInfo().c_str(), i, ans_c[i], h_c[i], diff, tol);
                free(h_c);
                return false;
            }
        }
        free(h_c);
        return true;
    }

    int M, N, K;
    bool check_correctness;
    float *h_a, *h_b, *ans_c;
    float *d_a, *d_b, *d_c;
};

// BENCH_PIPE(kernel_name, optional(suffix), template_args...)
// bench_pipe_##kernel_name[_suffix](bench, grid, block)
// profiling_pipe_##kernel_name[_suffix](bench, grid, block)

#define BENCH_PIPE_NO_SHARED_FUNC(kernel_name, name, func) \
void func##_pipe_##name(kernel_name##BenchBase& bench, dim3 grid, dim3 block) { \
    auto kl = [&bench, grid, block]() { \
        int M = bench.M; \
        int N = bench.N; \
        int K = bench.K; \
        name<<<grid, block>>>(bench.d_a, bench.d_b, bench.d_c, M, N, K); \
    }; \
    BenchBase::func##_pipe(bench, kl, std::string(#name), grid, block); \
}

#define BENCH_PIPE_SHARED_FUNC(kernel_name, name, func, suffix, ...) \
void func##_pipe_##name##_##suffix(kernel_name##BenchBase& bench, dim3 grid, dim3 block) { \
    auto kl = [&bench, grid, block]() { \
        int M = bench.M; \
        int N = bench.N; \
        int K = bench.K; \
        name<__VA_ARGS__><<<grid, block>>>(bench.d_a, bench.d_b, bench.d_c, M, N, K); \
    }; \
    BenchBase::func##_pipe(bench, kl, std::string(#name "<" #suffix ">"), grid, block); \
}

// 16 个命名槽, NAME 落在第 17 位: 可容纳 suffix + 12 个模板参数 (实参总数 16).
// 命名槽数量必须 >= 3 + 变参数, 否则第 7 位会落到某个模板字面量上而崩.
#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
                  _11, _12, _13, _14, _15, _16, NAME, ...) NAME

#define BENCH_PIPE_FUNC(...) \
    GET_MACRO(__VA_ARGS__,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_NO_SHARED_FUNC,  \
        0)(__VA_ARGS__)

#define BENCH_PIPE(name, ...) \
BENCH_PIPE_FUNC(KERNEL, name, bench, ##__VA_ARGS__) \
BENCH_PIPE_FUNC(KERNEL, name, profiling, ##__VA_ARGS__)

// cuBLAS sgemm baseline. 不走 BENCH_PIPE (它假设 kernel<<<>>> 启动), 而是把 cublasSgemm
// 包进 lambda 直接交给 BenchBase::bench_pipe / profiling_pipe. handle 创建在计时区外.
//
// 行主序 C(M,N)=A(M,K)@B(K,N) 用列主序转置技巧: cuBLAS 列主序下
//   C_rm 读作列主序 = C^T,  d_b 读作列主序 = B^T,  d_a 读作列主序 = A^T
//   想要 C^T = B^T @ A^T  =>  传 d_b 作 A(lda=N), d_a 作 B(ldb=K), m=N, n=M, k=K, 输出 d_c(ldc=N).
//
// 注意: name 不要用 "cublasSgemm" —— 它在 cublas_v2.h 里是个宏 (#define cublasSgemm cublasSgemm_v2),
// 当宏参数会被预扫描展开, 导致 bench_pipe_##name 粘出 bench_pipe_cublasSgemm_v2 与调用点对不上.
// 这里直接硬编码 GemmBenchBase (本宏为 gemm 专属, 引用了 bench.M/N/K/d_a/d_b/d_c).
//
// 两个入口, 用 cublasSetMathMode 显式指定是否走 Tensor Core (不依赖 cuBLAS 默认行为):
//   CUBLAS_SGEMM_PIPE(name)      -> CUBLAS_TF32_TENSOR_OP_MATH (TF32 TC, 性能上限)
//   CUBLAS_SGEMM_PIPE_NOTC(name) -> CUBLAS_PEDANTIC_MATH       (严格 FP32, 走 CUDA Core, 公平对照)
#define CUBLAS_SGEMM_PIPE_IMPL(name, MATH) \
void bench_pipe_##name(GemmBenchBase& bench, dim3, dim3) { \
    cublasHandle_t handle; \
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { \
        printf("[cublas] handle create failed\n"); return; \
    } \
    cublasSetMathMode(handle, MATH); \
    const float alpha = 1.0f, beta = 0.0f; \
    auto kl = [&]() { \
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
                    bench.N, bench.M, bench.K, &alpha, \
                    bench.d_b, bench.N, bench.d_a, bench.K, \
                    &beta, bench.d_c, bench.N); \
    }; \
    BenchBase::bench_pipe(bench, kl, std::string(#name), dim3(), dim3()); \
    cublasDestroy(handle); \
} \
void profiling_pipe_##name(GemmBenchBase& bench, dim3, dim3) { \
    cublasHandle_t handle; \
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { \
        printf("[cublas] handle create failed\n"); return; \
    } \
    cublasSetMathMode(handle, MATH); \
    const float alpha = 1.0f, beta = 0.0f; \
    auto kl = [&]() { \
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
                    bench.N, bench.M, bench.K, &alpha, \
                    bench.d_b, bench.N, bench.d_a, bench.K, \
                    &beta, bench.d_c, bench.N); \
    }; \
    BenchBase::profiling_pipe(bench, kl, std::string(#name), dim3(), dim3()); \
    cublasDestroy(handle); \
}
#define CUBLAS_SGEMM_PIPE(name)      CUBLAS_SGEMM_PIPE_IMPL(name, CUBLAS_TF32_TENSOR_OP_MATH)
#define CUBLAS_SGEMM_PIPE_NOTC(name) CUBLAS_SGEMM_PIPE_IMPL(name, CUBLAS_PEDANTIC_MATH)
