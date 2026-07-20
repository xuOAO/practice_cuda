#include "bench_framework.h"

// BENCH_PIPE(kernel_name, template_args...)
// bench_pipe_##kernel_name(bench, grid, block)
// profiling_pipe_##kernel_name(bench, grid, block)

#define KERNEL Reduce

class ReduceBenchBase: public BenchBase {
public:
    ReduceBenchBase(int s, int k, bool check_correctness=true)
        : S(s), K(k), check_correctness(check_correctness){
        size_t size_x = S * K * sizeof(float);
        size_t size_y = S * sizeof(float);
        
        h_x = (float*)malloc(size_x);
        
        cudaMalloc(&d_x, size_x);
        cudaMalloc(&d_y, size_y);
        
        // Initialize h_a and h_x with random values
        for (int i = 0; i < S * K; ++i) {
            h_x[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

        cudaEventCreate(&st);
        cudaEventCreate(&ed);

        // prepare ans
        ans_y = (float*)malloc(size_y);
        for (int i = 0; i < S; ++i) {
            ans_y[i] = 0.0f;
            for (int j = 0; j < K; ++j) {
                ans_y[i] += h_x[i * K + j];
            }
        }
    }

    ~ReduceBenchBase() override {
        free(h_x);
        free(ans_y);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    std::string BenchInfo() override {
        return std::string(std::to_string(S) + "x" + std::to_string(K));
    }

    bool CheckCorrectness(const std::function<void()>& fn) override {
        if (check_correctness == false)
            return true;
        float *h_y = (float*)malloc(S * sizeof(float));
        fn();
        cudaMemcpy(h_y, d_y, S * sizeof(float), cudaMemcpyDeviceToHost);
        // abs_err + rel_err: 适配数值量级跨度大的场景
        // tol = atol + rtol * |ans|
        const float atol = 1e-5f;
        const float rtol = 1e-4f;
        for (int i = 0; i < S; ++i) {
            float diff = fabs(h_y[i] - ans_y[i]);
            float tol = atol + rtol * fabs(ans_y[i]);
            if (diff > tol) {
                printf("[CheckFailed] %s idx=%d: expected=%.6f actual=%.6f diff=%.6f tol=%.6f\n",
                       BenchInfo().c_str(), i, ans_y[i], h_y[i], diff, tol);
                free(h_y);
                return false;
            }
        }
        free(h_y);
        return true;
    }

    int S, K;
    bool check_correctness;
    float *h_x, *ans_y;
    float *d_x, *d_y;
};

// BENCH_PIPE(kernel_name, template_args...)
// bench_pipe_##kernel_name(bench, grid, block)
// profiling_pipe_##kernel_name(bench, grid, block)

#define BENCH_PIPE_NO_SHARED_FUNC(kernel_name, name, func) \
void func##_pipe_##name(kernel_name##BenchBase& bench, dim3 grid, dim3 block) { \
    auto kl = [&bench, grid, block]() { \
        int S = bench.S; \
        int K = bench.K; \
        name<<<grid, block>>>(bench.d_x, bench.d_y, S, K); \
    }; \
    BenchBase::func##_pipe(bench, kl, std::string(#name), grid, block); \
}

#define BENCH_PIPE_SHARED_FUNC(kernel_name, name, func, suffix, ...) \
void func##_pipe_##name##_##suffix(kernel_name##BenchBase& bench, dim3 grid, dim3 block) { \
    auto kl = [&bench, grid, block]() { \
        int S = bench.S; \
        int K = bench.K; \
        name<__VA_ARGS__><<<grid, block>>>(bench.d_x, bench.d_y, S, K); \
    }; \
    BenchBase::func##_pipe(bench, kl, std::string(#name "<" #suffix ">"), grid, block); \
}

#define GET_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME

#define BENCH_PIPE_FUNC(...) \
    GET_MACRO(__VA_ARGS__,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_SHARED_FUNC,  \
        BENCH_PIPE_NO_SHARED_FUNC,  \
        0)(__VA_ARGS__)

#define BENCH_PIPE(name, ...) \
BENCH_PIPE_FUNC(KERNEL, name, bench, ##__VA_ARGS__) \
BENCH_PIPE_FUNC(KERNEL, name, profiling, ##__VA_ARGS__)