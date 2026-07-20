#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>

class BenchBase {
public:
    BenchBase() {
        cudaEventCreate(&st);
        cudaEventCreate(&ed);
    }
    BenchBase(int warmup_iters, int iters)
    : warmup_iters(warmup_iters), iters(iters) {
        cudaEventCreate(&st);
        cudaEventCreate(&ed);
    }
    virtual ~BenchBase() {
        cudaEventDestroy(st);
        cudaEventDestroy(ed);
    }

    virtual std::string BenchInfo() = 0;
    virtual bool CheckCorrectness(const std::function<void()>& fn) = 0;

    // ---- Structured CSV sink (for downstream plotting) -----------------
    // Call OpenCsv() once at program start; each bench_pipe() call appends
    // one row. Columns:
    //   kernel,size,grid_x,grid_y,grid_z,block_x,block_y,block_z,correct,time_ms
    inline static std::ofstream csv;
    inline static bool csv_open = false;

    static void OpenCsv(const std::string& path) {
        // std::ofstream won't create intermediate dirs; do it ourselves so a
        // path like "figures/reduce_results.csv" works from a clean run.
        if (auto p = std::filesystem::path(path).parent_path(); !p.empty()) {
            std::filesystem::create_directories(p);
        }
        csv.open(path);
        if (csv.is_open()) {
            csv << "kernel,size,"
                << "grid_x,grid_y,grid_z,"
                << "block_x,block_y,block_z,"
                << "correct,time_ms\n";
            csv_open = true;
        }
    }

    static void CloseCsv() {
        if (csv_open) {
            csv.flush();
            csv.close();
            csv_open = false;
        }
    }

    static void WriteCsvRow(const std::string& kernel, BenchBase& bench,
                            dim3 grid, dim3 block, bool correct,
                            double ms, bool has_time) {
        if (!csv_open) return;
        csv << kernel << ','
            << bench.BenchInfo() << ','
            << grid.x << ',' << grid.y << ',' << grid.z << ','
            << block.x << ',' << block.y << ',' << block.z << ','
            << (correct ? 1 : 0) << ',';
        if (has_time) csv << ms;
        csv << '\n';
        csv.flush();
    }

    // ---- Baseline (opt-in) --------------------------------------------
    // 每个 bench 实例可自选一个 baseline kernel 名 (SetBaselineKernel);
    // bench_pipe 每次会把实测 ms 记进 baseline_times (key = "size|kernel"),
    // 打印时若该 bench 设了 baseline 且已测到, 追加 "X.XXx baseline" 加速比.
    // 不调用 SetBaselineKernel 则完全走原逻辑, 对 sgemv/reduce 无影响.
    inline static std::map<std::string, double> baseline_times;
    std::string baseline_kernel;
    void SetBaselineKernel(const std::string& name) { baseline_kernel = name; }

    // ---- Profiling mode (opt-in, env PROFILE_KERNEL) -----------------
    // 设置 PROFILE_KERNEL=<bench_name> 后, 调用方据 ProfileSelected() 决定:
    //   选中 -> profiling_pipe 单次启动 (给 ncu 抓单次 profile);
    //   未选中 -> 跳过;
    //   未设环境变量 -> 正常 bench_pipe 计时.
    // ShouldRun(prefix) 供 test_* 函数早退: profile 模式下只有含选中 kernel 的那个
    // test 函数才构造 bench 并运行, 避免无关 bench 的 CPU 参考答案计算开销.
    inline static std::string profile_only;
    static void InitProfileEnv() {
        if (const char* p = std::getenv("PROFILE_KERNEL")) profile_only = p;
    }
    static bool ProfileMode() { return !profile_only.empty(); }
    static bool ProfileSelected(const std::string& bench_name) {
        return ProfileMode() && profile_only == bench_name;
    }
    static bool ShouldRun(const std::string& prefix) {
        return !ProfileMode() || profile_only.rfind(prefix, 0) == 0;
    }

    bool LaunchCheck(const std::function<void()>& fn) {
        fn();
        cudaError_t err_async = cudaGetLastError();
        cudaError_t err_sync = cudaDeviceSynchronize();
        if (err_async != cudaSuccess)
            printf("Launch error: %s\n", cudaGetErrorString(err_async));
        if (err_sync != cudaSuccess)
            printf("Launch error: %s\n", cudaGetErrorString(err_sync));
        return err_async == cudaSuccess;
    }

    double RunBench(const std::function<void()>& fn) {
        for (int i = 0; i < warmup_iters; ++i) {
            fn();
        }
        cudaDeviceSynchronize();
        
        cudaEventRecord(st);
        for (int i = 0; i < iters; ++i) {
            fn();
        }
        cudaEventRecord(ed);
        cudaEventSynchronize(ed);
        
        float ms;
        cudaEventElapsedTime(&ms, st, ed);
        return ms / iters;
    }

    void DoProfiling(const std::function<void()>& fn) {
        fn();
    }

    static void bench_pipe(BenchBase& bench, const std::function<void()>& fn,
                    const std::string& kernel_name, dim3 grid, dim3 block) {
        bool launch_success = bench.LaunchCheck(fn);
        if (!launch_success) {
            WriteCsvRow(kernel_name, bench, grid, block, false, 0.0, false);
            return;
        }
        bool is_correct = bench.CheckCorrectness(fn);
        if (!is_correct) {
            char cfg[96];
            std::snprintf(cfg, sizeof(cfg), "grid=(%5d,%3d) block=(%3d,%3d)",
                          grid.x, grid.y, block.x, block.y);
            std::printf("%-30s %-30s %-30s FALSE\n", kernel_name.c_str(), bench.BenchInfo().c_str(), cfg);
        }
        double ms = bench.RunBench(fn);
        // 记录该 (size, kernel) 的实测时间, 供后续任一 bench 选作 baseline.
        bench.baseline_times[bench.BenchInfo() + "|" + kernel_name] = ms;

        char cfg[96];
        std::snprintf(cfg, sizeof(cfg), "grid=(%5d,%3d) block=(%3d,%3d)",
                      grid.x, grid.y, block.x, block.y);
        // 若本 bench 选了 baseline 且已测到, 追加加速比 (base/ms)x.
        double base = 0.0;
        if (!bench.baseline_kernel.empty()) {
            auto it = bench.baseline_times.find(bench.BenchInfo() + "|" + bench.baseline_kernel);
            if (it != bench.baseline_times.end()) base = it->second;
        }
        if (base > 0.0) {
            std::printf("%-30s %-30s %-30s %14.6fms  %6.2fx baseline\n",
                        kernel_name.c_str(), bench.BenchInfo().c_str(), cfg, ms, base / ms);
        } else {
            std::printf("%-30s %-30s %-30s %14.6fms\n",
                        kernel_name.c_str(), bench.BenchInfo().c_str(), cfg, ms);
        }
        WriteCsvRow(kernel_name, bench, grid, block, is_correct, ms, true);
    }

    static void profiling_pipe(BenchBase& bench, const std::function<void()>& fn,
                    const std::string& kernel_name, dim3 grid, dim3 block) {
        char cfg[96];
        std::snprintf(cfg, sizeof(cfg), "grid=(%5d,%3d) block=(%3d,%3d)",
                      grid.x, grid.y, block.x, block.y);
        std::printf("[profiling] %-30s %-30s %-30s\n", kernel_name.c_str(), bench.BenchInfo().c_str(), cfg);
        bench.DoProfiling(fn);
    }

    int warmup_iters = 50;
    int iters = 100;
    
    cudaEvent_t st, ed;
};