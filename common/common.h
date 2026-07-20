#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>

const int KB = 1024;

uint GetGpuInfo(std::string prop_name, const int dev_id = 0) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev_id) != cudaSuccess) {
        std::fprintf(stderr, "[device %d] failed to query properties\n", dev_id);
        assert(false);
    }

    if (prop_name == std::string("sm_count")) {
        return prop.multiProcessorCount;
    }
    else if (prop_name == std::string("max_threads_per_sm")) {
        return prop.maxThreadsPerMultiProcessor;
    }
    else {
        std::fprintf(stderr, "no %s prop\n", prop_name.c_str());
        assert(false);
    }
    return 0xffffffff;
}

// Per-SM architecture constants. NOT exposed by cudaDeviceProp/cudaDeviceGetAttribute.
// FLOPs/clock 值为架构级常数 (与 SKU 的 SM 数 / 频率无关), 乘 multiProcessorCount*clock 即得峰值.
// 来源: CUDA C++ Programming Guide (mma 吞吐表) + NVIDIA 白皮书.
// 置信度: sm_80/sm_90 权威; sm_70/75/86/89 为 best-effort, 用前请按 CC 复核编程指南.
struct ArchPerf {
    int fp32_cores_per_sm;     // FP32 CUDA core 数/SM (FMA, ×2 得 FLOPs)
    int mufu_per_clock_per_sm; // SFU/MUFU 超越函数 ops/clock/SM (__expf/__sinf/__logf/rcp/rsqrt)
    int tc_tf32_dense;         // FLOPs/clock/SM (含 ×2 mul-add), 0 = 该精度 N/A
    int tc_fp16_dense;
    int tc_fp16_sparse;        // 结构化稀疏
    int tc_int8;               // TOPS/clock/SM
    int tc_fp8;                // 0 = N/A (仅 sm_89/sm_90 有)
    int l2_bw_gbs;             // L2->SM 读带宽 GB/s, 白皮书规格值 (runtime 不可查, 约值)
    bool valid;
};

static ArchPerf lookup_arch(int major, int minor) {
    switch (major * 10 + minor) {
        //                        fp32 mufu tf32  fp16 fp16s int8 fp8  l2    valid
        case 80: return { 64, 16, 1024, 2048, 4096, 4096,    0,  5000, true}; // A100  (权威)
        case 90: return {128, 32, 2048, 4096, 8192, 8192,16384, 12000, true}; // H100  (权威)
        case 70: return { 64, 16,    0, 1024,    0,    0,    0,  2000, true}; // V100  (best-effort)
        case 75: return { 64, 16,    0,  512,    0, 1024,    0,  2000, true}; // Turing(best-effort)
        case 86: return {128, 32,  512, 1024, 2048, 2048,    0,  6000, true}; // GA10x (best-effort)
        case 89: return {128, 32,  512, 1024, 2048, 2048, 4096, 12000, true}; // Ada   (best-effort)
        default: return {  0,  0,    0,    0,    0,    0,    0,     0, false};
    }
}

void PrintGpuInfo() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    std::printf("==== GPU Info (for CUDA profiling) ====\n");
    std::printf("device_count: %d\n", dev_count);

    int d = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) {
        std::printf("[device %d] failed to query properties\n", d);
        return;
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaSetDevice(d);
    cudaMemGetInfo(&free_bytes, &total_bytes);

    // clockRate / memoryClockRate were removed from cudaDeviceProp in
    // recent CUDA (13+); query them through the device-attribute API.
    int clock_rate_khz = 0, mem_clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz,
                            cudaDevAttrClockRate, d);
    cudaDeviceGetAttribute(&mem_clock_rate_khz,
                            cudaDevAttrMemoryClockRate, d);

    std::printf("\n---- device %d ----\n", d);
    std::printf("name:                 %s\n", prop.name);
    std::printf("compute_capability:   %d.%d\n", prop.major, prop.minor);
    // std::printf("uuid:                 ");
    // for (int i = 0; i < 16; ++i)
    //     std::printf("%02x%s", prop.uuid.bytes[i], (i == 15 ? "" : ":"));
    // std::printf("\n");
    // std::printf("pci_bus_id:           %04x:%02x:%02x.%x\n",
    //             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
    std::printf("multiprocessor_count: %d\n", prop.multiProcessorCount);
    std::printf("max_threads_per_sm:   %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("max_block_per_block:  %d\n", prop.maxBlocksPerMultiProcessor);
    std::printf("max_threads_per_block:%d\n", prop.maxThreadsPerBlock);
    std::printf("warp_size:            %d\n", prop.warpSize);
    std::printf("shared_mem_per_block: %zu KB\n", prop.sharedMemPerBlock / KB);
    std::printf("shared_mem_per_sm:    %zu KB\n", prop.sharedMemPerMultiprocessor / KB);
    std::printf("regs_per_block:       %d\n", prop.regsPerBlock);
    std::printf("regs_per_sm:          %d\n", prop.regsPerMultiprocessor);
    std::printf("l2_cache_size:        %d KB\n", prop.l2CacheSize / KB);
    // std::printf("mem_clock_rate:       %d kHz (%.2f GHz)\n",
    //             mem_clock_rate_khz, mem_clock_rate_khz / 1e6);
    std::printf("mem_bus_width:        %d bits\n", prop.memoryBusWidth);
    // std::printf("total_global_mem:     %zu B (%.2f GiB)\n",
    //             prop.totalGlobalMem, prop.totalGlobalMem / 1073741824.0);
    // std::printf("free_global_mem:      %zu B (%.2f GiB)\n",
    //             free_bytes, free_bytes / 1073741824.0);
    // std::printf("clock_rate:           %d kHz (%.2f GHz)\n",
    //             clock_rate_khz, clock_rate_khz / 1e6);
    // peak DRAM bandwidth: 2 transfers/clock (DDR) * mem_clock(Hz) * bus_bytes
    std::printf("peak_bandwidth:       %.2f GB/s\n",
                2.0 * mem_clock_rate_khz * 1000.0 *
                    (prop.memoryBusWidth / 8.0) / 1e9);
    // peak FLOPS / 吞吐: 按 compute capability 查 per-SM-per-clock 架构常数表,
    // 再乘 multiProcessorCount * clock. (这些 per-SM 吞吐 cudaDeviceProp 不暴露.)
    // 单位: clock_rate_khz 为 kHz, SMs*per_clock*clock_khz/1e6 即 Gops/s (FP32 再 ×2).
    ArchPerf a = lookup_arch(prop.major, prop.minor);
    if (a.valid) {
        // FP32 CUDA core: SMs * fp32_cores/SM * clock * 2(FMA)
        std::printf("peak_flops_fp32:      %.2f GFLOPS\n",
                    (double)prop.multiProcessorCount * a.fp32_cores_per_sm
                        * clock_rate_khz * 2.0 / 1e6);
        // SFU/MUFU: 超越函数 ops/s (exp/sin/log/rcp/rsqrt 的 __ 快速 intrinsic 走这里)
        std::printf("peak_sfu_ops:         %.2f Gops/s\n",
                    (double)prop.multiProcessorCount * a.mufu_per_clock_per_sm
                        * clock_rate_khz / 1e6);
        // Tensor Core (各精度, 0 = 该架构不支持)
        auto tc = [&](int per_clk, const char* suffix) {
            if (per_clk == 0) { std::printf("peak_flops_tc_%-11s N/A\n", suffix); return; }
            std::printf("peak_flops_tc_%-11s %.2f GFLOPS\n", suffix,
                        (double)prop.multiProcessorCount * per_clk * clock_rate_khz / 1e6);
        };
        tc(a.tc_tf32_dense,  "tf32");
        tc(a.tc_fp16_dense,  "fp16");
        tc(a.tc_fp16_sparse, "fp16_sparse");
        tc(a.tc_int8,        "int8");
        tc(a.tc_fp8,         "fp8");
        // L2 带宽: runtime 不可查, 取白皮书规格值 (约值)
        std::printf("l2_cache_bandwidth:   %d GB/s (spec, approx)\n", a.l2_bw_gbs);
    } else {
        std::printf("peak_flops_fp32:      N/A (unknown CC %d.%d; extend lookup_arch)\n",
                    prop.major, prop.minor);
    }
    // std::printf("concurrent_kernels:   %d\n", prop.concurrentKernels);
    // std::printf("async_engine_count:   %d\n", prop.asyncEngineCount);
    // std::printf("unified_addressing:   %d\n", prop.unifiedAddressing);

    // int drv_major = 0, drv_minor = 0;
    // int rt_major = 0, rt_minor = 0;
    // cudaDriverGetVersion(&drv_major);
    // cudaRuntimeGetVersion(&rt_major);
    // std::printf("\n---- runtime ----\n");
    // std::printf("driver_version:       %d\n", drv_major);
    // std::printf("runtime_version:      %d\n", rt_major);
    // (void)drv_minor; (void)rt_minor;
    std::printf("=======================================\n");
}