# fp8_bench

一个偏实验性质的 FP8 bench。当前内置了从旧实验迁过来的一组实现：

- per-tensor scaling Triton quant，支持 E4M3 / E5M2；
- per-channel scaling Triton quant，BMM 中 A 按 M 维、B 按 N 维量化；
- per-block scaling Triton quant/BMM，默认 quant block 为 `128×128×128`；
- FP8 BMM，右矩阵逻辑 shape 固定为 `[B,K,N]`，支持 N-major、
  预先 K-major 和 N-major 显式重排三条路径；
- 旧 benchmark 的 quant/BMM shapes；
- FSDP2 BF16 baseline 和一个用于验证训练链路的 fake-FP8 case。

所有命令都在本目录执行：

```bash
cd /path/to/practice_cuda/fp8_training
```

## 先检查远端环境

```bash
python3 -m fp8_bench.doctor
```

依赖由远端环境提供，只要求能正常 import `torch` 和 `triton`。FP8 BMM
需要支持 FP8 Tensor Core 的 GPU。NCU profiling 还需要 `ncu` 在 `PATH` 中。

## Quant

先跑小 shape：

```bash
python3 -m fp8_bench.bench_quant \
  --suite smoke \
  --impl triton_per_tensor \
  --mode both
```

跑迁移过来的全部 shape，只测性能：

```bash
python3 -m fp8_bench.bench_quant \
  --suite legacy \
  --impl triton_per_tensor \
  --mode perf \
  --warmup 20 --iters 100 --repeats 5
```

只跑一个 case：

```bash
python3 -m fp8_bench.bench_quant \
  --suite legacy \
  --case q_b32_m2048_k960 \
  --impl triton_per_tensor
```

Per-block quant 默认使用 `128×128`，也可以覆盖 block shape：

```bash
python3 -m fp8_bench.bench_quant \
  --suite smoke \
  --impl triton_per_block \
  --block-m 128 \
  --block-n 128 \
  --mode both
```

## BMM

小 shape，同时测纯 BMM、quant+BMM 和精度：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_tensor_n \
  --impl triton_per_tensor_k \
  --impl triton_per_tensor_n_transpose \
  --mode both
```

全量旧 shape，只测性能：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --impl triton_per_tensor_n \
  --impl triton_per_tensor_k \
  --impl triton_per_tensor_n_transpose \
  --mode perf \
  --warmup 20 --iters 100 --repeats 5
```

指定 shape、输出类型和 bias：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --case b16_m512_n960_k1280 \
  --impl triton_per_tensor_n \
  --input-dtype bf16 \
  --out-dtype bf16 \
  --fp8-dtype e4m3 \
  --bias
```

三个实现共用同一个 `triton_per_tensor_bmm` 和 Triton kernel：

```text
triton_per_tensor_n             N-major B，直接进入 Triton BMM
triton_per_tensor_k             BMM 计时前预先准备成 K-major
triton_per_tensor_n_transpose   N-major B，在 BMM wrapper 内显式重排成 K-major
```

三者的 B 逻辑 shape 始终是 `[B,K,N]`，N-major/K-major 只由 stride 区分。
`triton_per_tensor_n_transpose` 的 `bmm-only` 包含显式 N→K 重排，另外两种
`bmm-only` 不包含 quant 或 layout 准备；per-tensor dequant scale 会在
计时前合并。`pipeline` 则包含 A/B quant、必要的 layout 转换、scale 合并
和 BMM。

精度输出中，`kernel_rel_l2` 对比相同 FP8 输入的 FP32 累加 reference，
`pipeline_rel_l2` 对比原始输入的 FP32 BMM。

Per-channel BMM 使用同样的三种路径：

```text
triton_per_channel_n
triton_per_channel_k
triton_per_channel_n_transpose
```

其中 A 的 dequant scale shape 为 `[B,M]`，B 为 `[B,N]`，两者在 BMM
kernel 中广播相乘。快速验证三条路径：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_channel_n \
  --impl triton_per_channel_k \
  --impl triton_per_channel_n_transpose \
  --mode both \
  --warmup 5 --iters 20 --repeats 3
```

Per-block BMM 同样注册了三种 B layout：

```text
triton_per_block_n
triton_per_block_k
triton_per_block_n_transpose
```

A scale 的逻辑 shape 为
`[B,ceil(M/QBM),ceil(K/QBK)]`，B scale 为
`[B,ceil(K/QBK),ceil(N/QBN)]`。默认 `QBM=QBK=QBN=128`：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_block_n \
  --impl triton_per_block_k \
  --impl triton_per_block_n_transpose \
  --mode both \
  --warmup 5 --iters 20 --repeats 3
```

覆盖 quant block：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --case b32_m2048_n1600_k1600 \
  --impl triton_per_block_n \
  --quant-block-m 128 \
  --quant-block-k 256 \
  --quant-block-n 128 \
  --mode perf
```

当前 quant kernel 要求 block size 为 2 的幂；当前 BMM config 还要求
`QBK >= 128` 且能被 128 整除。M/N block 可以大于实际 M/N，尾块由 mask
处理。`bmm-only`、`pipeline`、TFLOPS 和三种 layout 的计时口径与
per-channel 相同。

BMM 使用 `2 * B * M * N * K` 计算 FLOPs，终端和 JSONL 都会输出：

```text
bmm_tflops       # 只计算预量化后的 BMM kernel
pipeline_tflops  # quant A + quant B + layout + BMM 的等效吞吐
```

Quant 没有适合的 TFLOPS 定义，因此单独使用读写有效带宽 `bandwidth_gbps`。

Quant 和 BMM 精度结果统一包含 `mean_abs`、`max_abs`、`mse`、`rmse`、
`rel_l2`、`cosine` 以及 NaN/Inf 计数。BMM 会分别记录 kernel reference
和完整 FP8 pipeline reference 两组指标。

## NCU

profile target 默认 warmup 5 次，然后再 launch 一次。NCU 用
`--launch-skip 5` 跳过 warmup。

Quant：

```bash
mkdir -p reports
ncu \
  --set full \
  --kernel-name 'regex:.*fp8_quant_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/quant_q_b32_m2048_k960 \
  python3 -m fp8_bench.profile_one \
    --op quant \
    --case q_b32_m2048_k960 \
    --impl triton_per_tensor \
    --warmup 5
```

Per-block quant：

```bash
ncu \
  --set basic \
  --kernel-name 'regex:.*fp8_per_block_quant_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/per_block_quant \
  python3 -m fp8_bench.profile_one \
    --op quant \
    --case q_smoke_3d \
    --impl triton_per_block \
    --block-m 128 \
    --block-n 128 \
    --warmup 5
```

BMM：

```bash
mkdir -p reports
ncu \
  --set full \
  --kernel-name 'regex:.*batch_fp8_per_tensor_bmm_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/bmm_b16_m512_n960_k1280 \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case b16_m512_n960_k1280 \
    --impl triton_per_tensor_n \
    --warmup 5
```

把 `--impl` 换成 `triton_per_tensor_k` 或
`triton_per_tensor_n_transpose` 即可 profile 另外两条路径。这里的
kernel-name 过滤器只采集最终 BMM kernel；显式重排的端到端成本以
`bench_bmm` 的 `bmm-only` 时间为准。

Per-channel BMM 的 NCU 命令：

```bash
ncu \
  --set basic \
  --kernel-name 'regex:.*batch_fp8_per_channel_bmm_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/per_channel_n \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case bmm_smoke_aligned \
    --impl triton_per_channel_n \
    --warmup 5
```

Per-block BMM：

```bash
ncu \
  --set basic \
  --kernel-name 'regex:.*batch_fp8_per_block_bmm_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/per_block_n \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case bmm_smoke_aligned \
    --impl triton_per_block_n \
    --quant-block-m 128 \
    --quant-block-k 128 \
    --quant-block-n 128 \
    --warmup 5
```

如果只想快速看报告，把 `--set full` 改为 `--set basic`。

## FSDP2

先跑 BF16 baseline：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fp8_bench.bench_fsdp2 \
  --impl bf16 \
  --layers 2 --hidden 1024 --ffn 2048 \
  --batch 2 --seq 256
```

再跑 fake-FP8：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fp8_bench.bench_fsdp2 \
  --impl fake_fp8 \
  --fp8-dtype e4m3 \
  --layers 2 --hidden 1024 --ffn 2048 \
  --batch 2 --seq 256
```

这里的 `fake_fp8` 用真实 FP8 cast 模拟量化误差，用 straight-through
estimator 做 backward，但 matmul 仍是 BF16。它用于先验证 FSDP2
端到端框架，不应该被当成 FP8 BMM 性能数据。以后有支持 backward 的实现时，
替换 `BenchLinear` 即可。

FSDP2 额外输出 `estimated_gemm_tflops`。它按两层 Linear 的 forward、
dgrad 和 wgrad 估算 GEMM FLOPs，不包含通信、LayerNorm、激活和 optimizer，
因此是端到端 step 下的模型 GEMM 等效吞吐，不是单 kernel TFLOPS。

## 输出

默认结果追加写到：

```text
results/quant.jsonl
results/bmm.jsonl
results/fsdp2.jsonl
```

可以用 `--results` 指定其他文件。每条记录带 shape、实现名、dtype、GPU、
CUDA/PyTorch 版本和 git commit。

新增实现时，在 `fp8_bench/impls/` 中实现函数，然后调用
`register_quant` 或 `register_bmm` 注册即可。
