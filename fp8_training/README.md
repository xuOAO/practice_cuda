# fp8_bench

一个偏实验性质的 FP8 bench。当前内置了从旧实验迁过来的一组实现：

- per-tensor scaling Triton quant，支持 E4M3 / E5M2；
- FP8 BMM，支持右矩阵 `[B,K,N]` 连续和 `[B,N,K]` 连续两种布局；
- 旧 benchmark 的 quant/BMM shapes；
- FSDP2 BF16 baseline 和一个用于验证训练链路的 fake-FP8 case。

所有命令都在本目录执行：

```bash
cd /path/to/practice_cuda/fp8_training
```

## 先检查远端环境

```bash
python -m fp8_bench.doctor
```

依赖由远端环境提供，只要求能正常 import `torch` 和 `triton`。FP8 BMM
需要支持 FP8 Tensor Core 的 GPU。NCU profiling 还需要 `ncu` 在 `PATH` 中。

## Quant

先跑小 shape：

```bash
python -m fp8_bench.bench_quant \
  --suite smoke \
  --impl triton_per_tensor \
  --mode both
```

跑迁移过来的全部 shape，只测性能：

```bash
python -m fp8_bench.bench_quant \
  --suite legacy \
  --impl triton_per_tensor \
  --mode perf \
  --warmup 20 --iters 100 --repeats 5
```

只跑一个 case：

```bash
python -m fp8_bench.bench_quant \
  --suite legacy \
  --case q_b32_m2048_k960 \
  --impl triton_per_tensor
```

## BMM

小 shape，同时测纯 BMM、quant+BMM 和精度：

```bash
python -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_tensor_n \
  --impl triton_per_tensor_k \
  --mode both
```

全量旧 shape，只测性能：

```bash
python -m fp8_bench.bench_bmm \
  --suite legacy \
  --impl triton_per_tensor_n \
  --impl triton_per_tensor_k \
  --mode perf \
  --warmup 20 --iters 100 --repeats 5
```

指定 shape、输出类型和 bias：

```bash
python -m fp8_bench.bench_bmm \
  --suite legacy \
  --case b16_m512_n960_k1280 \
  --impl triton_per_tensor_n \
  --input-dtype bf16 \
  --out-dtype bf16 \
  --fp8-dtype e4m3 \
  --bias
```

`bmm-only` 的计时不包含 quant；`pipeline` 包含 A/B quant、K-order 所需的
layout 转换和 BMM。精度输出中，`kernel_rel_l2` 对比相同 FP8 输入的 FP32
累加 reference，`pipeline_rel_l2` 对比原始输入的 FP32 BMM。

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
  python -m fp8_bench.profile_one \
    --op quant \
    --case q_b32_m2048_k960 \
    --impl triton_per_tensor \
    --warmup 5
```

BMM：

```bash
mkdir -p reports
ncu \
  --set full \
  --kernel-name 'regex:.*batch_fp8_bmm_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o reports/bmm_b16_m512_n960_k1280 \
  python -m fp8_bench.profile_one \
    --op bmm \
    --case b16_m512_n960_k1280 \
    --impl triton_per_tensor_n \
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
