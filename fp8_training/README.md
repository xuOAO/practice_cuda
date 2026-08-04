# fp8_bench

一个偏实验性质的 FP8 bench。当前内置了从旧实验迁过来的一组实现：

- per-tensor scaling Triton quant，支持 E4M3 / E5M2；
- per-channel scaling Triton quant，BMM 中 A 按 M 维、B 按 N 维量化；
- per-block scaling Triton quant/BMM，区分沿 K 的 1D `1×128` block 和
  `128×128` 的 2D block；
- FP8 BMM，A/B 的逻辑 shape 分别固定为 `[B,M,K]` / `[B,K,N]`，
  支持常见的直接布局和显式重排共五条路径；
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

依赖由远端环境提供，要求能正常 import `torch`、`triton` 和 `pandas`；
缺少 pandas 时可以运行 `python3 -m pip install -r requirements.txt`。FP8 BMM
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

Per-block quant 分为默认 `1×128` 的 1D block 和默认 `128×128` 的 2D block，
也可以覆盖 block shape：

```bash
python3 -m fp8_bench.bench_quant \
  --suite smoke \
  --impl triton_per_block_1d \
  --impl triton_per_block_2d \
  --mode both
```

需要实验其他 block shape 时仍可用 `--block-m` 和 `--block-n` 覆盖默认值。

## BMM

小 shape，同时测纯 BMM、quant+BMM 和精度：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_tensor_a_k_b_n \
  --impl triton_per_tensor_a_k_b_k \
  --impl triton_per_tensor_a_k_b_n_transpose \
  --mode both
```

全量旧 shape，只测性能：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --impl triton_per_tensor_a_k_b_n \
  --impl triton_per_tensor_a_k_b_k \
  --impl triton_per_tensor_a_k_b_n_transpose \
  --mode perf \
  --warmup 20 --iters 100 --repeats 5
```

全量 shape、全部实现，只计时预量化后的 BMM，并在同一次运行中比较
block-pointer 和 TMA kernel：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --mode perf \
  --perf-scope bmm \
  --backend both \
  --warmup 20 --iters 100 --repeats 5
```

`--backend` 可选 `block-ptr`、`tma` 或 `both`，默认是 `block-ptr`；
原有 `--use-tma` 等价于 `--backend tma`。TMA 要求相关 FP8
维度满足 16-byte 对齐，内置 legacy shapes 均满足。

同一批 shape 同时测纯 FP16 `torch.bmm` baseline、FP8 BMM-only 和
quant+FP8 BMM pipeline：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --impl triton_per_channel_a_k_b_n \
  --impl triton_per_block_1d_a_k_b_n \
  --impl triton_per_block_2d_a_k_b_n \
  --mode perf \
  --perf-scope both \
  --backend both \
  --fp16-baseline \
  --results results/bmm_with_fp16_baseline.jsonl
```

FP16 baseline 每个 case 测三种布局：`fp16_a_k_b_k`、
`fp16_a_k_b_n` 和 `fp16_a_m_b_n`。输入转换、布局准备和输出分配
不计时；JSONL 中使用对应的 `impl` 名称，并记为 `backend=torch`。
使用 `--bias` 时对应调用 FP16 `torch.baddbmm`。

指定 shape、输出类型和 bias：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --case b16_m512_n960_k1280 \
  --impl triton_per_tensor_a_k_b_n \
  --input-dtype bf16 \
  --out-dtype bf16 \
  --fp8-dtype e4m3 \
  --bias
```

布局名约定如下：A 的逻辑 shape 为 `[B,M,K]`，`a_k` 表示 K 维连续，
`a_m` 表示 M 维连续；B 的逻辑 shape 为 `[B,K,N]`，`b_n` 表示 N 维
连续，`b_k` 表示 K 维连续。每种 scaling 都注册下面五条路径；
per-tensor 的具体名字是：

```text
triton_per_tensor_a_k_b_k                                  a_k_b_k
triton_per_tensor_a_k_b_n                                  a_k_b_n
triton_per_tensor_a_k_b_n_transpose                        a_k_b_n_transpose
triton_per_tensor_a_m_b_n                            a_m_b_n
triton_per_tensor_a_m_transpose_b_n_transpose        a_m_transpose_b_n_transpose
```

所有路径的逻辑 shape 不变，layout 只由 stride 区分。
`triton_per_tensor_a_k_b_n_transpose` 的 `bmm-only` 包含显式 N→K 重排；直接布局
路径的 `bmm-only` 不包含 quant 或 layout 准备，per-tensor dequant scale
会在计时前合并。`a_m_transpose_b_n_transpose` 的 `bmm-only` 包含 A 和 B
两次显式重排。benchmark 会在计时前把 BF16 输入 materialize 成路径要求的
stride，quant 保持输入布局；因此直接布局的 `pipeline` 只包含 A/B quant、
scale 合并和 BMM。名字中显式带 `transpose` 的路径仍会计入其内部重排。
当前没有注册只重排 a_m 输入中一个 operand 的组合。

对应标准 linear `Y = X @ W.T`（X、W 和 dY 均按 PyTorch 默认连续存储）：

```text
forward: a_k_b_k
dX:      a_k_b_n
dW:      a_m_b_n
```

精度输出中，`kernel_rel_l2` 对比相同 FP8 输入的 FP32 累加 reference，
`pipeline_rel_l2` 对比原始输入的 FP32 BMM。

Per-channel BMM 使用同样的五种路径：

```text
triton_per_channel_a_k_b_k
triton_per_channel_a_k_b_n
triton_per_channel_a_k_b_n_transpose
triton_per_channel_a_m_b_n
triton_per_channel_a_m_transpose_b_n_transpose
```

其中 A 的 dequant scale shape 为 `[B,M]`，B 为 `[B,N]`，两者在 BMM
kernel 中广播相乘。快速验证五条路径：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_channel_a_k_b_n \
  --impl triton_per_channel_a_k_b_k \
  --impl triton_per_channel_a_k_b_n_transpose \
  --impl triton_per_channel_a_m_b_n \
  --impl triton_per_channel_a_m_transpose_b_n_transpose \
  --mode both \
  --warmup 5 --iters 20 --repeats 3
```

Per-block BMM 的 1D/2D scaling 各自注册了五种路径：

```text
triton_per_block_{1d,2d}_a_k_b_k
triton_per_block_{1d,2d}_a_k_b_n
triton_per_block_{1d,2d}_a_k_b_n_transpose
triton_per_block_{1d,2d}_a_m_b_n
triton_per_block_{1d,2d}_a_m_transpose_b_n_transpose
```

A scale 的逻辑 shape 为
`[B,ceil(M/QBM),ceil(K/QBK)]`，B scale 为
`[B,ceil(K/QBK),ceil(N/QBN)]`。1D 沿 K 分组，默认
`(QBM,QBK,QBN)=(1,128,1)`；2D 默认 `(128,128,128)`：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite smoke \
  --impl triton_per_block_1d_a_k_b_n \
  --impl triton_per_block_2d_a_k_b_n \
  --mode both \
  --warmup 5 --iters 20 --repeats 3
```

覆盖 quant block：

```bash
python3 -m fp8_bench.bench_bmm \
  --suite legacy \
  --case b32_m2048_n1600_k1600 \
  --impl triton_per_block_2d_a_k_b_n \
  --quant-block-m 128 \
  --quant-block-k 256 \
  --quant-block-n 128 \
  --mode perf
```

当前 quant kernel 要求 block size 为 2 的幂；当前 BMM config 还要求
`QBK >= 128` 且能被 128 整除。M/N block 可以大于实际 M/N，尾块由 mask
处理。`bmm-only`、`pipeline`、TFLOPS 和五种 layout 的计时口径与
per-channel 相同。

BMM 使用 `2 * B * M * N * K` 计算 FLOPs，终端和 JSONL 都会输出：

```text
bmm_tflops       # 只计算预量化后的 BMM kernel
pipeline_tflops  # quant A + quant B + BMM 的等效吞吐；显式 transpose 路径含重排
```

Quant 没有适合的 TFLOPS 定义，因此单独使用读写有效带宽 `bandwidth_gbps`。

Quant 和 BMM 精度结果统一包含 `mean_abs`、`max_abs`、`mse`、`rmse`、
`rel_l2`、`cosine` 以及 NaN/Inf 计数。BMM 会分别记录 kernel reference
和完整 FP8 pipeline reference 两组指标。

## Autotune config 搜索

`fp8_bench.tuning` 是离线搜索工具，不会修改 kernel 中现有的
`_CONFIGS`。它按下面的顺序工作：

1. 根据所选 suite 的最大 M/N/K 和命令行上限生成 2 的幂搜索空间；
2. 编译并加载每个 cubin，过滤编译失败、spill/local memory、资源超限、
   occupancy 为零以及没有生成 WGMMA/HGMMA 的配置；
3. 短测所有通过编译筛选的配置，再对耗时前几名做正式复测；
4. 写出完整 JSONL 和每个 shape 的最佳配置 JSON，并在终端格式化打印
   每个 impl 的 per-case 最佳配置和可直接贴回 `_CONFIGS` 的
   `triton.Config(...)` 列表。

例如搜索 per-tensor N-major：

```bash
python3 -m fp8_bench.tuning.search_bmm \
  --suite legacy \
  --impl triton_per_tensor_a_k_b_n \
  --block-m-cap 256 \
  --block-n-cap 256 \
  --block-k-cap 256
```

`--impl` 可重复传值，一次跑出多种实现的最佳配置；每个实现分别写入
`results/tuning/<impl>_<suite>.jsonl` 及配套的 `.best.json`，避免把不同
layout 混入同一结果文件。TMA 变体以
`*_tma` 结尾（如 `triton_per_tensor_a_k_b_n_tma`），它要求 fp8 的 K（以及
N-major B 的 N）为 16 的倍数，否则该 shape 会被跳过并记为
`case_failed`。一条命令同时搜 block_ptr 与 TMA 两种 per-tensor kernel：

```bash
python3 -m fp8_bench.tuning.search_bmm \
  --suite legacy \
  --impl triton_per_tensor_a_k_b_n \
  --impl triton_per_tensor_a_k_b_n_tma
```

直接搜索 A M-major、B N-major 的 linear wgrad 布局：

```bash
python3 -m fp8_bench.tuning.search_bmm \
  --suite legacy \
  --impl triton_per_tensor_a_m_b_n \
  --impl triton_per_tensor_a_m_b_n_tma
```

kernel 中现有的 base/TMA config 池已经分别合并了 legacy suite 上
`a_m_b_n` 的 per-tensor、per-channel 和 per-block 去重赢家；autotune key
包含 `A_K_MAJOR` 与 `B_N_MAJOR`，不同布局会独立选择配置。

每个 `.best.json` 的 `best_by_case` 保存该实现的逐 shape winner，`by_impl`
给出 `unique_configs` 和可直接贴回 `_CONFIGS` 的 `triton_configs`。搜索
per-channel K-major：

```bash
python3 -m fp8_bench.tuning.search_bmm \
  --suite legacy \
  --impl triton_per_channel_a_k_b_k
```

Per-block 会额外按 `QBK >= BLOCK_K` 且 `QBK % BLOCK_K == 0` 过滤：

```bash
python3 -m fp8_bench.tuning.search_bmm \
  --suite legacy \
  --impl triton_per_block_2d_a_k_b_n \
  --quant-block-m 128 \
  --quant-block-k 256 \
  --quant-block-n 128
```

默认搜索 `BM>=64、BN>=8、BK>=32`，并要求 WGMMA/HGMMA 和零 spill。
调试非 Hopper 环境或检查过滤逻辑时，可临时使用
`--no-require-wgmma`、`--no-reject-local-memory` 或
`--no-static-resource-filter`。短测所有通过编译筛选的配置，再按耗时取
前 `--quick-top-k`（默认 8）个做正式复测。

默认结果写到：

```text
results/tuning/<impl>_<suite>.jsonl
results/tuning/<impl>_<suite>.best.json
```

例如 per-tensor、A 按 M 连续、B 按 N 连续、TMA、legacy shape suite 的结果为：

```text
results/tuning/triton_per_tensor_a_m_b_n_tma_legacy.jsonl
results/tuning/triton_per_tensor_a_m_b_n_tma_legacy.best.json
```

JSONL 会保留每个 config 的编译失败原因、寄存器、local/shared memory、
occupancy、是否找到 WGMMA、短测和复测时间。`.best.json` 中同时有
结构化的 `unique_configs` 和可直接整理回 kernel `_CONFIGS` 的
`triton_configs` 字符串。

`*_n_transpose` 搜索的是显式重排之后的 K-major BMM kernel 配置，不把
transpose 时间混进 config 选择；端到端 transpose 成本仍由
`bench_bmm` 测量。之后增加 TMA kernel 时，只需在
`fp8_bench/tuning/adapters.py` 增加一个 adapter，不需要改搜索器。

## NCU

profile target 默认先 warmup 5 次。第一次 warmup 会完成 Triton autotune，
最后一次待采集 launch 放在 `fp8_bench_profile` NVTX range 中。NCU 只选择
这个 range，因此候选 config 和 warmup kernel 不会混进报告。

Quant：

```bash
mkdir -p reports
ncu \
  --set full \
  --import-source yes \
  --nvtx \
  --nvtx-include 'fp8_bench_profile/' \
  --kernel-name 'regex:.*fp8_quant_kernel.*' \
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
  --import-source yes \
  --nvtx \
  --nvtx-include 'fp8_bench_profile/' \
  --kernel-name 'regex:.*fp8_per_block_quant_kernel.*' \
  --launch-count 1 \
  -o reports/per_block_quant \
  python3 -m fp8_bench.profile_one \
    --op quant \
    --case q_smoke_3d \
    --impl triton_per_block_2d \
    --block-m 128 \
    --block-n 128 \
    --warmup 5
```

BMM：

```bash
mkdir -p reports
ncu \
  --set full \
  --import-source yes \
  --nvtx \
  --nvtx-include 'fp8_bench_profile/' \
  --kernel-name 'regex:.*batch_fp8_per_tensor_bmm_kernel.*' \
  --launch-count 1 \
  -o reports/bmm_b16_m512_n960_k1280 \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case b16_m512_n960_k1280 \
    --impl triton_per_tensor_a_k_b_n \
    --warmup 5
```

把 `--impl` 换成 `triton_per_tensor_a_k_b_k` 或
`triton_per_tensor_a_k_b_n_transpose` 即可 profile 另外两条路径。这里的
kernel-name 过滤器只采集最终 BMM kernel；显式重排的端到端成本以
`bench_bmm` 的 `bmm-only` 时间为准。

采集 TMA 时增加 `--backend tma`，并把 kernel filter 改为
`regex:.*batch_fp8_per_tensor_bmm_tma_kernel.*`。`--import-source yes`
会把关联到的 Triton 源码永久写进 `.ncu-rep`，报告复制到其他机器后也能查看。

Per-channel BMM 的 NCU 命令：

```bash
ncu \
  --set basic \
  --import-source yes \
  --nvtx \
  --nvtx-include 'fp8_bench_profile/' \
  --kernel-name 'regex:.*batch_fp8_per_channel_bmm_kernel.*' \
  --launch-count 1 \
  -o reports/per_channel_n \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case bmm_smoke_aligned \
    --impl triton_per_channel_a_k_b_n \
    --warmup 5
```

Per-block BMM：

```bash
ncu \
  --set basic \
  --import-source yes \
  --nvtx \
  --nvtx-include 'fp8_bench_profile/' \
  --kernel-name 'regex:.*batch_fp8_per_block_bmm_kernel.*' \
  --launch-count 1 \
  -o reports/per_block_n \
  python3 -m fp8_bench.profile_one \
    --op bmm \
    --case bmm_smoke_aligned \
    --impl triton_per_block_2d_a_k_b_n \
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

默认结果写到：

```text
results/quant.jsonl
results/bmm.jsonl
results/fsdp2.jsonl
```

`bench_bmm` 每次启动时会先清空目标 JSONL，再逐项实时写入本次运行的结果，
因此默认的 `results/bmm.jsonl` 不会混入以前的 benchmark。需要保留多次运行时，
应通过 `--results` 为每次运行指定不同文件名。Quant 和 FSDP2 仍保持追加写入。

三个 `bench_*` 会在每项完成后打印简短进度，并在本次运行结束时用 pandas
输出一张对齐宽表；性能和精度字段会合并在同一行。JSONL 仍然逐项立即写入，
因此长任务中途退出不会丢失已经完成的结果。

逐项进度里，BMM 会直接显示对应 scope 的耗时和 TFLOPS；Quant 与 FSDP2
显示 `duration`。汇总表中 Quant/FSDP2 的耗时统一命名为 `duration_ms`。
Triton Quant/BMM 在 autotune 完成后还会读取 kernel 的 `.best_config`：逐项输出
和 pandas 表显示紧凑配置，JSONL 的 `best_config` 字段保留完整结构化配置。
FSDP2 没有对应的单一 Triton autotuner，因此不输出 `best_config`。

已有 JSONL 可以重新排序、选择列或导出 CSV：

```bash
python3 -m fp8_bench.report results/bmm.jsonl \
  --kind bmm \
  --sort bmm_tflops \
  --descending

python3 -m fp8_bench.report results/bmm.jsonl \
  --column case \
  --column impl \
  --column backend \
  --column bmm_ms \
  --column bmm_tflops \
  --csv results/bmm.csv
```

可以用 `--results` 指定其他文件。每条记录带 shape、实现名、dtype、GPU、
CUDA/PyTorch 版本和 git commit。

新增实现时，在 `fp8_bench/impls/` 中实现函数，然后调用
`register_quant` 或 `register_bmm` 注册即可。
