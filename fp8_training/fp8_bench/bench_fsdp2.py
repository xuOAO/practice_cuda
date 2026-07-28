from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch import nn

from fp8_bench.utils import append_jsonl, environment_info, seed_everything


def fake_quant_ste(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Training-compatible fake FP8 used to validate the FSDP2 bench path.

    Forward observes per-tensor FP8 quantization error; backward uses a
    straight-through estimator. The matmul itself is BF16, so this provider is
    an infrastructure case rather than an FP8 tensor-core performance result.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    with torch.no_grad():
        max_abs = x.detach().abs().amax().clamp(min=1e-12)
        dequant_scale = max_abs / fp8_max
        dequant = (
            (x.detach() / dequant_scale).to(fp8_dtype).to(x.dtype)
            * dequant_scale
        )
    return x + (dequant - x).detach()


class BenchLinear(nn.Linear):
    def __init__(self, *args, fp8: bool, fp8_dtype: torch.dtype, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_fp8 = fp8
        self.fp8_dtype = fp8_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp8:
            return functional.linear(x, self.weight, self.bias)
        quant_x = fake_quant_ste(x, self.fp8_dtype)
        quant_weight = fake_quant_ste(self.weight, self.fp8_dtype)
        return functional.linear(quant_x, quant_weight, self.bias)


class Block(nn.Module):
    def __init__(self, hidden: int, ffn: int, *, fp8: bool, fp8_dtype: torch.dtype) -> None:
        super().__init__()
        self.up = BenchLinear(hidden, ffn, fp8=fp8, fp8_dtype=fp8_dtype)
        self.down = BenchLinear(ffn, hidden, fp8=fp8, fp8_dtype=fp8_dtype)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(functional.gelu(self.up(x)))
        return residual + x


class ToyModel(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        ffn: int,
        *,
        fp8: bool,
        fp8_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(hidden, ffn, fp8=fp8, fp8_dtype=fp8_dtype)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small FSDP2 end-to-end training bench.")
    parser.add_argument("--impl", choices=["bf16", "fake_fp8"], default="fake_fp8")
    parser.add_argument("--fp8-dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default="results/fsdp2.jsonl")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this script with torchrun")
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        # PyTorch releases before the public FSDP2 namespace exposed the same
        # composable API from this location.
        from torch.distributed._composable.fsdp import fully_shard

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_everything(args.seed)

    fp8_dtype = (
        torch.float8_e4m3fn if args.fp8_dtype == "e4m3" else torch.float8_e5m2
    )
    model = ToyModel(
        args.layers,
        args.hidden,
        args.ffn,
        fp8=args.impl == "fake_fp8",
        fp8_dtype=fp8_dtype,
    ).to(device=device, dtype=torch.bfloat16)
    for block in model.blocks:
        fully_shard(block)
    fully_shard(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(
        (args.batch, args.seq, args.hidden),
        device=device,
        dtype=torch.bfloat16,
    )

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.iters):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        losses.append(loss)

    local_median = torch.tensor(
        statistics.median(elapsed_ms),
        device=device,
        dtype=torch.float64,
    )
    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated(device),
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(local_median, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        median_ms = float(local_median.item())
        samples_per_second = args.batch * world_size / (median_ms / 1000)
        # Each block has two GEMMs (H->F and F->H). Forward is 4*T*H*F
        # FLOPs; forward + dgrad + wgrad is approximately three times that.
        # This deliberately excludes norm, GELU, communication and optimizer
        # work, so the metric is an estimated training GEMM throughput.
        tokens = args.batch * args.seq * world_size
        estimated_gemm_flops = (
            12 * tokens * args.layers * args.hidden * args.ffn
        )
        estimated_gemm_tflops = estimated_gemm_flops / median_ms / 1e9
        record = {
            "kind": "fsdp2",
            "environment": environment_info(),
            "impl": args.impl,
            "fp8_dtype": args.fp8_dtype,
            "world_size": world_size,
            "layers": args.layers,
            "hidden": args.hidden,
            "ffn": args.ffn,
            "batch_per_rank": args.batch,
            "seq": args.seq,
            "median_step_ms": median_ms,
            "samples_per_second": samples_per_second,
            "estimated_gemm_flops_per_step": estimated_gemm_flops,
            "estimated_gemm_tflops": estimated_gemm_tflops,
            "max_peak_memory_bytes": int(peak_memory.item()),
            "first_measured_loss": losses[0],
            "last_loss": losses[-1],
            "all_step_ms_rank0": elapsed_ms,
        }
        append_jsonl(args.results, record)
        print(
            f"FSDP2 {args.impl}: {median_ms:.3f} ms/step,"
            f" {samples_per_second:.2f} samples/s,"
            f" est_gemm={estimated_gemm_tflops:.2f} TFLOPS,"
            f" peak={peak_memory.item() / 1024**3:.2f} GiB,"
            f" loss={losses[0]:.6g}->{losses[-1]:.6g}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
