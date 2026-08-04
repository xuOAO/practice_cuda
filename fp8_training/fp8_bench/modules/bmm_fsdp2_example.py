from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Keep direct-script invocation working from fp8_bench/modules.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from fp8_bench.modules.bmm_linear import (
    BMMLinear,
    Float8BMMLinearConfig,
    convert_to_float8_bmm_training,
    precompute_bmm_float8_dynamic_scale_for_fsdp,
)
from fp8_bench.modules.float8.config import CastConfig, ScalingGranularity

BATCH = 32
SEQUANCE = 100
D_MODEL, D_FF = 4096, 4096 * 2
ITERS = 10
RANK = None
WORLD_SIZE = None
DEVICE = None
WARMUP = 5
PROFILE_WAIT = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 3


def setup():
    local_rank: int = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global DEVICE
    DEVICE = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", device_id=DEVICE)
    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()
    global RANK
    RANK = rank
    global WORLD_SIZE
    WORLD_SIZE = world_size


class MLPModule(torch.nn.Module):
    def __init__(self, batch_size, d_model, d_ff):
        super().__init__()
        self.linear1 = BMMLinear(batch_size, d_model, d_ff)
        self.linear2 = BMMLinear(batch_size, d_ff, d_model)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ToyModule(torch.nn.Module):
    def __init__(self, batch_size, d_model, d_ff, num_block: int = 32):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                MLPModule(batch_size, d_model, d_ff)
                for _ in range(num_block)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def train_step(model, optimizer, x):
    optimizer.zero_grad(set_to_none=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    return loss.detach()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        help="Use FP8 for training test",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Use torch.compile for training test",
    )
    parser.add_argument(
        "--use-precompute-scales",
        action="store_true",
        help="Use precomputed scales for FP8 training test",
    )
    parser.add_argument(
        "--scaling-granularity",
        choices=("per-tensor", "per-channel"),
        default="per-tensor",
        help="FP8 scaling recipe used by all three training BMMs",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a steady-state rank-0 CPU/CUDA profiler trace",
    )
    parser.add_argument(
        "--profile-all-ranks",
        action="store_true",
        help="Capture one profiler trace per rank (implies --profile)",
    )
    parser.add_argument(
        "--trace-dir",
        default="traces/bmm_fsdp2",
        help="Directory in which profiler traces are written",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--sequence-length", type=int, default=SEQUANCE)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--d-ff", type=int, default=D_FF)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    return parser.parse_args()


def run_profile(train_step_fn, model, optimizer, x, trace_dir, profile_all_ranks):
    profile_this_rank = profile_all_ranks or RANK == 0
    profiler = None
    trace_path = Path(trace_dir).resolve()

    if profile_this_rank:
        trace_path.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT,
                warmup=PROFILE_WARMUP,
                active=PROFILE_ACTIVE,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(trace_path),
                worker_name=f"rank{RANK}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()

    num_steps = PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE
    dist.barrier()
    for _ in range(num_steps):
        if profiler is not None:
            with torch.profiler.record_function("train_step"):
                loss = train_step_fn(model, optimizer, x)
            torch.cuda.synchronize()
            profiler.step()
        else:
            loss = train_step_fn(model, optimizer, x)

    torch.cuda.synchronize(DEVICE)
    dist.barrier()

    if profiler is not None:
        profiler.stop()
        if RANK == 0:
            print(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            )
            print(f"Profiler trace written to: {trace_path}")

    return loss


def main():
    args = parse_args()
    args.profile = args.profile or args.profile_all_ranks
    setup()
    torch.manual_seed(42)

    model = ToyModule(
        args.batch_size,
        args.d_model,
        args.d_ff,
        num_block=args.num_blocks,
    ).to(
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    if args.use_fp8:
        granularity = (
            ScalingGranularity.AXISWISE
            if args.scaling_granularity == "per-channel"
            else ScalingGranularity.TENSORWISE
        )
        fp8_config = Float8BMMLinearConfig(
            cast_config_input=CastConfig(scaling_granularity=granularity),
            cast_config_weight=CastConfig(scaling_granularity=granularity),
            cast_config_grad_output=CastConfig(scaling_granularity=granularity),
            enable_fsdp_float8_all_gather=True,
            round_scales_to_power_of_2=(
                granularity is ScalingGranularity.AXISWISE
            ),
        )
        model = convert_to_float8_bmm_training(model, config=fp8_config)

    mesh = init_device_mesh(
        "cuda",
        (WORLD_SIZE,),
        mesh_dim_names=("fsdp",),
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if args.use_precompute_scales:
        # Initialize the tensor-subclass field before torch.compile traces the
        # optimizer. Adding it for the first time from the post-step hook would
        # change the flattened optimizer inputs between the first two steps.
        precompute_bmm_float8_dynamic_scale_for_fsdp(model)
        optimizer.register_step_post_hook(
            lambda *args, **kwargs:
            precompute_bmm_float8_dynamic_scale_for_fsdp(model)
        )
    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.d_model,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    real_train_step = train_step
    if args.use_torch_compile:
        real_train_step = torch.compile(train_step, backend="inductor")

    for _ in range(args.warmup):
        loss = real_train_step(model, optimizer, x)

    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    if args.profile:
        loss = run_profile(
            real_train_step,
            model,
            optimizer,
            x,
            args.trace_dir,
            args.profile_all_ranks,
        )
        peak_allocated = torch.cuda.max_memory_allocated(DEVICE)
        if RANK == 0:
            print(f"Final loss: {loss.item()}")
            print(
                "Profiled peak memory allocated: "
                f"{peak_allocated / (1024**3):.2f} GB"
            )
        return

    dist.barrier()
    st = time.perf_counter()

    for step in range(args.iters):
        loss = real_train_step(model, optimizer, x)

    torch.cuda.synchronize(DEVICE)
    ed = time.perf_counter()

    if RANK == 0:
        print(f"Step {step}: Loss {loss.item()}")

    gb = 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(DEVICE)
    elapsed_time = torch.tensor(ed - st, device=DEVICE, dtype=torch.float64)
    dist.all_reduce(elapsed_time, op=dist.ReduceOp.MAX)
    e2e_time = elapsed_time.item()

    if RANK == 0:
        print(f"Peak memory allocated: {peak_allocated / gb:.2f} GB")
        print(
            f"Total time for {args.iters} iterations: "
            f"{e2e_time:.2f} seconds"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
