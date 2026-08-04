import torch

@torch.compile
def per_channel_quant_torch_compile(
    x: torch.Tensor,
    channel_axis: int,
    fp8_dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduction_dim = -1 if channel_axis == -2 else -2
    x_safe_amax = torch.amax(
        torch.abs(x.float()),
        dim=reduction_dim,
    ).clamp_min(eps)
    fp8_max = torch.finfo(fp8_dtype).max
    quant_scale = fp8_max / x_safe_amax
    dequant_scale = x_safe_amax / fp8_max
    output = torch.clamp(
        x.float() * quant_scale.unsqueeze(reduction_dim),
        min=-fp8_max,
        max=fp8_max,
    ).to(fp8_dtype)
    return output, dequant_scale

WARMUP = 10

if __name__ == "__main__":
    x = torch.randn(80, 2048, 1280, device="cuda", dtype=torch.float32)
    for _ in range(WARMUP):
        output, dequant_scale = per_channel_quant_torch_compile(
            x,
            channel_axis=-1,
            fp8_dtype=torch.float8_e4m3fn,
            eps=1e-6,
        )
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        # with_stack=True,
    ) as prof:
        output, dequant_scale = per_channel_quant_torch_compile(
            x,
            channel_axis=-1,
            fp8_dtype=torch.float8_e4m3fn,
            eps=1e-6,
        )
        torch.cuda.synchronize()

    prof.export_chrome_trace("per_channel_quant_torch_compile.json")