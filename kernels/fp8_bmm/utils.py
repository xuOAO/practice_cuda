import torch

def time_consumption(
    func,
    *args,
    warmup=20,
    iters=200,
    **kwargs
):
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    return elapsed / iters