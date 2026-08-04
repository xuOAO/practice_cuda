import multiprocessing as mp
import os
import foo
import bar


def checkpoint_worker():
    # 用户代码可能在进入 target 前就被导入，
    # 所以 tracer 仍然由 sitecustomize 安装。
    print("checkpoint worker started")


def start_checkpoint_worker():
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=checkpoint_worker)

    old_trace_value = os.environ.get("TRACE_SPAWN_IMPORTS")
    old_cuda_value = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        # 由 spawn 子进程继承。
        os.environ["TRACE_SPAWN_IMPORTS"] = "1"

        # # 如果还要用无 GPU 环境定位用户顶层 CUDA 操作：
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""

        process.start()

    finally:
        # 子进程启动后，恢复父进程环境。
        if old_trace_value is None:
            os.environ.pop("TRACE_SPAWN_IMPORTS", None)
        else:
            os.environ["TRACE_SPAWN_IMPORTS"] = old_trace_value

        if old_cuda_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_value

    process.join()

if __name__ == "__main__":
    start_checkpoint_worker()