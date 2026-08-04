# debug_hooks/sitecustomize.py

import os


if os.environ.get("TRACE_SPAWN_IMPORTS") == "1":
    try:
        from import_cuda_tracer import install_import_tracer

        install_import_tracer()

        print(
            f"[import tracer installed by sitecustomize] "
            f"pid={os.getpid()} "
            f"ppid={os.getppid()}",
            flush=True,
        )

    except Exception:
        # sitecustomize 出错可能直接影响解释器启动，
        # 调试时最好明确打印异常。
        import traceback

        traceback.print_exc()
        raise