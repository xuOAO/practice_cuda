"""
import_cuda_tracer.py

用途：
1. 记录模块导入发生在哪个进程、文件和代码行。
2. 记录嵌套 import 链。
3. 检查 import 前后 torch.cuda.is_initialized() 是否从 False 变成 True。
4. 在 CUDA_VISIBLE_DEVICES="" 场景下，保留用户模块导入失败的完整异常。

注意：
- 必须在导入训练框架、用户模型之前安装。
- 这是调试工具，不建议长期用于生产环境。
"""

import builtins
import importlib
import inspect
import os
import sys
import threading
import traceback


# 保存原始函数，后面 wrapper 仍然要调用真正的 import。
_ORIGINAL_IMPORT = builtins.__import__
_ORIGINAL_IMPORT_MODULE = importlib.import_module


# 只详细打印这些模块。
# 请根据实际项目修改，例如：
# WATCH_PREFIXES = ("my_framework", "user_project")
#
# 设置为空元组 () 表示打印所有 import，日志会非常多。
WATCH_PREFIXES = (
    # "my_framework",
    # "user_project",
)


# 每个线程维护自己的 import 栈，避免多线程导入时互相干扰。
_THREAD_STATE = threading.local()


# 一个进程只报告第一次 False -> True。
_FIRST_CUDA_TRANSITION_REPORTED = False
_REPORT_LOCK = threading.Lock()


def _write_log(message):
    """统一输出到 stderr，并立即刷新。"""
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:
        # 调试探针不应该因为日志输出失败而影响业务程序。
        pass


def _should_log(module_name):
    """判断是否详细打印某个模块的 import 日志。"""
    if not WATCH_PREFIXES:
        return True

    return module_name.startswith(WATCH_PREFIXES)


def _get_import_stack():
    """获取当前线程的嵌套 import 栈。"""
    if not hasattr(_THREAD_STATE, "import_stack"):
        _THREAD_STATE.import_stack = []

    return _THREAD_STATE.import_stack


def _torch_cuda_initialized():
    """
    检查当前进程中，PyTorch CUDA 是否已经初始化。

    这里不能直接写 `import torch`，否则探针自己会改变
    程序原有的 import 顺序，甚至造成递归。

    返回：
        False：torch 尚未导入，或者 PyTorch CUDA 尚未初始化。
        True：PyTorch CUDA 已经完成 lazy initialization。
    """
    torch_module = sys.modules.get("torch")

    if torch_module is None:
        # 程序尚未导入 torch。
        return False

    try:
        cuda_module = getattr(torch_module, "cuda", None)

        if cuda_module is None:
            # torch 可能还处于“导入了一半”的状态。
            return False

        return bool(cuda_module.is_initialized())

    except Exception:
        # 探针不能干扰正常 import。
        return False


def _make_import_entry(kind, module_name, caller_frame):
    """
    创建一条 import 记录。

    kind:
        "import"         普通 import/from import
        "import_module"  importlib.import_module()
    """
    if caller_frame is None:
        return {
            "kind": kind,
            "module": module_name,
            "file": "<unknown>",
            "line": 0,
            "scope": "<unknown>",
        }

    return {
        "kind": kind,
        "module": module_name,
        "file": caller_frame.f_code.co_filename,
        "line": caller_frame.f_lineno,
        "scope": caller_frame.f_code.co_name,
    }


def _format_entry(entry):
    """把一条 import 记录格式化成人类容易阅读的字符串。"""
    return (
        f"{entry['kind']} {entry['module']} "
        f"requested at {entry['file']}:{entry['line']} "
        f"scope={entry['scope']}"
    )


def _report_import_stack(title, import_stack):
    """打印当前完整的嵌套 import 链。"""
    _write_log("")
    _write_log(
        f"===== {title} "
        f"pid={os.getpid()} ppid={os.getppid()} ====="
    )

    for depth, entry in enumerate(import_stack):
        indent = "  " * depth
        _write_log(f"{indent}└─ {_format_entry(entry)}")

    _write_log("===== end import stack =====")
    _write_log("")


def _run_with_trace(entry, original_call):
    """
    所有 import wrapper 的公共逻辑。

    original_call 是一个函数，调用它才会真正执行原始 import。
    """
    global _FIRST_CUDA_TRANSITION_REPORTED

    import_stack = _get_import_stack()
    import_stack.append(entry)

    # 进入本次 import 前，记录 PyTorch CUDA 状态。
    cuda_before = _torch_cuda_initialized()

    if _should_log(entry["module"]):
        depth = len(import_stack) - 1
        indent = "  " * depth

        _write_log(
            f"{indent}[import begin] "
            f"pid={os.getpid()} "
            f"cuda_initialized={cuda_before} "
            f"{_format_entry(entry)}"
        )

    try:
        # 真正执行原始 import。
        #
        # 如果被导入模块中存在顶层代码：
        #
        #     model = MyModel()
        #     model.to("cuda")
        #
        # 那么这些代码会在这个调用内部执行。
        return original_call()

    except BaseException:
        # CUDA_VISIBLE_DEVICES="" 时，用户模块顶层的 .to("cuda")
        # 很可能在这里抛出异常。
        if _should_log(entry["module"]):
            _report_import_stack(
                title=f"IMPORT FAILED: {entry['module']}",
                import_stack=list(import_stack),
            )

            # format_exc() 会包含具体 Python 文件和行号。
            _write_log(traceback.format_exc())

        # 不能吞掉原异常，否则程序可能带着未完整初始化的模块继续运行。
        raise

    finally:
        # 无论 import 成功还是失败，都检查 CUDA 状态。
        cuda_after = _torch_cuda_initialized()

        # 只关注 False -> True。
        if not cuda_before and cuda_after:
            with _REPORT_LOCK:
                if not _FIRST_CUDA_TRANSITION_REPORTED:
                    _FIRST_CUDA_TRANSITION_REPORTED = True

                    _report_import_stack(
                        title="PYTORCH CUDA INITIALIZED DURING IMPORT",
                        import_stack=list(import_stack),
                    )

        # 恢复当前线程的 import 栈。
        import_stack.pop()


def _traced_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    包装 builtins.__import__。

    覆盖：
        import foo
        import foo.bar
        from foo import bar
        from .foo import bar
    """
    caller_frame = inspect.currentframe().f_back

    try:
        entry = _make_import_entry(
            kind="import",
            module_name=name,
            caller_frame=caller_frame,
        )
    finally:
        # frame 会引用局部变量，及时删除可以减少引用环。
        del caller_frame

    return _run_with_trace(
        entry=entry,
        original_call=lambda: _ORIGINAL_IMPORT(
            name,
            globals,
            locals,
            fromlist,
            level,
        ),
    )


def _traced_import_module(name, package=None):
    """
    包装 importlib.import_module。

    很多训练框架会根据配置动态导入用户模型：

        importlib.import_module("user_project.models.llama")
    """
    caller_frame = inspect.currentframe().f_back

    try:
        entry = _make_import_entry(
            kind="import_module",
            module_name=name,
            caller_frame=caller_frame,
        )
    finally:
        del caller_frame

    return _run_with_trace(
        entry=entry,
        original_call=lambda: _ORIGINAL_IMPORT_MODULE(name, package),
    )


def install_import_tracer():
    """启用 import 拦截。应当在导入训练框架之前调用。"""
    builtins.__import__ = _traced_import
    importlib.import_module = _traced_import_module

    _write_log(
        f"[import tracer installed] "
        f"pid={os.getpid()} ppid={os.getppid()}"
    )


def uninstall_import_tracer():
    """恢复 Python 原始 import 行为。"""
    if builtins.__import__ is _traced_import:
        builtins.__import__ = _ORIGINAL_IMPORT

    if importlib.import_module is _traced_import_module:
        importlib.import_module = _ORIGINAL_IMPORT_MODULE

    _write_log(f"[import tracer uninstalled] pid={os.getpid()}")