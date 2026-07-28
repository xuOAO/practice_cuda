"""Small FP8 experiment bench.

The package intentionally keeps policy out of the kernels: cases describe what
to run, the registry describes available implementations, and the entrypoints
decide whether to measure performance, accuracy, or profiling.
"""

__all__ = ["cases", "registry", "utils"]
