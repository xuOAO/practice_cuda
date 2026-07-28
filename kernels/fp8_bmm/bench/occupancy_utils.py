import ctypes

_libcuda = ctypes.CDLL("libcuda.so.1")

_get_occupancy = (
    _libcuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
)
_get_occupancy.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,   # compiled.function
    ctypes.c_int,      # threads_per_cta
    ctypes.c_size_t,   # dynamic shared memory bytes
]
_get_occupancy.restype = ctypes.c_int


def get_active_ctas_per_sm(
    compiled,
    threads_per_cta: int,
    shared_bytes: int,
) -> int:
    value = ctypes.c_int()

    status = _get_occupancy(
        ctypes.byref(value),
        ctypes.c_void_p(int(compiled.function)),
        threads_per_cta,
        shared_bytes,
    )

    if status != 0:
        raise RuntimeError(
            f"CUDA occupancy query failed: {status}"
        )

    return value.value