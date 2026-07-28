import torch
from enum import Enum
from typing import NamedTuple

class _FormatHelper(NamedTuple):
    max_fwd: float
    max_bwd: float
    dtype_fwd: torch.dtype
    dtype_bwd: torch.dtype

class FP8Format(Enum):
    E4M3 = _FormatHelper(
        max_fwd=448.0,
        max_bwd=448.0,
        dtype_fwd=torch.float8_e4m3fn, 
        dtype_bwd= torch.float8_e4m3fn,
    )

    E5M2 = _FormatHelper(
        max_fwd=57344.0,
        max_bwd=57344.0,
        dtype_fwd=torch.float8_e5m2,
        dtype_bwd=torch.float8_e5m2,
    )

    HYBRID = _FormatHelper(
        max_fwd=E4M3.max_fwd,
        max_bwd=E5M2.max_bwd,
        dtype_fwd=E4M3.dtype_fwd,
        dtype_bwd=E5M2.dtype_bwd,
    )