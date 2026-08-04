from __future__ import annotations

import json
import shutil

import pandas as pd
import torch
import triton

from fp8_bench.cases import BMM_SUITES, QUANT_SUITES
from fp8_bench.registry import QUANT_IMPLS, bmm_impl_names, load_builtin_impls
from fp8_bench.utils import environment_info


def main() -> None:
    load_builtin_impls()
    info = environment_info()
    info["triton"] = triton.__version__
    info["pandas"] = pd.__version__
    info["ncu"] = shutil.which("ncu")
    info["quant_impls"] = sorted(QUANT_IMPLS)
    info["bmm_impls"] = bmm_impl_names()
    info["quant_cases"] = {
        suite: [case.name for case in cases]
        for suite, cases in QUANT_SUITES.items()
    }
    info["bmm_cases"] = {
        suite: [case.name for case in cases]
        for suite, cases in BMM_SUITES.items()
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA")


if __name__ == "__main__":
    main()
