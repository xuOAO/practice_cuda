from __future__ import annotations

import json
import shutil

import torch
import triton

from fp8_bench.cases import BMM_SUITES, QUANT_SUITES
from fp8_bench.registry import BMM_IMPLS, QUANT_IMPLS, load_builtin_impls
from fp8_bench.utils import environment_info


def main() -> None:
    load_builtin_impls()
    info = environment_info()
    info["triton"] = triton.__version__
    info["ncu"] = shutil.which("ncu")
    info["quant_impls"] = sorted(QUANT_IMPLS)
    info["bmm_impls"] = sorted(BMM_IMPLS)
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
