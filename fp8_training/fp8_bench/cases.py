from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantCase:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BMMCase:
    name: str
    batch: int
    m: int
    n: int
    k: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.m, self.n, self.k


# "smoke" is only for checking that the remote environment and command line
# are wired correctly. "legacy" carries over the shapes used by the previous
# experiment.
QUANT_SUITES: dict[str, list[QuantCase]] = {
    "smoke": [
        QuantCase("q_smoke_2d", (128, 256)),
        QuantCase("q_smoke_3d", (2, 128, 256)),
    ],
    "legacy": [
        QuantCase("q_b32_m2048_k960", (32, 2048, 960)),
        QuantCase("q_b80_m2048_k640", (80, 2048, 640)),
        QuantCase("q_b14_m2048_k8192", (14, 2048, 8192)),
    ],
}


_LEGACY_BMM_SHAPES = [
    (16, 512, 960, 1280),
    (16, 2048, 640, 1280),
    (16, 2048, 1280, 1280),
    (16, 2048, 1280, 960),
    (32, 2048, 640, 1280),
    (32, 2048, 960, 1600),
    (32, 2048, 1280, 1600),
    (32, 2048, 1600, 1600),
    (32, 2048, 1280, 960),
    (80, 512, 640, 1280),
    (80, 2048, 640, 640),
    (80, 2048, 960, 640),
    (80, 2048, 1280, 640),
    (80, 2048, 1280, 960),
    (80, 2048, 640, 1280),
]


def _bmm_case(shape: tuple[int, int, int, int]) -> BMMCase:
    batch, m, n, k = shape
    return BMMCase(f"b{batch}_m{m}_n{n}_k{k}", batch, m, n, k)


BMM_SUITES: dict[str, list[BMMCase]] = {
    "smoke": [
        BMMCase("bmm_smoke_aligned", 2, 128, 128, 128),
        BMMCase("bmm_smoke_unaligned", 3, 129, 131, 127),
    ],
    "legacy": [_bmm_case(shape) for shape in _LEGACY_BMM_SHAPES],
}


def find_quant_case(name: str) -> QuantCase:
    for cases in QUANT_SUITES.values():
        for case in cases:
            if case.name == name:
                return case
    raise KeyError(f"unknown quant case: {name}")


def find_bmm_case(name: str) -> BMMCase:
    for cases in BMM_SUITES.values():
        for case in cases:
            if case.name == name:
                return case
    raise KeyError(f"unknown bmm case: {name}")
