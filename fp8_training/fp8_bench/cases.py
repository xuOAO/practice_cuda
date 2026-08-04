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

# batch_dB_kernel.py labels these tuples as (B, M, N, K), but its dB
# operation is [B,K,M] @ [B,M,N] -> [B,K,N].  Keep the source labels here
# for traceability, then convert them to this benchmark's conventional
# (B, output_M, output_N, reduction_K) order below.
_DB_BMM_SOURCE_SHAPES_1 = [
    (36, 1024, 512, 4096),
    (36, 1024, 2048, 4096),
    (32, 1024, 4608, 576),
    (32, 1024, 2304, 288),
    (32, 1024, 2304, 4608),
    (32, 1024, 288, 2304),
    (36, 1024, 2048, 256),
    (36, 1024, 4096, 512),
    (36, 1024, 4096, 2048),
    (32, 1024, 576, 4608),
    (36, 1024, 2048, 2048),
    (32, 1024, 4608, 2304),
    (36, 1024, 256, 2048),
]

_DB_BMM_SHAPES_1 = [
    (batch, output_k, output_n, reduction_m)
    for batch, reduction_m, output_n, output_k in _DB_BMM_SOURCE_SHAPES_1
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
    "dB_bmm_shapes_1": [_bmm_case(shape) for shape in _DB_BMM_SHAPES_1],
}


# "smoke" is only for checking that the remote environment and command line
# are wired correctly. Quant legacy follows the BMM legacy workload, with
# duplicate operand shapes removed.
QUANT_SUITES: dict[str, list[QuantCase]] = {
    "smoke": [
        QuantCase("q_smoke_2d", (128, 256)),
        QuantCase("q_smoke_3d", (2, 128, 256)),
    ],
    "legacy": [
        QuantCase("q_b16_m512_k1280", (16, 512, 1280)),
        QuantCase("q_b16_m1280_k960", (16, 1280, 960)),
        QuantCase("q_b16_m2048_k1280", (16, 2048, 1280)),
        QuantCase("q_b16_m1280_k640", (16, 1280, 640)),
        QuantCase("q_b16_m1280_k1280", (16, 1280, 1280)),
        QuantCase("q_b16_m2048_k960", (16, 2048, 960)),
        QuantCase("q_b16_m960_k1280", (16, 960, 1280)),
        QuantCase("q_b32_m2048_k1280", (32, 2048, 1280)),
        QuantCase("q_b32_m1280_k640", (32, 1280, 640)),
        QuantCase("q_b32_m2048_k1600", (32, 2048, 1600)),
        QuantCase("q_b32_m1600_k960", (32, 1600, 960)),
        QuantCase("q_b32_m1600_k1280", (32, 1600, 1280)),
        QuantCase("q_b32_m1600_k1600", (32, 1600, 1600)),
        QuantCase("q_b32_m2048_k960", (32, 2048, 960)),
        QuantCase("q_b32_m960_k1280", (32, 960, 1280)),
        QuantCase("q_b80_m512_k1280", (80, 512, 1280)),
        QuantCase("q_b80_m1280_k640", (80, 1280, 640)),
        QuantCase("q_b80_m2048_k640", (80, 2048, 640)),
        QuantCase("q_b80_m640_k640", (80, 640, 640)),
        QuantCase("q_b80_m640_k960", (80, 640, 960)),
        QuantCase("q_b80_m640_k1280", (80, 640, 1280)),
        QuantCase("q_b80_m2048_k960", (80, 2048, 960)),
        QuantCase("q_b80_m960_k1280", (80, 960, 1280)),
        QuantCase("q_b80_m2048_k1280", (80, 2048, 1280)),
    ],
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
