# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# torchAO repository.

"""Quantization granularity types required by the vendored float8 package."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Granularity:
    """Base class for quantization granularity specifications."""


@dataclass(frozen=True)
class PerTensor(Granularity):
    """Use one set of quantization parameters for the entire tensor."""


@dataclass(frozen=True)
class PerGroup(Granularity):
    """Use one set of quantization parameters per group of elements."""

    group_size: int


@dataclass(frozen=True)
class PerRow(Granularity):
    """Reduce over ``dim`` and keep separate parameters for other dimensions."""

    dim: int = -1


@dataclass(frozen=True)
class PerBlock(Granularity):
    """Use a multidimensional block as the quantization unit."""

    block_size: tuple[int, ...]


torch.serialization.add_safe_globals([PerBlock, PerRow, PerTensor])
