# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# torchAO repository.

"""Small torchAO utility subset required by the vendored float8 package."""

import re

import torch


def _is_rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def is_MI300() -> bool:
    if not _is_rocm():
        return False
    arch_name = torch.cuda.get_device_properties(0).gcnArchName
    return any(arch in arch_name for arch in ("gfx940", "gfx941", "gfx942"))


def is_MI350() -> bool:
    if not _is_rocm():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def is_sm_at_least_89() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (8, 9)
    )


def _parse_version(version_string: str) -> tuple[int, int, int]:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_string)
    if match is None:
        raise ValueError(f"Invalid version string format: {version_string}")
    major, minor, patch = map(int, match.groups())
    if re.search(r"(git|dev)", version_string):
        patch = -1
    return major, minor, patch


def torch_version_at_least(min_version: str) -> bool:
    # Internal fbcode builds do not expose git_version and are always treated
    # as recent enough by torchAO.
    if not hasattr(torch.version, "git_version"):
        return True
    return _parse_version(torch.__version__) >= _parse_version(min_version)
