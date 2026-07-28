"""Importing this module registers the built-in experimental implementations."""

from . import triton_per_channel as _triton_per_channel
from . import triton_per_tensor as _triton_per_tensor

__all__ = []
