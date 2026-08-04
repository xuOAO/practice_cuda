from __future__ import annotations

import triton
import triton.language as tl

# Supported fused activations. ``ACTIVATION`` is a ``tl.constexpr`` str so the
# branch is resolved at compile time and unsupported values fall through to the
# identity (no-op) path.
SUPPORTED = ("none", "gelu")


@triton.jit
def apply_activation(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == "gelu":
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). The 1/sqrt(2) literal is
        # inlined because Triton jit kernels cannot read module-level globals.
        return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    # "none" / "identity": leave the accumulator untouched.
    return x
