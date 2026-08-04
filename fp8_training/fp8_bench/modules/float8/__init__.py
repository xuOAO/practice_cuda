# Lets define a few top level things here
# Needed to load Float8TrainingTensor with weights_only = True
from torch.serialization import add_safe_globals

from .config import (
    CastConfig,
    Float8GemmConfig,
    Float8LinearConfig,
    Float8LinearRecipeName,
    ScalingGranularity,
    ScalingType,
)
from .float8_linear_utils import (
    _auto_filter_for_recipe,
    convert_to_float8_training,
)
from .float8_training_tensor import (
    Float8TrainingTensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from .fsdp_utils import precompute_float8_dynamic_scale_for_fsdp
from .inference import Float8MMConfig
from .types import FP8Granularity

add_safe_globals(
    [
        Float8TrainingTensor,
        ScaledMMConfig,
        GemmInputRole,
        LinearMMConfig,
        Float8MMConfig,
        ScalingGranularity,
    ]
)

__all__ = [
    # configuration
    "ScalingType",
    "ScalingGranularity",
    "Float8GemmConfig",
    "Float8LinearConfig",
    "Float8LinearRecipeName",
    "CastConfig",
    "ScalingGranularity",
    # top level UX
    "convert_to_float8_training",
    "precompute_float8_dynamic_scale_for_fsdp",
    "_auto_filter_for_recipe",
    # types
    "FP8Granularity",
    # note: Float8TrainingTensor and Float8Linear are not public APIs
]
