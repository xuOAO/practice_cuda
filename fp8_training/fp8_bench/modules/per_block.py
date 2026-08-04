import torch
from torch import nn
from fp8_training.fp8_bench.impls.triton_per_block import (
    triton_per_block_quant,
    triton_per_block_bmm
)
from float8.config import (
    Float8LinearConfig
)

from float8.distributed_utils import tensor_already_casted_to_fp8

class FP8BatchMatMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A,
        B,
        bias,
        use_bias: bool = False,
        config: Float8LinearConfig = None, 
    ):
        ctx.use_bias = use_bias
        ctx.config = config

        assert A.dim() == 3 and B.dim() == 3, "A and B must be 3D tensors"
        assert A.shape[0] == B.shape[0], "Batch sizes of A and B must match"
        assert A.shape[2] == B.shape[2], "Inner dimensions of A and B must match for batch matmul"

        assert (use_bias and bias is not None) or (not use_bias and bias is None), "Bias must be provided if use_bias is True"
        if use_bias:
            assert bias.dim() == 3, "Bias must be a 3D tensor"
            assert bias.shape[0] == A.shape[0], "Batch size of bias must match A and B"
            assert bias.shape[1] == 1, "Bias must have shape (batch, 1, out_features)"
            assert bias.shape[2] == A.shape[1], "Bias must have shape (batch, 1, out_features)"

        # before all-gather
        # linear.weight = DTensor
        # Dtensor._local_tensor = WeightWithDynamicFloat8CastTensor
        # WeightWithDynamicFloat8CastTensor._tensor = torch.Tensor (sharded weight)

        # run WeightWithDynamicFloat8CastTensor.fsdp_pre_all_gather(self, mesh)
        #   -> allgather_inputs, metadata

        # allgather(allgather_inputs) -> allgather_outputs

        # run WeightWithDynamicFloat8CastTensor.fsdp_post_all_gather(allgather_outputs, metadata, ...)
        #   -> Float8TrainingTensor, (data,)

        # fsdp_param._unsharded_param = Float8TrainingTensor

        # after all-gather
        # linear.weight = fsdp_param._unsharded_param = Float8TrainingTensor
        # Float8TrainingTensor._data & Float8TrainingTensor._scale
        if tensor_already_casted_to_fp8(A):
            A_fp8, A_scale = A._data, A._scale
        else:
            quant_res_a = triton_per_block_quant(A)
            A_fp8, A_scale = quant_res_a.tensor, quant_res_a.scale

        if tensor_already_casted_to_fp8(B):
            B_fp8, B_scale = B._data, B._scale
        else:
            quant_res_b = triton_per_block_quant(B)
            B_fp8, B_scale = quant_res_b.tensor, quant_res_b.scale

        ctx.save_for_backward(A_fp8, B_fp8, A_scale, B_scale)


class FP8BatchMatMulDense(nn.Module):
    def __init__(
        self,
        batch: int,
        in_features: int,
        out_features: int,
        activation=None,
        use_bias: bool = True,
        weight_initializer=torch.nn.init.normal_,
        bias_initializer=torch.nn.init.zeros_,
        fp8_all_gather=False,
        fp8_force_recompute=True,
        fp8_compute=True,
        **kwargs
    ):
        super().__init__()

        self.activation = activation
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.no_clip = kwargs.get("no_clip", False)

        self.fp8_compute = fp8_compute
        self.fp8_all_gather = fp8_all_gather
        self.fp8_force_recompute = fp8_force_recompute
        # k-major weight
        self.weight_shape = (batch, out_features, in_features)
        self.weight = torch.empty(self.weight_shape, dtype=torch.float16)
        self.weight_initializer(self.weight)

        if self.fp8_all_gather:
            # TODO
            assert False, "FP8 all-gather not implemented yet"
        else:
            self.weight = torch.nn.Parameter(self.weight)

        if self.use_bias:
            self.bias = torch.empty((batch, 1, out_features), dtype=torch.float16)
            self.bias = torch.nn.Parameter(self.bias)
            self.bias_initializer(self.bias)
        else:
            self.bias = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fp8_compute:


        