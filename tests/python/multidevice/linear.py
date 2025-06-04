# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial, lru_cache
from fusion_definition_wrapper import FusionDefinitionWrapper
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement
from typing import Iterable


@dataclass(frozen=True)
class LinearConfig:
    in_features: int
    out_features: int
    # Whether the input/output tensors have a leading batch dimension. This is
    # typically true for MLA and false for MoE.
    has_batch: bool


# I omitted biases because DeepSeek V3 uses non-biased linear layers in MLA and
# MoE.
def define_linear_forward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    inp_shape = [-1, -1, e_in] if config.has_batch else [-1, e_in]
    inp = fd.define_tensor(inp_shape, contiguity=True, dtype=DataType.BFloat16)
    weight = fd.define_tensor([e_out, e_in], contiguity=True, dtype=DataType.BFloat16)
    out = fd.ops.linear(inp, weight)
    fd.add_output(out)


def define_linear_backward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    x_shape = [-1, -1, e_in] if config.has_batch else [-1, e_in]
    x = fd.define_tensor(x_shape, contiguity=True, dtype=DataType.BFloat16)
    w = fd.define_tensor([e_out, e_in], contiguity=True, dtype=DataType.BFloat16)
    grad_shape = [-1, -1, e_out] if config.has_batch else [-1, e_out]
    grad = fd.define_tensor(grad_shape, contiguity=True, dtype=DataType.BFloat16)

    grad_x = fd.ops.matmul(grad, w)

    grad_flat_t = fd.ops.reshape(grad_flat_t, [-1, e_out]) if config.has_batch else grad
    grad_flat_t = fd.ops.permute(grad_flat_t, [1, 0])

    x_flat = fd.ops.reshape(x, [-1, e_in]) if config.has_batch else x

    grad_w = fd.ops.matmul(grad_flat_t, x_flat)

    fd.add_output(grad_x)
    fd.add_output(grad_w)


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


# Cache so we create only one FusionDefinitionWrapper per math definition. This
# is more efficient and is necessary to avoid #4507.
@lru_cache
def get_fusion_definition_wrapper(
    compute_type: ComputeType, linear_config: LinearConfig
) -> FusionDefinitionWrapper:
    match compute_type:
        case ComputeType.FORWARD:
            fn = define_linear_forward
        case ComputeType.BACKWARD:
            fn = define_linear_backward

    return FusionDefinitionWrapper(partial(fn, linear_config))


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: DTensor,
        weight: DTensor,
    ):
        assert input.dim() in (2, 3)
        op = get_fusion_definition_wrapper(
            ComputeType.FORWARD,
            LinearConfig(weight.size(1), weight.size(0), has_batch=(input.dim() == 3)),
        )
        (output,) = op([input, weight])
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        input, weight = ctx.saved_tensors
        assert input.dim() in (2, 3)

        op = get_fusion_definition_wrapper(
            ComputeType.BACKWARD,
            LinearConfig(weight.size(1), weight.size(0), has_batch=(input.dim() == 3)),
        )
        grad_x, grad_w = op([input, weight, grad_output])
        return (grad_x, grad_w)


class TensorParallelLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_placements: Iterable[Placement] = [],
    ):
        # Unlike normal layers whose `__init__` allocates the parameters,
        # `TensorParallelLinear` is expected to be created from a
        # non-distributed Linear layer via the `distribute` method. Therefore,
        # here, we construct super() with no memory allocated for weights. The
        # weights will be derived by the `distribute` method.
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.in_placements = in_placements

    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, in_placements={self.in_placements}, weight_placements={self.weight.data.placements})"

    @classmethod
    def distribute(
        cls,
        linear: torch.nn.Linear,
        mesh: dist.device_mesh.DeviceMesh,
        in_placements: Iterable[Placement] = [],
        weight_placements: Iterable[Placement] = [],
    ):
        tp_linear = cls(linear.in_features, linear.out_features, in_placements)
        tp_linear.weight = torch.nn.Parameter(
            dist.tensor.distribute_tensor(linear.weight, mesh, weight_placements)
        )
        return tp_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_dtensor = DTensor.from_local(
            input, self.weight.device_mesh, self.in_placements
        )
        out_dtensor = LinearFunction.apply(in_dtensor, self.weight)
        return out_dtensor.to_local()
