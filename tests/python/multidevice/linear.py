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
from torch.distributed.tensor.placement_types import Shard, Replicate


@dataclass(frozen=True)
class LinearConfig:
    in_features: int
    out_features: int
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

    grad_flat_t = grad
    if config.has_batch:
        grad_flat_t = fd.ops.reshape(grad_flat_t, [-1, e_out])
    grad_flat_t = fd.ops.permute(grad_flat_t, [1, 0])

    x_flat = x
    if config.has_batch:
        x_flat = fd.ops.reshape(x, [-1, e_in])

    grad_w = fd.ops.matmul(grad_flat_t, x_flat)

    fd.add_output(grad_x)
    fd.add_output(grad_w)


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


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
        assert input.dim() in [2, 3]
        op = get_fusion_definition_wrapper(
            ComputeType.FORWARD,
            LinearConfig(weight.size(1), weight.size(0), input.dim() == 3),
        )
        (output,) = op([input, weight])
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        input, weight = ctx.saved_tensors
        assert input.dim() in [2, 3]

        op = get_fusion_definition_wrapper(
            ComputeType.BACKWARD,
            LinearConfig(weight.size(1), weight.size(0), input.dim() == 3),
        )
        grad_x, grad_w = op([input, weight, grad_output])
        return (grad_x, grad_w)


class ColumnParallelLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, num_devices: int):
        super().__init__(in_features, out_features, bias=False)
        self.num_devices = num_devices

    @classmethod
    def distribute(cls, linear: torch.nn.Linear, num_devices: int):
        distributed_linear = cls(linear.in_features, linear.out_features, num_devices)
        mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
        distributed_linear.weight = torch.nn.Parameter(
            dist.tensor.distribute_tensor(linear.weight, mesh, [Shard(0)])
        )
        return distributed_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mesh = dist.device_mesh.init_device_mesh("cuda", [self.num_devices])
        input_dtensor = DTensor.from_local(input, mesh, [Replicate()])
        output_dtensor = LinearFunction.apply(input_dtensor, self.weight)
        return output_dtensor.to_local()


class RowParallelLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, num_devices: int):
        super().__init__(in_features, out_features, bias=False)
        self.num_devices = num_devices

    @classmethod
    def distribute(cls, linear: torch.nn.Linear, num_devices: int):
        distributed_linear = cls(linear.in_features, linear.out_features, num_devices)
        mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
        distributed_linear.weight = torch.nn.Parameter(
            dist.tensor.distribute_tensor(linear.weight, mesh, [Shard(-1)])
        )
        return distributed_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mesh = dist.device_mesh.init_device_mesh("cuda", [self.num_devices])
        input_dtensor = DTensor.from_local(input, mesh, [Shard(-1)])
        output_dtensor = LinearFunction.apply(input_dtensor, self.weight)
        return output_dtensor.to_local()


# class DistributedLinear(torch.nn.Linear):
#     def __init__(self, in_features: int, out_features: int, num_devices: int):
#         super().__init__(in_features, out_features, bias=False)
#         self.num_devices = num_devices

#     def __repr__(self):
#         base_repr = super().__repr__()
#         return f"{base_repr[:-1]}, weight_sharding={self.weight.data.placements})"

#     @classmethod
#     def distribute(cls, linear: torch.nn.Linear, num_devices: int, shard_dim: int):
#         distributed_linear = cls(linear.in_features, linear.out_features, num_devices)
#         mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
#         distributed_linear.weight = torch.nn.Parameter(
#             dist.tensor.distribute_tensor(linear.weight, mesh, [Shard(shard_dim)])
#         )
#         return distributed_linear

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         input_dtensor = DTensor.from_local(input, self.weight.device_mesh)
#         return LinearFunction.apply(input, self.weight)
