# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass
from functools import partial
from fusion_definition_wrapper import FusionDefinitionWrapper
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor


@dataclass
class LinearConfig:
    in_features: int
    out_features: int


# I omitted biases because DeepSeek V3 uses non-biased linear layers in MLA and
# MoE.
def define_linear_forward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    inp = fd.define_tensor([-1, -1, e_in], contiguity=True, dtype=DataType.BFloat16)
    weight = fd.define_tensor([e_out, e_in], contiguity=True, dtype=DataType.BFloat16)
    out = fd.ops.linear(inp, weight)
    fd.add_output(out)


def define_linear_backward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    x = fd.define_tensor([-1, -1, e_in], contiguity=True, dtype=DataType.BFloat16)
    w = fd.define_tensor([e_out, e_in], contiguity=True, dtype=DataType.BFloat16)
    grad = fd.define_tensor([-1, -1, e_out], contiguity=True, dtype=DataType.BFloat16)

    grad_x = fd.ops.matmul(grad, w)

    grad_flat_t = fd.ops.permute(fd.ops.reshape(grad, [-1, e_out]), [1, 0])
    x_flat = fd.ops.reshape(x, [-1, e_in])
    grad_w = fd.ops.matmul(grad_flat_t, x_flat)

    fd.add_output(grad_x)
    fd.add_output(grad_w)


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: DTensor,
        weight: DTensor,
    ):
        op = FusionDefinitionWrapper(
            partial(define_linear_forward, LinearConfig(weight.size(1), weight.size(0)))
        )
        (output,) = op([input, weight])
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        input, weight = ctx.saved_tensors

        op = FusionDefinitionWrapper(
            partial(
                define_linear_backward, LinearConfig(weight.size(1), weight.size(0))
            )
        )
        grad_x, grad_w = op([input, weight, grad_output])
        return (grad_x, grad_w)
