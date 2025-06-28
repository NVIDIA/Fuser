# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import auto, Enum
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard
from typing import Iterable, cast

import nvfuser_direct as nvfd
from nvfuser_direct import DataType, FusionDefinition


class DTensorList:
    def __init__(self, dtensors: Iterable[DTensor]):
        self._list = dtensors
        self._key = tuple(
            (repr(dt.device_mesh), repr(dt.placements)) for dt in dtensors
        )

    def __hash__(self):
        return hash(self._key)

    def __getitem__(self, index):
        return self._list[index]


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


def multidevice_schedule(fd: FusionDefinition, in_dtensors: Iterable[DTensor]) -> None:
    for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
        # Set the device mesh.
        assert (
            in_dtensor.device_mesh.ndim == 1
        ), "nvFuser's Python API only supports 1D meshes."
        mesh = nvfd.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

        in_tv.set_device_mesh(mesh)

        assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"

        # Split and parallelize.
        # When the mesh is multi-dimensional, iterate through the
        # placements in descending order of Placement.dim.
        placement: Placement = in_dtensor.placements[0]
        if placement.is_shard():
            dim = cast(Shard, placement).dim
            in_tv.split(dim, mesh.size, inner_split=False)
            in_tv.axis(dim).parallelize(nvfd.ParallelType.mesh_x)
            in_tv.set_allocation_domain(in_tv.get_loop_domain(), new_contiguity=True)


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


# Cache so we create only one FusionDefinition
def get_fusion_definition(
    compute_type: ComputeType,
    linear_config: LinearConfig,
    in_dtensors: DTensorList,
) -> FusionDefinition:
    if not hasattr(get_fusion_definition, "cache"):
        get_fusion_definition.cache = {}

    key = (compute_type, linear_config)
    if key in get_fusion_definition.cache:
        return get_fusion_definition.cache[key]

    match compute_type:
        case ComputeType.FORWARD:
            with FusionDefinition() as fwd_fd:
                define_linear_forward(linear_config, fwd_fd)
                multidevice_schedule(fwd_fd, in_dtensors)
            get_fusion_definition.cache[key] = fwd_fd
            return fwd_fd
        case ComputeType.BACKWARD:
            assert False, "Linear Backwards FusionDefinition is not implemented"


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: DTensor,
        weight: DTensor,
    ):
        assert input.dim() in (2, 3)
        in_dtensors = [input, weight]
        op = get_fusion_definition(
            ComputeType.FORWARD,
            LinearConfig(weight.size(1), weight.size(0), has_batch=(input.dim() == 3)),
            DTensorList(in_dtensors),
        )
        (output,) = op.multigpu_execute(in_dtensors)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        assert False, "Linear backward function is not implemented."


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
