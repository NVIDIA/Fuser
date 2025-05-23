# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import nvfuser
import pytest
import torch
import torch.distributed as dist
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from typing import Callable, cast


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable[[FusionDefinition], None]):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        self._define_fusion = define_fusion

    def _create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> FusionDefinition:
        define_fn = self._define_fusion

        class Model(FusionDefinition):
            def definition(self) -> None:
                define_fn(self)

            def _find_tensor_by_index(self, index: int) -> nvfuser.Tensor:
                for t in self.sched.tensors():
                    if t.index == index:
                        return t
                return None

            def multidevice_schedule(self) -> None:
                for in_tensor_index, in_dtensor in zip(self.inputs(), in_dtensors):
                    in_tensor = self._find_tensor_by_index(in_tensor_index)

                    # Set the device mesh.
                    assert (
                        in_dtensor.device_mesh.ndim == 1
                    ), "nvFuser's Python API only supports 1D meshes."
                    mesh = nvfuser.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

                    self.sched._set_device_mesh(in_tensor, mesh)

                    # Split and parallelize.
                    assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"
                    # When the mesh is multi-dimensional, iterate through the
                    # placements in descending order of Placement.dim.
                    placement: Placement = in_dtensor.placements[0]
                    if placement.is_shard():
                        dim = cast(Shard, placement).dim
                        self.sched.split(in_tensor, dim, mesh.size, False)
                        self.sched.parallelize(
                            in_tensor, dim, nvfuser.ParallelType.mesh_x
                        )
                        self.sched.set_allocation_as_loop(in_tensor)

        return Model()

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        fusion_def = self._create_fusion_definition(in_dtensors)

        in_tensors = [in_dtensor.to_local() for in_dtensor in in_dtensors]
        out_tensors, out_shardings = fusion_def.execute(in_tensors)
        assert len(out_tensors) == len(out_shardings)

        out_dtensors: list[DTensor] = []
        for out_tensor, out_sharding in zip(out_tensors, out_shardings):
            mesh = dist.device_mesh.init_device_mesh("cuda", (out_sharding.mesh.size,))
            placements: list[Placement] = []
            for parallel_type in [nvfuser.ParallelType.mesh_x]:
                axis: int = out_sharding.axis_sharded_on(parallel_type)
                placements.append(Replicate() if axis == -1 else Shard(axis))
            out_dtensors.append(DTensor.from_local(out_tensor, mesh, placements))
        return out_dtensors


@pytest.mark.mpi
def test_plus_one(setup_default_process_group, multidevice_test):
    def define_fusion(fd: FusionDefinition):
        inp = fd.define_tensor((-1, -1), contiguity=False, dtype=DataType.Float)
        one = fd.define_scalar(1.0, dtype=DataType.Float)
        out = fd.ops.add(inp, one)
        fd.add_output(out)

    op = FusionDefinitionWrapper(define_fusion)

    num_devices = dist.get_world_size()

    in_tensor = torch.randn(num_devices, 4)
    mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
    in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

    (out_dtensor,) = op([in_dtensor])
    torch.testing.assert_close(out_dtensor.to_local(), in_dtensor.to_local() + 1)
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements


@dataclass
class LinearConfig:
    in_features: int
    out_features: int


# I omitted biases because DeepSeek V3 uses non-biased linear layers in MLA and
# MoE.
def define_linear_forward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    inp = fd.define_tensor([-1, -1, e_in], contiguity=True)
    weight = fd.define_tensor([e_out, e_in], contiguity=True)
    out = fd.ops.linear(inp, weight)
    fd.add_output(out)


def define_linear_backward(config: LinearConfig, fd: FusionDefinition) -> None:
    e_in, e_out = config.in_features, config.out_features

    x = fd.define_tensor([-1, -1, e_in], contiguity=True)
    w = fd.define_tensor([e_out, e_in], contiguity=True)
    grad = fd.define_tensor([-1, -1, e_out], contiguity=True)

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


@pytest.mark.mpi
def test_column_parallel_linear(setup_default_process_group, multidevice_test):
    d, b, s, e = dist.get_world_size(), 2, 1024, 768

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randn(b, s, e, requires_grad=True)
    weight_tensor = torch.randn(d * e, e, requires_grad=True)

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Replicate()])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(0)])

    def assert_close(expected_tensor, dtensor):
        torch.testing.assert_close(
            expected_tensor, dtensor.to_local().cpu(), rtol=1.3e-6, atol=1e-3
        )

    out_tensor = torch.nn.functional.linear(inp_tensor, weight_tensor)
    out_dtensor = LinearFunction.apply(inp_dtensor, weight_dtensor)
    rank = dist.get_rank()
    assert_close(out_tensor.split(e, dim=-1)[rank], out_dtensor)

    (expected_grad_x, expected_grad_w) = torch.autograd.grad(
        out_tensor,
        (inp_tensor, weight_tensor),
        torch.ones_like(out_tensor),
    )
    (grad_x, grad_w) = torch.autograd.grad(
        out_dtensor,
        (inp_dtensor, weight_dtensor),
        torch.ones_like(out_dtensor),
    )
    assert_close(expected_grad_x, grad_x)
    assert_close(expected_grad_w.split(e, dim=0)[rank], grad_w)


@pytest.mark.mpi
def test_row_parallel_linear(setup_default_process_group, multidevice_test):
    d, b, s, e = dist.get_world_size(), 2, 1024, 768

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randn(b, s, d * e, requires_grad=True)
    weight_tensor = torch.randn(e, d * e, requires_grad=True)

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Shard(-1)])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(-1)])

    def assert_close(expected_tensor, dtensor):
        torch.testing.assert_close(
            expected_tensor, dtensor.to_local().cpu(), rtol=1.3e-6, atol=1e-3
        )

    out_tensor = torch.nn.functional.linear(inp_tensor, weight_tensor)
    out_dtensor = LinearFunction.apply(inp_dtensor, weight_dtensor)
    assert_close(out_tensor, out_dtensor)

    (expected_grad_x, expected_grad_w) = torch.autograd.grad(
        out_tensor,
        (inp_tensor, weight_tensor),
        torch.ones_like(out_tensor),
    )
    (grad_x, grad_w) = torch.autograd.grad(
        out_dtensor,
        (inp_dtensor, weight_dtensor),
        torch.ones_like(out_dtensor),
    )
    rank = dist.get_rank()
    assert_close(expected_grad_x.split(e, dim=-1)[rank], grad_x)
    assert_close(expected_grad_w.split(e, dim=-1)[rank], grad_w)
