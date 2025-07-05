# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run command:
# mpirun -np 1 pytest tests/python/multidevice/test_dtensor.py --only-mpi -s

import pytest
import torch
import torch.distributed as dist
from collections.abc import Iterable
from typing import cast

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

import nvfuser_direct as nvfd
from nvfuser_direct import FusionDefinition, DataType
from linear_direct import LinearFunction


@pytest.mark.mpi
def test_plus_one(setup_default_process_group, multidevice_direct_test):
    def define_fusion(fd: FusionDefinition):
        inp = fd.define_tensor((-1, -1), contiguity=False, dtype=DataType.Float)
        one = fd.define_scalar(1.0, dtype=DataType.Float)
        out = fd.ops.add(inp, one)
        fd.add_output(out)

    def multidevice_schedule(
        fd: FusionDefinition, in_dtensors: Iterable[DTensor]
    ) -> None:
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
                in_tv.set_allocation_domain(
                    in_tv.get_loop_domain(), new_contiguity=True
                )

    num_devices = dist.get_world_size()

    in_tensor = torch.randn(num_devices, 4)
    mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
    in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

    with FusionDefinition() as op:
        define_fusion(op)
        multidevice_schedule(op, [in_dtensor])

    (out_dtensor,) = nvfd.dtensor_execute(op, [in_dtensor])
    torch.testing.assert_close(out_dtensor.to_local(), in_dtensor.to_local() + 1)
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements


@pytest.mark.mpi
def test_column_parallel_linear(setup_default_process_group, multidevice_direct_test):
    d, b, s, e = dist.get_world_size(), 2, 3, 5

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randint(
        -4, 4, (b, s, e), dtype=torch.bfloat16, requires_grad=True
    )
    weight_tensor = torch.randint(
        -4, 4, (d * e, e), dtype=torch.bfloat16, requires_grad=True
    )

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Replicate()])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(0)])

    def assert_close(expected_tensor: torch.Tensor, dtensor: DTensor):
        torch.testing.assert_close(expected_tensor, dtensor.to_local().cpu())

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
def test_row_parallel_linear(setup_default_process_group, multidevice_direct_test):
    d, b, s, e = dist.get_world_size(), 2, 3, 5

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randint(
        -4, 4, (b, s, d * e), dtype=torch.bfloat16, requires_grad=True
    )
    weight_tensor = torch.randint(
        -4, 4, (e, d * e), dtype=torch.bfloat16, requires_grad=True
    )

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Shard(-1)])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(-1)])

    def assert_close(expected_tensor: torch.Tensor, dtensor: DTensor):
        torch.testing.assert_close(expected_tensor, dtensor.to_local().cpu())

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
