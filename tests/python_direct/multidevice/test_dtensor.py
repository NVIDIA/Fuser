# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run command:
# mpirun -np 2 pytest tests/python_direct/multidevice/test_dtensor.py --only-mpi -s

import pytest
from collections.abc import Iterable
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

import nvfuser_direct as nvfd
from nvfuser_direct import FusionDefinition

hidden_size = 16


@pytest.mark.mpi
def test_plus_one(setup_default_process_group, multidevice_test):
    def define_mul_forward(fd: FusionDefinition) -> None:
        inp = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.Float)
        weight = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.Float)
        out = fd.ops.mul(inp, weight)
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
    mesh = DeviceMesh("cuda", list(range(num_devices)))

    weight = distribute_tensor(
        torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.float),
        mesh,
        [
            Shard(0),
        ],
    )
    in_dtensor = distribute_tensor(
        torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.float),
        mesh,
        [
            Shard(0),
        ],
    )
    in_dtensors = [weight, in_dtensor]

    with FusionDefinition() as fd1:
        define_mul_forward(fd1)
        multidevice_schedule(fd1, in_dtensors)

    (out_dtensor,) = fd1.multigpu_execute(in_dtensors)
    torch.testing.assert_close(
        out_dtensor.to_local(), in_dtensor.to_local() * weight.to_local()
    )
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements
