# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import multidevice_fixtures
import nvfuser
import pytest
import torch
import torch.distributed as dist
from collections.abc import Iterable
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from typing import Callable, cast


multidevice_test = multidevice_fixtures.multidevice_test


@pytest.fixture(scope="module")
def setup_process_group(multidevice_test):
    # The default port as used by https://github.com/pytorch/pytorch/blob/45a8b5682eb69d865cbf68c7f2f689b56b4efd53/torch/csrc/distributed/c10d/TCPStore.hpp#L51.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29500",
        world_size=multidevice_test.size,
        rank=multidevice_test.rank,
    )
    yield
    dist.destroy_process_group()


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable[[FusionDefinition], None]):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        self._define_fusion = define_fusion

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        define_fn = self._define_fusion

        class Model(FusionDefinition):
            def definition(self):
                define_fn(self)

            def _find_tensor_by_index(self, index: int) -> nvfuser.Tensor:
                for t in self.sched.tensors():
                    if t.index == index:
                        return t
                return None

            def multidevice_schedule(self):
                for in_tensor_index, in_dtensor in zip(self.inputs(), in_dtensors):
                    in_tensor = self._find_tensor_by_index(in_tensor_index)

                    # Set the device mesh.
                    assert (
                        in_dtensor.device_mesh.ndim == 1
                    ), "nvFuser's Python API only supports 1D meshes."
                    mesh = nvfuser.DeviceMesh(
                        in_dtensor.device_mesh.mesh.view(-1).tolist()
                    )
                    self.sched._set_device_mesh(in_tensor, mesh)

                    # Parallelize.
                    assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"
                    placement: Placement = in_dtensor.placements[0]
                    if placement.is_shard():
                        dim = cast(Shard, placement).dim
                        self.sched.parallelize(
                            in_tensor, dim, nvfuser.ParallelType.mesh_x
                        )

        in_tensors = [in_dtensor.to_local() for in_dtensor in in_dtensors]
        model = Model()
        out_tensors = model.execute(in_tensors)

        for i, out_tensor in enumerate(out_tensors):
            if isinstance(out_tensor, nvfuser.DistributedTensor):
                mesh = dist.device_mesh.init_device_mesh("cuda", [out_tensor.mesh.size])
                placements: list[Placement] = []
                for parallel_type in [nvfuser.ParallelType.mesh_x]:
                    axis: int = out_tensor.axis_sharded_on(parallel_type)
                    placements.append(Replicate() if axis == -1 else Shard(axis))
                out_tensors[i] = DTensor.from_local(out_tensor.local, mesh, placements)
        return out_tensors


@pytest.mark.mpi
def test_plus_one(setup_process_group):
    def define_fusion(fd: FusionDefinition):
        inp = fd.define_tensor(
            (-1, -1), contiguity=(False, False), dtype=DataType.Float
        )
        one = fd.define_scalar(1.0, dtype=DataType.Float)
        out = fd.ops.add(inp, one)
        fd.add_output(out)

    op = FusionDefinitionWrapper(define_fusion)

    num_devices = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    in_tensor = torch.randn(num_devices, 4)
    mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
    in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

    (out_dtensor,) = op([in_dtensor])
    torch.testing.assert_close(out_dtensor.to_local(), in_dtensor.to_local() + 1)
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements
