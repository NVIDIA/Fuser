# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import mpi_fixtures
import nvfuser
from nvfuser import DataType, FusionDefinition


mpi_test = mpi_fixtures.mpi_test


@pytest.mark.mpi
def test_allgather(mpi_test):
    d = mpi_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                (d * 4,), contiguity=True, dtype=DataType.Float
            )
            self.out = self.ops.set(self.inp)
            self.add_output(self.out)

        def multidevice_schedule(self):
            self.sched._set_device_mesh(self.inp, mesh)
            self.sched._set_device_mesh(self.out, mesh)

            self.sched.split(self.inp, 0, d, False)
            self.sched.parallelize(self.inp, 0, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.inp)

            self.sched.split(self.out, 0, d, False)
            self.sched.set_allocation_as_loop(self.out)

    unsharded = torch.randn(d * 4)
    sharded = mpi_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    outputs = fd.execute([sharded])
    torch.testing.assert_close(outputs[0].cpu(), unsharded)


@pytest.mark.mpi
def test_allreduce(mpi_test):
    d = mpi_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor((d, 4), contiguity=True, dtype=DataType.Float)
            self.out = self.ops.sum(self.inp, [0])
            self.add_output(self.out)

        def multidevice_schedule(self):
            self.sched._set_device_mesh(self.inp, mesh)
            self.sched._set_device_mesh(self.out, mesh)

            self.sched.parallelize(self.inp, 0, nvfuser.ParallelType.mesh_x)

    unsharded = torch.randn(d, 4)
    sharded = mpi_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    outputs = fd.execute([sharded])
    torch.testing.assert_close(outputs[0].cpu(), unsharded.sum(0))


@pytest.mark.mpi
def test_reduce_scatter(mpi_test):
    d = mpi_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                (d, d * 4), contiguity=True, dtype=DataType.Float
            )
            self.out = self.ops.sum(self.inp, [0])
            self.add_output(self.out)

        def multidevice_schedule(self):
            self.sched._set_device_mesh(self.inp, mesh)
            self.sched._set_device_mesh(self.out, mesh)

            self.sched.parallelize(self.inp, 0, nvfuser.ParallelType.mesh_x)

            self.sched.split(self.out, -1, d, False)
            self.sched.parallelize(self.out, -2, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.out)

    unsharded = torch.randn(d, d * 4)
    sharded = mpi_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    outputs = fd.execute([sharded])
    torch.testing.assert_close(
        outputs[0], mpi_test.shard_tensor(unsharded.sum(0), 0, mesh)
    )
