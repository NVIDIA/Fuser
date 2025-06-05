# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import nvfuser
from nvfuser import DataType, FusionDefinition


@pytest.mark.mpi
def test_allgather(multidevice_test):
    d = multidevice_test.size
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
    sharded = multidevice_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    (output,), _ = fd.execute([sharded])
    torch.testing.assert_close(output.cpu(), unsharded)


@pytest.mark.mpi
def test_allreduce(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                (-1, -1, -1), contiguity=True, dtype=DataType.Float
            )
            self.out = self.ops.sum(self.inp, [1])
            self.add_output(self.out)

        def multidevice_schedule(self):
            for tv in [self.inp, self.out]:
                self.sched._set_device_mesh(tv, mesh)
                self.sched.split(tv, 1, d, False)
                self.sched.parallelize(tv, 1, nvfuser.ParallelType.mesh_x)
                self.sched.set_allocation_as_loop(tv)

    m = 2
    k = d * 3
    n = 5
    unsharded = torch.randn(m, k, n)
    sharded = multidevice_test.shard_tensor(unsharded, 1, mesh)

    fd = Model()
    outputs, _ = fd.execute([sharded])
    torch.testing.assert_close(outputs[0].cpu(), unsharded.sum(1))


@pytest.mark.mpi
def test_reduce_scatter(multidevice_test):
    d = multidevice_test.size
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
    sharded = multidevice_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    (output,), _ = fd.execute([sharded])
    torch.testing.assert_close(
        output, multidevice_test.shard_tensor(unsharded.sum(0), 0, mesh)
    )


@pytest.mark.mpi
def test_reduce_scatter_noncontiguous(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                (d, 3, d * 4), contiguity=True, dtype=DataType.Float
            )
            self.out = self.ops.sum(self.inp, [0])
            self.add_output(self.out)

        def multidevice_schedule(self):
            self.sched._set_device_mesh(self.inp, mesh)
            self.sched._set_device_mesh(self.out, mesh)

            # inp: [iDID{d}, i{3}, i{d*4}]
            # out: [r{d}, i{3}, i{d*4}]
            #                   /    \
            #               iDID{d}  i{4}
            #
            # Unlike test_reduce_scatter, this leads to extra data copy because
            # the scattered axis is not outermost in allocation.
            # ProcessGroupNCCL::reduce_scatter was able to handle
            # non-contiguous scattering in a functional but suboptimal way.
            self.sched.parallelize(self.inp, 0, nvfuser.ParallelType.mesh_x)

            self.sched.split(self.out, -1, d, False)
            self.sched.parallelize(self.out, -2, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.out)

    unsharded = torch.randn(d, 3, d * 4)
    sharded = multidevice_test.shard_tensor(unsharded, 0, mesh)

    fd = Model()
    (output,), _ = fd.execute([sharded])
    torch.testing.assert_close(
        output, multidevice_test.shard_tensor(unsharded.sum(0), 1, mesh)
    )
