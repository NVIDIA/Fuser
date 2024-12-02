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
    num_devices = mpi_test.size
    rank = mpi_test.rank

    unsharded = torch.randn(num_devices * 4)
    sharded = mpi_test.shard_tensor(unsharded, 0)

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                (num_devices * 4,), contiguity=True, dtype=DataType.Float
            )
            self.out = self.ops.set(self.inp)
            self.add_output(self.out)

        def multidevice_schedule(self):
            mesh = self.sched._create_device_mesh(range(num_devices))
            self.sched._set_device_mesh(self.inp, mesh)
            self.sched._set_device_mesh(self.out, mesh)

            self.sched.split(self.inp, 0, num_devices, False)
            self.sched.parallelize(self.inp, 0, nvfuser.ParallelType.mesh_x)
            self.sched.set_allocation_as_loop(self.inp)

            self.sched.split(self.out, 0, num_devices, False)
            self.sched.set_allocation_as_loop(self.out)

    fd = Model()
    outputs = fd.execute([sharded])
    torch.testing.assert_close(outputs[0].cpu(), unsharded)
