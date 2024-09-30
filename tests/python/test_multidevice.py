# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import multidevice
import nvfuser
from nvfuser import DataType, FusionDefinition


multidevice_test = multidevice.multidevice_test


@pytest.mark.mpi
def test_sizes_and_ranks(multidevice_test):
    size, rank, local_size, local_rank = (
        multidevice_test.size,
        multidevice_test.rank,
        multidevice_test.local_size,
        multidevice_test.local_rank,
    )
    assert size > 0
    assert rank >= 0 and rank < size
    assert local_size > 0
    assert local_rank >= 0 and local_rank < local_size


@pytest.mark.mpi
def test_pointwise(multidevice_test):
    num_devices = multidevice_test.size
    rank = multidevice_test.rank

    torch.cuda.set_device(multidevice_test.local_rank)

    # Just so each rank receives the same unsharded input.
    torch.manual_seed(0)
    unsharded_input = torch.randn(num_devices, 4, device="cuda")
    sharded_input = unsharded_input[rank : rank + 1]

    class Model(FusionDefinition):
        def definition(self):
            self.t0 = self.define_tensor(
                (-1, -1), contiguity=(False, False), dtype=DataType.Float
            )
            self.t1 = self.ops.relu(self.t0)
            self.t2 = self.ops.add(self.t1, self.t1)
            self.add_output(self.t2)

        def multidevice_schedule(self):
            mesh = self.sched._create_device_mesh(range(num_devices))
            self.sched._set_device_mesh(self.t0, mesh)
            self.sched._set_device_mesh(self.t1, mesh)
            self.sched._set_device_mesh(self.t2, mesh)
            self.sched.parallelize(self.t0, 0, nvfuser.ParallelType.mesh_x)

    fn = Model()
    outputs = fn.execute([sharded_input])
    torch.testing.assert_close(outputs[0], unsharded_input.relu() * 2)
