# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch

from mpi4py import MPI

import nvfuser
from nvfuser import DataType, FusionDefinition


class MultideviceTest:
    def __init__(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self._size = comm.size
        self._rank = comm.rank
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    @property
    def local_size(self):
        return self._local_size

    @property
    def local_rank(self):
        return self._local_rank


@pytest.fixture
def multidevice_test():
    return MultideviceTest()


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


@pytest.mark.skip(reason="#2929")
@pytest.mark.mpi
def test_pointwise(multidevice_test):
    num_devices = multidevice_test.size
    rank = multidevice_test.rank

    torch.cuda.set_device(multidevice_test.local_rank)

    # Inputs
    inputs = [
        torch.randn(num_devices, 4, device="cuda")[rank : rank + 1, ...],
    ]

    # dynamic shape isn't supported;
    # scalar inputs isn't supported;
    class MultiDeviceModel(FusionDefinition):
        def definition(self):
            self.t0 = self.define_tensor((2, 4), (False, False), dtype=DataType.Float)
            self.t1 = self.ops.relu(self.t0)
            self.t2 = self.ops.add(self.t1, self.t1)
            self.add_output(self.t2)

        def schedule(self):
            mesh = self.sched._create_device_mesh((0, 1))
            self.sched._set_device_mesh(self.t0, mesh)
            self.sched._set_device_mesh(self.t1, mesh)
            self.sched._set_device_mesh(self.t2, mesh)
            self.sched.parallelize(self.t0, 0, nvfuser.ParallelType.mesh_x)

    fn = MultiDeviceModel()

    outputs = fn.execute(inputs)

    for i in range(3):
        outputs = fn.execute(inputs)

    assert (inputs[0].relu() * 2).allclose(outputs[0][rank])
