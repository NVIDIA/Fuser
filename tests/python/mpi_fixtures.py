# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import nvfuser

from mpi4py import MPI


class MpiTest:
    def __init__(self):
        self._communicator = MPI.COMM_WORLD
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        # This way, when individual tests create unsharded input, each rank
        # receives the same data.
        torch.manual_seed(0)

    @property
    def size(self):
        return self._communicator.size

    @property
    def rank(self):
        return self._communicator.rank

    @property
    def local_size(self):
        return self._local_size

    @property
    def local_rank(self):
        return self._local_rank

    def barrier(self):
        self._communicator.barrier()

    def shard_tensor(self, t: torch.Tensor, dim: int, mesh: nvfuser.DeviceMesh) -> torch.Tensor:
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )
        return mesh.shard_tensor(t, dim, self.rank).cuda(self.local_rank)


@pytest.fixture(scope="session")
def mpi_test():
    fixture = MpiTest()
    yield fixture
    # Sync all ranks after each test for isolation.
    fixture.barrier()
