# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest

from mpi4py import MPI


class MpiTest:
    def __init__(self):
        self._communicator = MPI.COMM_WORLD
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

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


@pytest.fixture(scope="session")
def mpi_test():
    fixture = MpiTest()
    yield fixture
    # Sync all ranks after each test for isolation.
    fixture.barrier()
