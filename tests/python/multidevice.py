# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from mpi4py import MPI


class MultideviceTest:
    def __init__(self):
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
