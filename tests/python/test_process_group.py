import os
import pytest

import torch.distributed as dist
from mpi4py import MPI


class MultideviceTest:
    def __init__(self):
        comm = MPI.COMM_WORLD
        self._size = comm.size
        self._rank = comm.rank
        self._local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        self._local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        print(f"[Rank {self._rank}] MultideviceTest.__init__")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self._size,
            rank=self._rank,
        )
        dist.barrier()

    def __del__(self):
        dist.barrier()
        print(f"[Rank {self._rank}] MultideviceTest.__del__")
        dist.destroy_process_group()

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


@pytest.fixture(scope="session")
def multidevice_test():
    return MultideviceTest()


@pytest.mark.mpi
def test_foo(multidevice_test):
    group = dist.new_group()
    dist.destroy_process_group(group=group)


@pytest.mark.mpi
def test_bar(multidevice_test):
    group = dist.new_group()
    dist.destroy_process_group(group=group)
