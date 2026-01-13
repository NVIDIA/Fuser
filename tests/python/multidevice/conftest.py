# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run command:
# mpirun -np [num_devices] pytest tests/python/multidevice/[test_name].py --only-mpi -s

import os
import pytest
from typing import Iterable

import torch
import torch.distributed as dist

import nvfuser_direct as nvfuser


class MultideviceTest:
    def __init__(self):
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

        self._communicator = nvfuser.multidevice.Communicator.instance()

        # This way, when individual tests create unsharded input, each rank
        # receives the same data.
        torch.manual_seed(0)

    @property
    def communicator(self):
        return self._communicator

    @property
    def size(self):
        return self._communicator.size()

    @property
    def rank(self):
        return self._communicator.rank()

    @property
    def local_size(self):
        return self._communicator.local_size()

    @property
    def local_rank(self):
        return self._communicator.local_rank()

    def shard_tensor_1d(self, t: torch.Tensor, dim: int, mesh) -> torch.Tensor:
        """Shard tensor along a single dimension (1D sharding only).

        Args:
            t: Tensor to shard (preferably on CPU for memory efficiency)
            dim: Dimension to shard along
            mesh: DeviceMesh to use for sharding

        Returns:
            Sharded tensor on current GPU device

        Example:
            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(num_devices))
            unsharded = torch.randn(num_devices, 4)
            sharded = self.shard_tensor_1d(unsharded, 0, mesh)
        """
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )
        return mesh.shard_tensor(t, dim).cuda(self.local_rank)

    def shard_tensor(self, t: torch.Tensor, tv) -> torch.Tensor:
        """Shard tensor using TensorView's parallelization and device mesh.

        Args:
            t: Tensor to shard (preferably on CPU for memory efficiency)
            tv: TensorView with device mesh and parallelization information

        Returns:
            Sharded tensor on current GPU device

        Example:
            with nvfuser.FusionDefinition() as fd:
                inp_tv = fd.define_tensor([-1, -1])
                inp_tv.set_device_mesh(mesh)
                inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
                # ... rest of fusion

            unsharded = torch.randn(num_devices, 4)
            sharded = self.shard_tensor(unsharded, inp_tv)
        """
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )

        sharded = nvfuser.multidevice.shard_tensor(t, tv)
        return sharded.cuda(self.local_rank)


@pytest.fixture
def multidevice_test():
    fixture = MultideviceTest()
    yield fixture
    fixture.communicator.barrier()


def get_env(envs: Iterable[str], /, *, default: str) -> str:
    for env in envs:
        if value := os.environ.get(env):
            return value
    return default


# Set up the default process group for torch APIs like
# dist.device_mesh.init_device_mesh.
#
# This fixture is used by multi-GPU tests that use torch.distributed directly.
#
# I use "session" instead of "module" because
# https://github.com/pytorch/pytorch/issues/119196 reported race conditions
# when reinitializing process groups.
@pytest.fixture(scope="session")
def setup_default_process_group():
    # I avoided using nvfuser.Communicator to minimize fixture dependencies on
    # nvFuser. This makes the transition from legacy bindings to direct
    # bindings easier.
    rank = int(get_env(["OMPI_COMM_WORLD_RANK", "RANK"], default="0"))
    world_size = int(get_env(["OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"], default="1"))
    local_rank = int(get_env(["OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"], default="0"))

    torch.cuda.set_device(local_rank)

    # The default port as used by https://github.com/pytorch/pytorch/blob/45a8b5682eb69d865cbf68c7f2f689b56b4efd53/torch/csrc/distributed/c10d/TCPStore.hpp#L51.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29500",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    yield
    dist.destroy_process_group()
