# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run command:
# mpirun -np [num_devices] pytest tests/python/multidevice/[test_name].py --only-mpi -s

import os
from enum import Enum, auto
from typing import Iterable

import sys
import pytest

import torch
import torch.distributed as dist


class Binding(Enum):
    LEGACY = auto()
    DIRECT = auto()


class MultideviceTest:
    def __init__(self, binding: Binding):
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

        if binding == Binding.LEGACY:
            assert (
                "nvfuser_direct" not in sys.modules
            ), "nvfuser_direct is already imported"
            import nvfuser

            # Reset the cache here to work around a bug in FusionDefintion.execute.
            # FusionDefinition._finalize_definition maps the same `definition` to the
            # same FusionSchedules and therefore the same FusionExecutorCache. This was
            # correct until multiple FusionDefinitions started to have the same
            # `definition` but different `multidevice_schedule`s. This seems to be a
            # known issue beacuse a similar workaround for single-GPU schedules is done
            # here:
            # https://github.com/NVIDIA/Fuser/blob/f44f1913c26f8325099ab6fe46d678cbea435658/tests/python/test_schedule_ops.py#L115.
            #
            # I couldn't think of an easy way to fix this issue properly. Also, that
            # FusionCache is obsolete makes me less motivated to do so.
            nvfuser.FusionCache.reset()
            self._communicator = nvfuser.Communicator.instance()
        elif binding == Binding.DIRECT:
            assert "nvfuser" not in sys.modules, "nvfuser is already imported"
            import nvfuser_direct as nvfd

            self._communicator = nvfd.multidevice.Communicator.instance()

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

    def shard_tensor(self, t: torch.Tensor, dim: int, mesh) -> torch.Tensor:
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )
        return mesh.shard_tensor(t, dim, self.rank).cuda(self.rank)


# Existing tests that use legacy python bindings use this.
@pytest.fixture
def multidevice_test():
    fixture = MultideviceTest(Binding.LEGACY)
    yield fixture
    fixture.communicator.barrier()


# Migrated tests to new direct python bindings use this.
@pytest.fixture
def multidevice_direct_test():
    fixture = MultideviceTest(Binding.DIRECT)
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
