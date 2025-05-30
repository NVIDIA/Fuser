# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import nvfuser
import os
import pytest
import torch
import torch.distributed as dist


class MultideviceTest:
    def __init__(self):
        self._communicator = nvfuser.Communicator.instance()

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

    def shard_tensor(
        self, t: torch.Tensor, dim: int, mesh: nvfuser.DeviceMesh
    ) -> torch.Tensor:
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )
        return mesh.shard_tensor(t, dim, self.rank).cuda(self.rank)


@pytest.fixture
def multidevice_test():
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

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

    fixture = MultideviceTest()
    yield fixture
    # Sync all ranks after each test for isolation.
    fixture.communicator.barrier()


# Set up the default process group for torch APIs like
# dist.device_mesh.init_device_mesh.
#
# This fixture is used by multi-GPU tests that use torch.distributed.
#
# I use "session" instead of "module" because
# https://github.com/pytorch/pytorch/issues/119196 reported race conditions
# when reinitializing process groups.
@pytest.fixture(scope="session")
def setup_default_process_group():
    communicator = nvfuser.Communicator.instance()

    local_rank = communicator.local_rank()
    torch.cuda.set_device(local_rank)

    # The default port as used by https://github.com/pytorch/pytorch/blob/45a8b5682eb69d865cbf68c7f2f689b56b4efd53/torch/csrc/distributed/c10d/TCPStore.hpp#L51.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29500",
        world_size=communicator.size(),
        rank=communicator.rank(),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    yield
    dist.destroy_process_group()
