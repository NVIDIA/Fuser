# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import nvfuser
import pytest
import torch


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
