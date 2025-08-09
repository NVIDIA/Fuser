# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import nvfuser_direct as nvfuser
from nvfuser_direct import FusionDefinition
from collections.abc import Iterable
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard
from typing import Callable, cast, TypeAlias


DTensorsKey: TypeAlias = tuple[tuple[str, str], ...]


def make_key_from_dtensors(dtensors: Iterable[DTensor]) -> DTensorsKey:
    key = tuple((repr(dt.device_mesh), repr(dt.placements)) for dt in dtensors)
    return key


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        # The complete FusionDefinition (w/ multidevice_scheduler) will have to
        # be created at "call" time according to the input DTensors.
        self._define_fusion = define_fusion

        # TODO: In theory, a FusionDefinitionWrapper can own multiple
        # `FusionDefinition`s, because different shardings lead to different
        # `multidevice_schedule`s. Currently, cache FusionDefinition based on input DTensors.
        self._fusion_definition_cache: dict[DTensorsKey, nvfuser.FusionDefinition] = {}

    def _create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> nvfuser.FusionDefinition:
        with FusionDefinition() as fd:
            self._define_fusion(fd)
            self._multidevice_schedule(fd, in_dtensors)
        return fd

    def _multidevice_schedule(
        self, fd: FusionDefinition, in_dtensors: Iterable[DTensor]
    ) -> None:
        for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
            print(in_dtensor.device_mesh.mesh)
            mesh = nvfuser.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh)

            in_tv.set_device_mesh(mesh)

            assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"

            # Split and parallelize.
            # When the mesh is multi-dimensional, iterate through the
            # placements in descending order of Placement.dim.
            placement: Placement = in_dtensor.placements[0]
            if placement.is_shard():
                dim = cast(Shard, placement).dim
                in_tv.split(dim, mesh.size, inner_split=False)
                in_tv.axis(dim).parallelize(nvfuser.ParallelType.mesh_x)

    def _get_or_create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> nvfuser.FusionDefinition:
        key = make_key_from_dtensors(in_dtensors)
        return self._fusion_definition_cache.setdefault(
            key, (lambda: self._create_fusion_definition(in_dtensors))()
        )

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        fusion_def = self._get_or_create_fusion_definition(in_dtensors)
        return nvfuser.execute_with_dtensors(fusion_def, in_dtensors)
