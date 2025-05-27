# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import nvfuser
import torch.distributed as dist
from collections.abc import Iterable
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from typing import Callable, cast, Optional


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable[[nvfuser.FusionDefinition], None]):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        # The complete FusionDefinition (w/ multidevice_scheduler) will have to
        # be created at "call" time according to the input DTensors.
        self._define_fusion = define_fusion

        # In theory, a FusionDefinitionWrapper can own multiple
        # `FusionDefinition`s, because different shardings lead to different
        # `multidevice_schedule`s. In pratice, this would trigger #4507 so I
        # chose to let each FusionDefinitionWrapper own only one
        # FusionDefinition.
        self._fusion_definition: Optional[nvfuser.FusionDefinition] = None

    def _create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> nvfuser.FusionDefinition:
        define_fn = self._define_fusion

        class Model(nvfuser.FusionDefinition):
            def definition(self) -> None:
                define_fn(self)

            def _find_tensor_by_index(self, index: int) -> nvfuser.Tensor:
                for t in self.sched.tensors():
                    if t.index == index:
                        return t
                return None

            def multidevice_schedule(self) -> None:
                for in_tensor_index, in_dtensor in zip(self.inputs(), in_dtensors):
                    in_tensor = self._find_tensor_by_index(in_tensor_index)

                    # Set the device mesh.
                    assert (
                        in_dtensor.device_mesh.ndim == 1
                    ), "nvFuser's Python API only supports 1D meshes."
                    mesh = nvfuser.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

                    self.sched._set_device_mesh(in_tensor, mesh)

                    # Split and parallelize.
                    assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"
                    # When the mesh is multi-dimensional, iterate through the
                    # placements in descending order of Placement.dim.
                    placement: Placement = in_dtensor.placements[0]
                    if placement.is_shard():
                        dim = cast(Shard, placement).dim
                        self.sched.split(in_tensor, dim, mesh.size, False)
                        self.sched.parallelize(
                            in_tensor, dim, nvfuser.ParallelType.mesh_x
                        )
                        self.sched.set_allocation_as_loop(in_tensor)

        return Model()

    def _get_or_create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> nvfuser.FusionDefinition:
        if self._fusion_definition is None:
            self._fusion_definition = self._create_fusion_definition(in_dtensors)

        # When self._fusion_definition already exists, we can and should check
        # whether its multidevice_schedule is consistant with how `in_dtensors`
        # are sharded.
        return self._fusion_definition

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        fusion_def = self._get_or_create_fusion_definition(in_dtensors)

        in_tensors = [in_dtensor.to_local() for in_dtensor in in_dtensors]
        out_tensors, out_shardings = fusion_def.execute(in_tensors)
        assert len(out_tensors) == len(out_shardings)

        out_dtensors: list[DTensor] = []
        for out_tensor, out_sharding in zip(out_tensors, out_shardings):
            mesh = dist.device_mesh.init_device_mesh("cuda", (out_sharding.mesh.size,))
            placements: list[Placement] = []
            for parallel_type in [nvfuser.ParallelType.mesh_x]:
                axis: int = out_sharding.axis_sharded_on(parallel_type)
                placements.append(Replicate() if axis == -1 else Shard(axis))
            out_dtensors.append(DTensor.from_local(out_tensor, mesh, placements))
        return out_dtensors
