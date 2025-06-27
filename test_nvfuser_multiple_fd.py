# torchrun --local-ranks-filter=0 --nnodes 1 --nproc-per-node 2 test_nvfuser_multiple_fd.py
import torch.nn as nn
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
import os
from thunder.dynamo import thunderfx
import thunder
import torch.distributed as dist

import nvfuser
import torch.distributed as dist
from collections.abc import Iterable
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from typing import Callable, cast, TypeAlias

import torch
from dataclasses import dataclass
from functools import partial
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor
from functools import lru_cache
from enum import auto, Enum
import uuid


DTensorsKey: TypeAlias = tuple[tuple[str, str], ...]


def make_key_from_dtensors(dtensors: Iterable[DTensor]) -> DTensorsKey:
    key = tuple((repr(dt.device_mesh), repr(dt.placements)) for dt in dtensors)
    return key


LOCAL_RANK = int(os.environ["LOCAL_RANK"])

device = torch.device("cuda", LOCAL_RANK)
torch.cuda.set_device(device)
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

hidden_size = 16


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable[[nvfuser.FusionDefinition], None]):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        # The complete FusionDefinition (w/ multidevice_scheduler) will have to
        # be created at "call" time according to the input DTensors.
        self._define_fusion = define_fusion

        # In theory, a FusionDefinitionWrapper can own multiple
        # `FusionDefinition`s, because different shardings lead to different
        # `multidevice_schedule`s. In practice, this would trigger #4507.
        self._fusion_definition_cache: dict[DTensorsKey, nvfuser.FusionDefinition] = {}

        self.uuid = uuid.UUID(int=1234, version=4)

    def _create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> nvfuser.FusionDefinition:
        define_fn = self._define_fusion
        print("_create_fusion_definition")

        class Model(nvfuser.FusionDefinition):
            def definition(self) -> None:
                define_fn(self)

            def _find_tensor_by_index(self, index: int) -> nvfuser.Tensor:
                all_tensors = self.sched.tensors()
                print("_find_tensor_by_index", len(all_tensors))
                for t in self.sched.tensors():
                    if t.index == index:
                        return t
                return None

            def multidevice_schedule(self) -> None:
                print("multidevice-schedule")
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
        key = make_key_from_dtensors(in_dtensors)
        print("found", key in self._fusion_definition_cache)
        return self._fusion_definition_cache.setdefault(
            key, (lambda: self._create_fusion_definition(in_dtensors))()
        )

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        fusion_def = self._get_or_create_fusion_definition(in_dtensors)
        print(fusion_def.id())

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


weight = distribute_tensor(
    torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.bfloat16),
    mesh,
    [
        Shard(0),
    ],
)
in_dtensor = distribute_tensor(
    torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.bfloat16),
    mesh,
    [
        Shard(0),
    ],
)


def define_mul_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    out = fd.ops.mul(inp, weight)
    fd.add_output(out)


def define_add_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    out = fd.ops.add(inp, weight)
    fd.add_output(out)


class ComputeType(Enum):
    MUL = auto()
    ADD = auto()


@lru_cache
def get_fusion_definition_wrapper(compute_type: ComputeType) -> FusionDefinitionWrapper:
    match compute_type:
        case ComputeType.MUL:
            fn = define_mul_forward
        case ComputeType.ADD:
            fn = define_add_forward

    return FusionDefinitionWrapper(fn)


fd = get_fusion_definition_wrapper(ComputeType.MUL)
print(fd.uuid)
result = fd([in_dtensor, weight])

fd = get_fusion_definition_wrapper(ComputeType.MUL)
print(fd.uuid)
result = fd([in_dtensor, weight])

# Use input we same global shape but different placements.
weight = distribute_tensor(
    torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.bfloat16),
    mesh,
    [
        Shard(1),
    ],
)
in_dtensor = distribute_tensor(
    torch.randn(hidden_size, hidden_size, requires_grad=True, dtype=torch.bfloat16),
    mesh,
    [
        Shard(0),
    ],
)

# [rank0]:   File "/opt/pytorch/lightning-thunder/test_nvfuser_multiple_fd.py", line 66, in _find_tensor_by_index
# [rank0]:     for t in self.sched.tensors():
# [rank0]:              ^^^^^^^^^^^^^^^^^^^^
# [rank0]: IndexError: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
fd = get_fusion_definition_wrapper(ComputeType.MUL)
print(fd.uuid)
result = fd([in_dtensor, weight])
