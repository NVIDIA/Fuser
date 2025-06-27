import os

# TODO: Disabled for AssertionError: Cannot import nvfuser_direct if nvfuser module is already imported.
# import thunder
# from thunder.dynamo import thunderfx

from collections.abc import Iterable
from typing import Callable, cast, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
import torch.distributed as dist

from dataclasses import dataclass
from functools import partial
from functools import lru_cache
from enum import auto, Enum

import nvfuser_direct as nvfd
from nvfuser_direct import FusionDefinition

hidden_size = 16


def define_add_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.BFloat16)
    out = fd.ops.add(inp, weight)
    fd.add_output(out)


def define_mul_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=nvfd.DataType.BFloat16)
    out = fd.ops.add(inp, weight)
    fd.add_output(out)


def multidevice_schedule(fd: FusionDefinition, in_dtensors: Iterable[DTensor]) -> None:
    for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
        # Set the device mesh.
        assert (
            in_dtensor.device_mesh.ndim == 1
        ), "nvFuser's Python API only supports 1D meshes."
        mesh = nvfd.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

        in_tv.set_device_mesh(mesh)

        assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"

        # Split and parallelize.
        # When the mesh is multi-dimensional, iterate through the
        # placements in descending order of Placement.dim.
        placement: Placement = in_dtensor.placements[0]
        if placement.is_shard():
            dim = cast(Shard, placement).dim
            in_tv.split(dim, mesh.size(), inner_split=False)
            in_tv.axis(dim).parallelize(nvfd.ParallelType.mesh_x)
            in_tv.set_allocation_domain(in_tv.get_loop_domain(), new_contiguity=True)


LOCAL_RANK = int(os.environ["LOCAL_RANK"])

device = torch.device("cuda", LOCAL_RANK)
torch.cuda.set_device(device)
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

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
in_dtensors = [weight, in_dtensor]

with FusionDefinition() as fd:
    define_mul_forward(fd)
    multidevice_schedule(fd, in_dtensors)

outputs = fd.multigpu_execute(in_dtensors)
print(outputs)
