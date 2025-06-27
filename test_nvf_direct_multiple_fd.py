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

from nvfuser_direct import FusionDefinition, DataType

hidden_size = 16


def define_add_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    out = fd.ops.add(inp, weight)
    fd.add_output(out)


def define_mul_forward(fd: FusionDefinition) -> None:
    inp = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    weight = fd.define_tensor([hidden_size, hidden_size], dtype=DataType.BFloat16)
    out = fd.ops.add(inp, weight)
    fd.add_output(out)


with FusionDefinition() as fd:
    define_mul_forward(fd)

weight = torch.randn(
    hidden_size, hidden_size, requires_grad=True, device="cuda", dtype=torch.bfloat16
)
in_dtensor = torch.randn(
    hidden_size, hidden_size, requires_grad=True, device="cuda", dtype=torch.bfloat16
)
outputs = fd.execute([weight, in_dtensor])
print(outputs)
