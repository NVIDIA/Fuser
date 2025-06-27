# torchrun --local-ranks-filter=0 --nnodes 1 --nproc-per-node 2 test_nvf_direct_multiple_fd.py

# TODO: Disabled for AssertionError: Cannot import nvfuser_direct if nvfuser module is already imported.
# import thunder
# from thunder.dynamo import thunderfx

import os
from collections.abc import Iterable
from typing import cast

import torch
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor

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

with FusionDefinition() as fd1:
    define_mul_forward(fd1)
    multidevice_schedule(fd1, in_dtensors)

outputs = fd1.multigpu_execute(in_dtensors)
print(outputs)

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
in_dtensors = [weight, in_dtensor]

with FusionDefinition() as fd2:
    define_mul_forward(fd2)
    multidevice_schedule(fd2, in_dtensors)

outputs = fd2.multigpu_execute(in_dtensors)
print(outputs)

# [rank0]: RuntimeError:  INTERNAL ASSERT FAILED at "/opt/pytorch/nvfuser/csrc/host_ir/lower_to_communication.cpp":311, please report a bug with repro script to NVFuser at https://github.com/NVIDIA/Fuser/issues. getCommunicationInfo should only be called when `e` is known to be a communication. So `e` should be either a LoadStoreOp or a ReductionOp. Given: T4_g_float[ideviceIdx.x18{2}, iS19{8}, iS9{16}] (DeviceMesh{0 1})
# [rank0]:    = T2_l_float[ideviceIdx.x14{2}, iS15{8}, iS5{16}] (DeviceMesh{0 1})
# [rank0]:    * T5_l_float[iS20{16}, iS21{16}] (DeviceMesh{0 1});
