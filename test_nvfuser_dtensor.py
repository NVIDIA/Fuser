# torchrun --local-ranks-filter=0 --nproc_per_node=2 test_nvfuser_dtensor.py
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard, Replicate
from nvfuser_direct import FusionDefinition, DataType
import nvfuser_direct as nvfd
from typing import cast
from torch.distributed.tensor.placement_types import Placement
import torch

# Initialize process group
torch.distributed.init_process_group(backend="nccl")

# Get local rank and world size
local_rank = int(torch.distributed.get_rank())
world_size = torch.distributed.get_world_size()

# Set device
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)

K = 4


def define_fusion(fd):
    tv0 = fd.define_tensor(
        shape=[K, K], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False
    )
    tv1 = fd.define_tensor(
        shape=[K, K], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False
    )
    tv3 = fd.ops.add(tv0, tv1)
    fd.add_output(tv3)


def multidevice_schedule(fd: FusionDefinition, in_dtensors) -> None:
    for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
        # Set the device mesh.
        assert (
            in_dtensor.device_mesh.ndim == 1
        ), "nvFuser's Python API only supports 1D meshes."
        mesh = nvfd.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

        in_tv.set_device_mesh(mesh)

        assert (
            len(in_dtensor.placements) == 1
        ), "nvFuser's Python API only supports 1D meshes."

        # Split and parallelize.
        # When the mesh is multi-dimensional, iterate through the
        # placements in descending order of Placement.dim.
        placement: Placement = in_dtensor.placements[0]
        if placement.is_shard():
            dim = cast(Shard, placement).dim
            in_tv.split(dim, mesh.size, inner_split=False)
            in_tv.axis(dim).parallelize(nvfd.ParallelType.mesh_x)
            in_tv.set_allocation_domain(in_tv.get_loop_domain(), new_contiguity=True)


def test_dtensor():
    world_size = torch.distributed.get_world_size()

    num_devices = world_size
    mesh = DeviceMesh("cuda", list(range(num_devices)))

    a_dtensor = distribute_tensor(
        torch.randn(K, K, requires_grad=False, dtype=torch.bfloat16),
        mesh,
        [Replicate()],
    )
    b_dtensor = distribute_tensor(
        torch.randn(K, K, requires_grad=False, dtype=torch.bfloat16), mesh, [Shard(0)]
    )

    expected = torch.add(a_dtensor, b_dtensor)

    fd = FusionDefinition()

    args = [a_dtensor, b_dtensor]

    with fd:
        define_fusion(fd)
        multidevice_schedule(fd, args)
    actual = nvfd.execute_with_dtensors(fd, args)


test_dtensor()
torch.distributed.destroy_process_group()
