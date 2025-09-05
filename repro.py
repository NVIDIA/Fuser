# torchrun --local-ranks-filter=0 --nnodes 1 --nproc-per-node 2 test_nvfuser_dtensor_stride_mismatch.py

import torch
from thunder.tests.make_tensor import make_tensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
import os
import torch.distributed as dist
from typing import cast

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

strided_cases = (((5, 6, 2), (1, 1, 7), 2),)
shape, strides, offset = strided_cases[0]

a = make_tensor(
    500,
    dtype=torch.float32,
    device="cuda",
    requires_grad=False,
).as_strided(shape, strides, offset)
a = a.detach()

print(a.shape, "A SHAPE")  # torch.Size([5, 6, 2])
print(a.stride(), "A STRIDE")  # (1, 1, 7)

# in_dtensor = distribute_tensor(a, mesh, [Shard(2)])  # Works
# in_dtensor = distribute_tensor(a, mesh, [Shard(1)])  # Works
in_dtensor = distribute_tensor(a, mesh, [Replicate()])

print(in_dtensor.shape, "DTENSOR SHAPE")  # torch.Size([5, 6, 2])
print(in_dtensor.stride(), "DTENSOR STRIDE")  # (1, 1, 7)
print(in_dtensor.is_contiguous(), "DTENSOR CONTIGUOUS")  # False

# NOTE: The strides for local tensor. They are contiguous.
print(in_dtensor._local_tensor.shape, "LOCAL TENSOR SHAPE")  # torch.Size([5, 6, 2])
print(in_dtensor._local_tensor.stride(), "LOCAL TENSOR STRIDE")  # (12, 2, 1)
print(in_dtensor._local_tensor.is_contiguous(), "LOCAL TENSOR CONTIGUOUS")  # True

# Check PyTorch eager
expected = torch.neg(in_dtensor)


from nvfuser_direct import FusionDefinition, DataType
import nvfuser_direct as nvfd
from thunder.executors.nvfuserex_impl import compute_contiguity

fd = FusionDefinition()

contiguity, stride_order = compute_contiguity(in_dtensor.shape, in_dtensor.stride())

print(contiguity, "CONTIGUITY")
print(stride_order, "STRIDE ORDER")


# This is roughly what thunder does.
def multidevice_schedule(fd: FusionDefinition, in_dtensors: list[DTensor]) -> None:
    for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
        assert isinstance(in_dtensor, DTensor)
        # Set the device mesh.
        assert (
            in_dtensor.device_mesh.ndim == 1
        ), "nvFuser's Python API only supports 1D meshes."
        mesh = nvfd.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh)

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


with fd:
    t0 = fd.define_tensor(
        shape=in_dtensor.shape,
        contiguity=contiguity,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=stride_order,
    )
    t1 = fd.ops.neg(t0)
    fd.add_output(t1)

    multidevice_schedule(fd, [in_dtensor])


# Currently this fails.
# RuntimeError: Stride mismatch with contiguity info.  allocation domain: iS2{2}, iS0{5}, iS1{6}: sizes: [2, 5, 6]: strides: [1, 12, 2]; contiguity: f, f, t; dim: 2; expected stride: 1; actual stride: 2
actual = nvfd.execute_with_dtensors(fd, [in_dtensor])

torch.testing.assert_close(actual[0], expected)

dist.destroy_process_group()
