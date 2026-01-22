# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition


@pytest.mark.mpi
def test_allgather(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((d * 4,), contiguity=True, dtype=DataType.Float)
        out = fd.ops.set(inp_tv)
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(0, d)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out.set_device_mesh(mesh)

    inp_ref = torch.randn(d * 4)
    out_ref = inp_ref

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_allgather_expanded_broadcast(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([d], contiguity=True, dtype=DataType.Float)
        expanded = fd.ops.broadcast_in_dim(inp_tv, [d, 3], [0])
        out = fd.ops.set(expanded)
        fd.add_output(out)

        for tv in [inp_tv, expanded, out]:
            tv.set_device_mesh(mesh)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        expanded.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d)
    out_ref = inp_ref.unsqueeze(-1).expand(-1, 3)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_allreduce(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((-1, -1, -1), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp_tv, [1])
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(1, d)
        inp_tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    m = 2
    k = d * 3
    n = 5
    inp_ref = torch.randn(m, k, n)
    out_ref = inp_ref.sum(1)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])

    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_reduce_scatter(multidevice_test):
    d = multidevice_test.size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((d, d * 4), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp_tv, [0])
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out.set_device_mesh(mesh)
        out.outer_split(-1, d)
        out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d, d * 4)
    out_ref = inp_ref.sum(0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor_1d(out_ref, 0, mesh))


@pytest.mark.mpi
def test_reduce_scatter_noncontiguous(multidevice_test):
    d = multidevice_test.size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((d, 3, d * 4), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp_tv, [0])
        fd.add_output(out)

        # inp: [iDID{d}, i{3}, i{d*4}]
        # out: [r{d}, i{3}, i{d*4}]
        #                   /    \
        #               iDID{d}  i{4}
        #
        # Unlike test_reduce_scatter, this leads to extra data copy because
        # the scattered axis is not outermost in allocation.
        inp_tv.set_device_mesh(mesh)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out.set_device_mesh(mesh)
        out.outer_split(-1, d)
        out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d, 3, d * 4)
    out_ref = inp_ref.sum(0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor_1d(out_ref, 1, mesh))


# AllToAll patterns seen in expert parallelism
@pytest.mark.mpi
@pytest.mark.parametrize(
    "inp_axis,out_axis", [(0, 1), (1, 0)], ids=["dispatch", "combine"]
)
def test_alltoall(multidevice_test, inp_axis, out_axis):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    n = 3

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((d, d * n), contiguity=True, dtype=DataType.Half)
        out = fd.ops.set(inp_tv)
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(inp_axis, d)
        inp_tv.axis(inp_axis).parallelize(nvfuser.ParallelType.mesh_x)

        out.set_device_mesh(mesh)
        out.outer_split(out_axis, d)
        out.axis(out_axis).parallelize(nvfuser.ParallelType.mesh_x)

    in_ref = torch.randn(d, d * n, dtype=torch.float16)
    out_ref = in_ref

    inp = multidevice_test.shard_tensor(in_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(
        out, multidevice_test.shard_tensor_1d(out_ref, out_axis, mesh)
    )
