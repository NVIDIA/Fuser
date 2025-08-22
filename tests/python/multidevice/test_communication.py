# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition


@pytest.mark.mpi
def test_allgather(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor((d * 4,), contiguity=True, dtype=DataType.Float)
        out = fd.ops.set(inp)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        for out in fd.fusion.outputs():
            out.set_device_mesh(mesh)

        for inp in fd.fusion.inputs():
            inp.set_device_mesh(mesh)
            inp.split(0, d, inner_split=False)
            inp.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded = torch.randn(d * 4)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 0, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (output,) = fd.execute([sharded])
    torch.testing.assert_close(output.cpu(), unsharded)


@pytest.mark.mpi
def test_allgather_expanded_broadcast(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp = fd.define_tensor([d], contiguity=True, dtype=DataType.Float)
        expanded = fd.ops.broadcast_in_dim(inp, [d, 3], [0])
        out = fd.ops.set(expanded)
        fd.add_output(out)

        for tv in [inp, expanded, out]:
            tv.set_device_mesh(mesh)
        inp.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        expanded.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded_inp = torch.randn(d)
    inp = multidevice_direct_test.shard_tensor(unsharded_inp, 0, mesh)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), unsharded_inp.unsqueeze(-1).expand(-1, 3))


@pytest.mark.mpi
def test_allreduce(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor((-1, -1, -1), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp, [1])
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        for inp in fd.fusion.inputs():
            inp.set_device_mesh(mesh)
            inp.split(1, d, inner_split=False)
            inp.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    m = 2
    k = d * 3
    n = 5
    unsharded = torch.randn(m, k, n)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 1, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (output,) = fd.execute([sharded])
    torch.testing.assert_close(output.cpu(), unsharded.sum(1))


@pytest.mark.mpi
def test_reduce_scatter(multidevice_direct_test):
    d = multidevice_direct_test.size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor((d, d * 4), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp, [0])
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        for inp in fd.fusion.inputs():
            inp.set_device_mesh(mesh)
            inp.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        for out in fd.fusion.outputs():
            out.set_device_mesh(mesh)
            out.split(-1, d, inner_split=False)
            out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded = torch.randn(d, d * 4)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 0, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (output,) = fd.execute([sharded])
    torch.testing.assert_close(
        output, multidevice_direct_test.shard_tensor(unsharded.sum(0), 0, mesh)
    )


@pytest.mark.mpi
def test_reduce_scatter_noncontiguous(multidevice_direct_test):
    d = multidevice_direct_test.size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor((d, 3, d * 4), contiguity=True, dtype=DataType.Float)
        out = fd.ops.sum(inp, [0])
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        # inp: [iDID{d}, i{3}, i{d*4}]
        # out: [r{d}, i{3}, i{d*4}]
        #                   /    \
        #               iDID{d}  i{4}
        #
        # Unlike test_reduce_scatter, this leads to extra data copy because
        # the scattered axis is not outermost in allocation.
        for inp in fd.fusion.inputs():
            inp.set_device_mesh(mesh)
            inp.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        for out in fd.fusion.outputs():
            out.set_device_mesh(mesh)
            out.split(-1, d, inner_split=False)
            out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded = torch.randn(d, 3, d * 4)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 0, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (output,) = fd.execute([sharded])
    torch.testing.assert_close(
        output, multidevice_direct_test.shard_tensor(unsharded.sum(0), 1, mesh)
    )
