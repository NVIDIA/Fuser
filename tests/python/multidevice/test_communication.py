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
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))

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
def test_allreduce(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))

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

    # The first dimension of the TensorView is the world size.
    # When it is one, the axis becomes a broadcast iterDomain.
    # The sum operations uses squeeze for size-1 broadcast domains.
    # Squeeze is not supported by getCommunicationInfo.
    # This test is skipped when the world size is 1.
    if d == 1:
        pytest.skip("This test requires > 1 MPI processes")

    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))

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

    # The first dimension of the TensorView is the world size.
    # When it is one, the axis becomes a broadcast iterDomain.
    # The sum operations uses squeeze for size-1 broadcast domains.
    # Squeeze is not supported by getCommunicationInfo.
    # This test is skipped when the world size is 1.
    if d == 1:
        pytest.skip("This test requires > 1 MPI processes")

    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))

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
