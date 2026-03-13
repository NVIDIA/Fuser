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
def test_allgather_2d(multidevice_test):
    d = multidevice_test.size
    tp_size = 2
    if d % tp_size != 0:
        pytest.skip(f"Number of devices ({d=}) must be divisible by {tp_size=}")
    dp_size = d // tp_size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d).reshape(dp_size, tp_size))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        out_tv = fd.ops.set(inp_tv)
        fd.add_output(out_tv)

        for tv in [inp_tv, out_tv]:
            tv.set_device_mesh(mesh)
            tv.outer_split(0, dp_size)
            tv.axis(0).parallelize(nvfuser.ParallelType.mesh_y)
        inp_tv.outer_split(2, tp_size)
        inp_tv.axis(2).parallelize(nvfuser.ParallelType.mesh_x)

    rows_per_rank, cols_per_rank = 2, 3
    rows, cols = dp_size * rows_per_rank, tp_size * cols_per_rank
    inp_ref = torch.randn(rows, cols)
    out_ref = inp_ref

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor(out_ref, out_tv))


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
        out_tv = fd.ops.sum(inp_tv, [0])
        fd.add_output(out_tv)

        inp_tv.set_device_mesh(mesh)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out_tv.set_device_mesh(mesh)
        out_tv.outer_split(-1, d)
        out_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d, d * 4)
    out_ref = inp_ref.sum(0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor(out_ref, out_tv))


@pytest.mark.mpi
def test_reduce_scatter_noncontiguous(multidevice_test):
    d = multidevice_test.size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((d, 3, d * 4), contiguity=True, dtype=DataType.Float)
        out_tv = fd.ops.sum(inp_tv, [0])
        fd.add_output(out_tv)

        # inp: [iDID{d}, i{3}, i{d*4}]
        # out: [r{d}, i{3}, i{d*4}]
        #                   /    \
        #               iDID{d}  i{4}
        #
        # Unlike test_reduce_scatter, this leads to extra data copy because
        # the scattered axis is not outermost in allocation.
        inp_tv.set_device_mesh(mesh)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out_tv.set_device_mesh(mesh)
        out_tv.outer_split(-1, d)
        out_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d, 3, d * 4)
    out_ref = inp_ref.sum(0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor(out_ref, out_tv))


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
        out_tv = fd.ops.set(inp_tv)
        fd.add_output(out_tv)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(inp_axis, d)
        inp_tv.axis(inp_axis).parallelize(nvfuser.ParallelType.mesh_x)

        out_tv.set_device_mesh(mesh)
        out_tv.outer_split(out_axis, d)
        out_tv.axis(out_axis).parallelize(nvfuser.ParallelType.mesh_x)

    in_ref = torch.randn(d, d * n, dtype=torch.float16)
    out_ref = in_ref

    inp = multidevice_test.shard_tensor(in_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, multidevice_test.shard_tensor(out_ref, out_tv))


# Reduce-per-iteration decomposition of ReduceScatter via Swizzle1D.
#
# Models the output of rFactorLoopSplits: c_unreduced has DIDx(D_k) in its
# logical domain (the rFactored k-partial), which gets reduced by sum.
# The scatter dimension is created by outer_split(M) with DIDx in allocation
# and Stream (via Swizzle1D) in loop, so getCommunicationInfo sees a Reduce
# (not Allreduce).
#
# c_unreduced domains:
#   logical: [M, N, D_k]
#   alloc:   [M, N, DIDx(D_k)]
#   loop:    [Stream(Swz(D_m)), M/D, N, DIDx(D_k)]
#
# c domains:
#   logical: [M, N]
#   alloc:   [DIDx(D_m), M/D, N]
#   loop:    [Stream(Swz(D_m)), M/D, N]
@pytest.mark.mpi
def test_decompose_reduce_scatter(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    m, k, n = 3, 5, 7
    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor(
            (d * m, d * k), contiguity=True, dtype=DataType.BFloat16
        )
        w_tv = fd.define_tensor((d * k, n), contiguity=True, dtype=DataType.BFloat16)
        out_tv = fd.ops.matmul(inp_tv, w_tv)  # [d * m, n, r(d * k)]
        fd.add_output(out_tv)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(1, d)
        inp_tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)
        inp_tv.set_allocation_domain(inp_tv.get_loop_domain(), True)

        w_tv.set_device_mesh(mesh)
        w_tv.outer_split(0, d)
        w_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        w_tv.set_allocation_domain(w_tv.get_loop_domain(), True)

        out_tv.set_device_mesh(mesh)
        out_tv.outer_split(-1, d)
        out_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        local_matmul = out_tv.rfactor(axes=[-1])
        local_matmul.outer_split(0, d)
        local_matmul.swizzle1d(0, nvfuser.ParallelType.mesh_x)
        local_matmul.axis(0).parallelize(nvfuser.ParallelType.stream)

        out_tv.outer_split(0, d)
        out_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        out_tv.set_allocation_domain(out_tv.get_loop_domain(), True)
        out_tv.swizzle1d(0, nvfuser.ParallelType.mesh_x)
        out_tv.axis(0).parallelize(nvfuser.ParallelType.stream)

    unsharded_inp = torch.randn(d * m, d * k, dtype=torch.bfloat16)
    inp = multidevice_test.shard_tensor(unsharded_inp, inp_tv)
    print(inp.shape)

    unsharded_w = torch.randn(d * k, n, dtype=torch.bfloat16)
    w = multidevice_test.shard_tensor(unsharded_w, w_tv)
    print(w.shape)

    (out,) = fd.execute(
        [inp, w],
        _enable_options=["host_ir_lowering"],
        _disable_options=["infer_contiguity"],
    )

    out_ref = torch.matmul(unsharded_inp, unsharded_w)
    out_ref = multidevice_test.shard_tensor(out_ref, out_tv)
    torch.testing.assert_close(out, out_ref)
