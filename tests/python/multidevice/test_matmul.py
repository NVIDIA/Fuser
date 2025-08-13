# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition


# Avoid doing this when possible. This test started to exist before nvFuser
# supports DID loop split. As a result of that, the weight in this test has to be
# 3D, different from a normal linear.
@pytest.mark.mpi
def test_linear_logical_split(multidevice_direct_test):
    def _definition(fd: FusionDefinition, d: int, b: int, s: int, e: int):
        inp = fd.define_tensor([b, s, e])
        weight = fd.define_tensor([d, e, e], contiguity=True)
        bias = fd.define_tensor([d, e], contiguity=True)
        out = fd.ops.linear(inp, weight, bias)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition, d: int):
        mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
        inp, weight, bias = fd.fusion.inputs()
        for t in [inp, weight, bias]:
            t.set_device_mesh(mesh)
        for t in [weight, bias]:
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    d = multidevice_direct_test.size
    rank = multidevice_direct_test.rank

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s, e = 2, 1024, 768
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(d * e, e, device="cuda")
    weight_tensor = unsharded_weight_tensor.view([d, e, e])[rank : rank + 1]
    unsharded_bias_tensor = torch.randn(d * e, device="cuda")
    bias_tensor = unsharded_bias_tensor.view([d, e])[rank : rank + 1]

    with FusionDefinition() as fd:
        _definition(fd, d, b, s, e)
        _multidevice_schedule(fd, d)

    (out_tensor,) = fd.execute([inp_tensor, weight_tensor, bias_tensor])
    (out_sharding,) = fd.fec.get_output_shardings()

    # [b, s, d*e]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor, unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = unsharded_out_tensor.view([b, s, d, e]).permute(2, 0, 1, 3)[
        rank : rank + 1
    ]
    # rtol is the same as the default for fp32. atol is slightly increased.
    assert out_sharding.axis_sharded_on(nvfuser.ParallelType.mesh_x) == 0
    torch.testing.assert_close(out_tensor, expected_out_tensor, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_column_parallel_linear(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 768

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, -1, e])
        weight = fd.define_tensor([d * e, e])
        bias = fd.define_tensor([d * e])
        out = fd.ops.linear(inp, weight, bias)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, weight, bias = fd.fusion.inputs()
        for t in [inp, weight, bias]:
            t.set_device_mesh(mesh)

        # Shard N for weight (N, K) and bias (N)
        for t in [weight, bias]:
            t.split(0, d, inner_split=False)
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s = 2, 1024
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(d * e, e)
    sharded_weight_tensor = multidevice_direct_test.shard_tensor(
        unsharded_weight_tensor, 0, mesh
    )
    unsharded_bias_tensor = torch.randn(d * e)
    sharded_bias_tensor = multidevice_direct_test.shard_tensor(
        unsharded_bias_tensor, 0, mesh
    )

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out_tensor,) = fd.execute([inp_tensor, sharded_weight_tensor, sharded_bias_tensor])

    # [b, s, d*e]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor.cpu(), unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = multidevice_direct_test.shard_tensor(
        unsharded_out_tensor, -1, mesh
    )
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out_tensor, expected_out_tensor, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_row_parallel_linear(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 768

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, -1, d * e])
        weight = fd.define_tensor([e, d * e])
        out = fd.ops.linear(inp, weight, None)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, weight = fd.fusion.inputs()
        for t in [inp, weight]:
            t.set_device_mesh(mesh)
            t.split(-1, d, inner_split=False)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s = 2, 1024
    unsharded_inp = torch.randn(b, s, d * e)
    unsharded_weight = torch.randn(e, d * e)

    inp = multidevice_direct_test.shard_tensor(unsharded_inp, -1, mesh)
    weight = multidevice_direct_test.shard_tensor(unsharded_weight, -1, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([inp, weight])

    unsharded_out = torch.nn.functional.linear(unsharded_inp, unsharded_weight, None)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out.cpu(), unsharded_out, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_row_parallel_linear_with_bias(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 5

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, -1, d * e])
        weight = fd.define_tensor([e, d * e])
        bias = fd.define_tensor([e])
        out = fd.ops.linear(inp, weight, bias)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, weight, _ = fd.fusion.inputs()
        for t in [inp, weight]:
            t.set_device_mesh(mesh)
            t.split(-1, d, inner_split=False)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s = 2, 3
    unsharded_inp = torch.randn(b, s, d * e)
    unsharded_weight = torch.randn(e, d * e)
    bias = torch.randn(e)

    inp = multidevice_direct_test.shard_tensor(unsharded_inp, -1, mesh)
    weight = multidevice_direct_test.shard_tensor(unsharded_weight, -1, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([inp, weight, bias.cuda()])

    unsharded_out = torch.nn.functional.linear(unsharded_inp, unsharded_weight, bias)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out.cpu(), unsharded_out, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_linear_reduce_scatter(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 768

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, -1, d * e])
        weight = fd.define_tensor([e, d * e])
        out = fd.ops.linear(inp, weight, None)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, weight = fd.fusion.inputs()
        (out,) = fd.fusion.outputs()
        for t in [inp, weight, out]:
            t.set_device_mesh(mesh)
            t.split(-1, d, inner_split=False)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        # Scatter
        out.split(1, d, inner_split=False)
        out.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    # set b=1 as a temporary fix for the test to pass.
    # TODO: set b>1 once reduce scatter is fixed.
    b, s = 2, 1024
    unsharded_inp = torch.randn(b, s, d * e)
    unsharded_weight = torch.randn(e, d * e)

    inp = multidevice_direct_test.shard_tensor(unsharded_inp, -1, mesh)
    weight = multidevice_direct_test.shard_tensor(unsharded_weight, -1, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([inp, weight])

    unsharded_out = torch.nn.functional.linear(unsharded_inp, unsharded_weight, None)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out,
        multidevice_direct_test.shard_tensor(unsharded_out, 1, mesh),
        rtol=1.3e-6,
        atol=1e-3,
    )


@pytest.mark.mpi
def test_column_parallel_matmul(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 768

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, -1, e])
        weight = fd.define_tensor([e, d * e])
        out = fd.ops.matmul(inp, weight)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, weight = fd.fusion.inputs()
        (out,) = fd.fusion.outputs()
        for t in [inp, weight, out]:
            t.set_device_mesh(mesh)

        # Shard N for weight (K, N)
        weight.split(-1, d, inner_split=False)
        weight.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        # Output of linear: {.., i{M}, i{N}, r{K}}
        # Shard N -> axis(-2)
        out.split(-2, d, inner_split=False)
        out.axis(-3).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s = 2, 1024
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(e, d * e)
    sharded_weight_tensor = multidevice_direct_test.shard_tensor(
        unsharded_weight_tensor, -1, mesh
    )

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out_tensor,) = fd.execute([inp_tensor, sharded_weight_tensor])

    # [b, s, d*e]
    unsharded_out_tensor = torch.matmul(inp_tensor.cpu(), unsharded_weight_tensor)
    expected_out_tensor = multidevice_direct_test.shard_tensor(
        unsharded_out_tensor, -1, mesh
    )
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out_tensor, expected_out_tensor.squeeze(0), rtol=1.3e-6, atol=1e-3
    )


@pytest.mark.mpi
def test_row_parallel_matmul(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    e = 8

    def _definition(fd: FusionDefinition) -> None:
        inp = fd.define_tensor([-1, d * e], contiguity=True, dtype=DataType.Half)
        weight = fd.define_tensor([d * e, e], contiguity=True, dtype=DataType.Half)
        out = fd.ops.matmul(inp, weight)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition) -> None:
        inp, weight = fd.fusion.inputs()
        (out,) = fd.fusion.outputs()
        for t in [inp, weight, out]:
            t.set_device_mesh(mesh)

        # Shard K for inp (M, K)
        inp.split(-1, d, inner_split=False)
        inp.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        # Shard K for weight (K, N)
        weight.split(0, d, inner_split=False)
        weight.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        # [i{M}, i{N}, r{K}]
        out.split(-1, d, inner_split=False)
        # [i{M}, i{N}, r{d}, r{K//d}]
        local_out = out.rfactor(axes=[-1])
        # local_out = [i{M}, i{N}, i{d}, r{K//d}]
        # out = [i{M}, i{N}, r{d}]
        local_out.set_device_mesh(mesh)
        local_out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    b, s = 1, 4
    unsharded_inp = torch.randn(b * s, d * e, dtype=torch.half)
    unsharded_weight = torch.randn(d * e, e, dtype=torch.half)
    sharded_inp = multidevice_direct_test.shard_tensor(unsharded_inp, -1, mesh)
    sharded_weight = multidevice_direct_test.shard_tensor(unsharded_weight, 0, mesh)

    expected_out = torch.matmul(unsharded_inp, unsharded_weight)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([sharded_inp, sharded_weight])

    torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-3, atol=1e-2)


@pytest.mark.mpi
def test_column_parallel_grouped_mm(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    g = 4
    k = 16
    n = 16 * d

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, k], dtype=DataType.BFloat16, contiguity=True)
        w = fd.define_tensor([g, k, n], dtype=DataType.BFloat16, contiguity=True)
        offsets = fd.define_tensor([g], dtype=DataType.Int32, contiguity=True)
        out = fd.ops.grouped_mm(inp, w, offsets)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, w, offsets = fd.fusion.inputs()
        for t in [inp, w, offsets]:
            t.set_device_mesh(mesh)

        w.split(-1, d, inner_split=False)
        w.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    m = 32
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(g, k, n, dtype=torch.bfloat16)
    sharded_w = multidevice_direct_test.shard_tensor(w, -1, mesh)
    group_sizes = [5, 7, 9, 11]
    assert sum(group_sizes) == m
    offsets = torch.cumsum(torch.tensor(group_sizes), 0, dtype=torch.int32).cuda()

    group_outs = [
        group_in.cpu() @ group_w
        for group_in, group_w in zip(inp.split(group_sizes), w.unbind())
    ]
    expected_out = torch.cat(group_outs, dim=0)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([inp, sharded_w, offsets])

    torch.testing.assert_close(
        out, multidevice_direct_test.shard_tensor(expected_out, -1, mesh)
    )


@pytest.mark.mpi
def test_row_parallel_grouped_mm(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))
    g = 4
    k = 16 * d
    n = 16

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, k], dtype=DataType.BFloat16, contiguity=True)
        w = fd.define_tensor([g, k, n], dtype=DataType.BFloat16, contiguity=True)
        offsets = fd.define_tensor([g], dtype=DataType.Int32, contiguity=True)
        out = fd.ops.grouped_mm(inp, w, offsets)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, w, offsets = fd.fusion.inputs()
        for t in [inp, w, offsets]:
            t.set_device_mesh(mesh)

        inp.split(-1, d, inner_split=False)
        inp.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        w.split(1, d, inner_split=False)
        w.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    m = 32
    inp = torch.randint(-2, 3, (m, k), dtype=torch.bfloat16)
    sharded_inp = multidevice_direct_test.shard_tensor(inp, -1, mesh)
    w = torch.randint(-2, 3, (g, k, n), dtype=torch.bfloat16)
    sharded_w = multidevice_direct_test.shard_tensor(w, 1, mesh)
    group_sizes = [5, 7, 9, 11]
    assert sum(group_sizes) == m
    offsets = torch.cumsum(torch.tensor(group_sizes), 0, dtype=torch.int32).cuda()

    group_outs = [
        group_in @ group_w
        for group_in, group_w in zip(inp.split(group_sizes), w.unbind())
    ]
    expected_out = torch.cat(group_outs, dim=0)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (out,) = fd.execute([sharded_inp, sharded_w, offsets])

    torch.testing.assert_close(
        out.cpu(),
        expected_out,
    )


@pytest.mark.mpi
def test_issue4729(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.tensor(range(d)))

    def _definition(fd: FusionDefinition):
        x = fd.define_tensor([1, 1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        y = fd.define_tensor([1, 1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        w = fd.define_tensor([-1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        x = fd.ops.cast(x, DataType.Float)
        y = fd.ops.cast(y, DataType.Float)
        xy = fd.ops.mul(x, y)
        xy = fd.ops.cast(xy, DataType.BFloat16)
        out = fd.ops.linear(xy, w)
        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        x, y, w = fd.fusion.inputs()
        for t in [x, y, w]:
            t.set_device_mesh(mesh)
            t.split(-1, d, inner_split=False)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    x_ref = torch.randint(-2, 3, (1, 1, d * 3), dtype=torch.bfloat16)
    y_ref = torch.randint(-2, 3, (1, 1, d * 3), dtype=torch.bfloat16)
    w_ref = torch.randint(-2, 3, (2, d * 3), dtype=torch.bfloat16)
    x = multidevice_direct_test.shard_tensor(x_ref, -1, mesh)
    y = multidevice_direct_test.shard_tensor(y_ref, -1, mesh)
    w = multidevice_direct_test.shard_tensor(w_ref, -1, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (z,) = fd.execute([x, y, w])

    torch.testing.assert_close(
        z.cpu(), torch.nn.functional.linear(x_ref * y_ref, w_ref)
    )
