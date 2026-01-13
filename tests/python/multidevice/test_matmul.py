# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition, PythonProfiler


# Avoid doing this when possible. This test started to exist before nvFuser
# supports DID loop split. As a result of that, the weight in this test has to be
# 3D, different from a normal linear.
@pytest.mark.mpi
def test_linear_logical_split(multidevice_test):
    d = multidevice_test.size

    b, s, e = 2, 1024, 768

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([b, s, e])
        weight_tv = fd.define_tensor([d, e, e], contiguity=True)
        bias_tv = fd.define_tensor([d, e], contiguity=True)
        out_tv = fd.ops.linear(inp_tv, weight_tv, bias_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, weight_tv, bias_tv]:
            t.set_device_mesh(mesh)
        for t in [weight_tv, bias_tv]:
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(b, s, e)
    weight_ref = torch.randn(d, e, e)
    bias_ref = torch.randn(d, e)
    # [b, s, d * e]
    out_ref = torch.nn.functional.linear(
        inp_ref, weight_ref.view(-1, e), bias_ref.view(-1)
    )
    out_ref = out_ref.view(b, s, d, e).permute(2, 0, 1, 3)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    bias = multidevice_test.shard_tensor(bias_ref, bias_tv)

    (out,) = fd.execute([inp, weight, bias])
    (out_sharding,) = fd.fec.get_output_shardings()

    expected_out = multidevice_test.shard_tensor(out_ref, out_tv)
    # rtol is the same as the default for fp32. atol is slightly increased.
    assert out_sharding.axis_sharded_on(nvfuser.ParallelType.mesh_x) == 0
    torch.testing.assert_close(out, expected_out, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_column_parallel_linear(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    e = 768

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1, e])
        weight_tv = fd.define_tensor([d * e, e])
        bias_tv = fd.define_tensor([d * e])
        out_tv = fd.ops.linear(inp_tv, weight_tv, bias_tv)
        fd.add_output(out_tv)

        for t in [inp_tv, weight_tv, bias_tv]:
            t.set_device_mesh(mesh)

        # Shard N for weight (N, K) and bias (N)
        for t in [weight_tv, bias_tv]:
            t.outer_split(0, d)
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 1024
    inp_ref = torch.randn(b, s, e)
    weight_ref = torch.randn(d * e, e)
    bias_ref = torch.randn(d * e)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    bias = multidevice_test.shard_tensor(bias_ref, bias_tv)

    (out,) = fd.execute([inp, weight, bias])

    # [b, s, d*e]
    out_ref = torch.nn.functional.linear(inp_ref, weight_ref, bias_ref)
    expected_out = multidevice_test.shard_tensor(out_ref, out_tv)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out, expected_out, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_row_parallel_linear(multidevice_test):
    d = multidevice_test.size
    e = 768

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1, d * e])
        weight_tv = fd.define_tensor([e, d * e])
        out_tv = fd.ops.linear(inp_tv, weight_tv, None)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, weight_tv]:
            t.set_device_mesh(mesh)
            t.outer_split(-1, d)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 1024
    inp_ref = torch.randn(b, s, d * e)
    weight_ref = torch.randn(e, d * e)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)

    (out,) = fd.execute([inp, weight])

    out_ref = torch.nn.functional.linear(inp_ref, weight_ref)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out, multidevice_test.shard_tensor(out_ref, out_tv), rtol=1.3e-6, atol=1e-3
    )


@pytest.mark.mpi
def test_row_parallel_linear_with_bias(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    e = 5

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1, d * e])
        weight_tv = fd.define_tensor([e, d * e])
        bias_tv = fd.define_tensor([e])
        out_tv = fd.ops.linear(inp_tv, weight_tv, bias_tv)
        fd.add_output(out_tv)

        for t in [inp_tv, weight_tv]:
            t.set_device_mesh(mesh)
            t.outer_split(-1, d)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 3
    inp_ref = torch.randn(b, s, d * e)
    weight_ref = torch.randn(e, d * e)
    bias_ref = torch.randn(e)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    bias = multidevice_test.shard_tensor(bias_ref, bias_tv)

    (out,) = fd.execute([inp, weight, bias])

    out_ref = torch.nn.functional.linear(inp_ref, weight_ref, bias_ref)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out, multidevice_test.shard_tensor(out_ref, out_tv), rtol=1.3e-6, atol=1e-3
    )


@pytest.mark.mpi
def test_linear_reduce_scatter(multidevice_test):
    d = multidevice_test.size
    b, s, e = 3, 5, 7

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, d * s, d * e], dtype=DataType.BFloat16)
        weight_tv = fd.define_tensor([-1, d * e], dtype=DataType.BFloat16)
        bias_tv = fd.define_tensor([e], dtype=DataType.BFloat16)
        out_tv = fd.ops.linear(inp_tv, weight_tv, bias_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        bias_tv.set_device_mesh(mesh)
        for tv in [inp_tv, weight_tv, out_tv]:
            tv.set_device_mesh(mesh)
            tv.split(-1, d, inner_split=False)
            tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        out_tv.outer_split(1, d)
        out_tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randint(-2, 3, (b, d * s, d * e)).to(torch.bfloat16)
    weight_ref = torch.randint(-2, 3, (e, d * e)).to(torch.bfloat16)
    bias_ref = torch.randint(-2, 3, (e,)).to(torch.bfloat16)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    bias = multidevice_test.shard_tensor(bias_ref, bias_tv)

    with PythonProfiler() as prof:
        (out,) = fd.execute([inp, weight, bias])

    # Only one reduce scatter kernel should be scheduled.
    assert len(
        [kp for kp in prof.profile.kernel_profiles if kp.scheduler == "communication"]
    ) == (1 if d > 1 else 0)

    out_ref = torch.nn.functional.linear(inp_ref, weight_ref, bias_ref)
    torch.testing.assert_close(
        out,
        multidevice_test.shard_tensor(out_ref, out_tv),
    )


@pytest.mark.mpi
def test_column_parallel_matmul(multidevice_test):
    d = multidevice_test.size
    e = 768

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1, e])
        weight_tv = fd.define_tensor([e, d * e])
        out_tv = fd.ops.matmul(inp_tv, weight_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, weight_tv, out_tv]:
            t.set_device_mesh(mesh)

        # Shard N for weight (K, N)
        weight_tv.outer_split(-1, d)
        weight_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        # Output of linear: {.., i{M}, i{N}, r{K}}
        # Shard N -> axis(-2)
        out_tv.outer_split(-2, d)
        out_tv.axis(-3).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 1024
    inp_ref = torch.randn(b, s, e)
    weight_ref = torch.randn(e, d * e)
    out_ref = torch.matmul(inp_ref, weight_ref)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    (out,) = fd.execute([inp, weight])

    # [b, s, d*e]
    expected_out = multidevice_test.shard_tensor(out_ref, out_tv)
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(out, expected_out, rtol=1.3e-6, atol=1e-3)


@pytest.mark.mpi
def test_row_parallel_matmul(multidevice_test):
    d = multidevice_test.size
    e = 8

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, d * e], contiguity=True, dtype=DataType.Half)
        weight_tv = fd.define_tensor([d * e, e], contiguity=True, dtype=DataType.Half)
        out_tv = fd.ops.matmul(inp_tv, weight_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, weight_tv, out_tv]:
            t.set_device_mesh(mesh)

        # Shard K for inp (M, K)
        inp_tv.outer_split(-1, d)
        inp_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        # Shard K for weight (K, N)
        weight_tv.outer_split(0, d)
        weight_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        # I don't think the manual rFactor is necessary any more.
        # DecomposeReshardingsPass should handle it.
        #
        # [i{M}, i{N}, r{K}]
        out_tv.outer_split(-1, d)
        # [i{M}, i{N}, r{d}, r{K//d}]
        local_out = out_tv.rfactor(axes=[-1])
        # local_out = [i{M}, i{N}, i{d}, r{K//d}]
        # out = [i{M}, i{N}, r{d}]
        local_out.set_device_mesh(mesh)
        local_out.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 1, 4
    inp_ref = torch.randn(b * s, d * e, dtype=torch.half)
    weight_ref = torch.randn(d * e, e, dtype=torch.half)
    out_ref = torch.matmul(inp_ref, weight_ref)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    weight = multidevice_test.shard_tensor(weight_ref, weight_tv)
    (out,) = fd.execute([inp, weight])

    torch.testing.assert_close(
        out, multidevice_test.shard_tensor(out_ref, out_tv), rtol=1e-3, atol=1e-2
    )


@pytest.mark.mpi
def test_column_parallel_grouped_mm(multidevice_test):
    d = multidevice_test.size
    g = 4
    k = 16
    n = 64 * d

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, k], dtype=DataType.BFloat16, contiguity=True)
        w_tv = fd.define_tensor([g, n, k], dtype=DataType.BFloat16, contiguity=True)
        w_t = fd.ops.permute(w_tv, [0, 2, 1])
        offsets_tv = fd.define_tensor([g], dtype=DataType.Int32, contiguity=True)
        out_tv = fd.ops.grouped_mm(inp_tv, w_t, offsets_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, w_tv, offsets_tv]:
            t.set_device_mesh(mesh)

        w_tv.outer_split(1, d)
        w_tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    m = 32
    inp_ref = torch.randn(m, k, dtype=torch.bfloat16)
    w_ref = torch.randn(g, n, k, dtype=torch.bfloat16)
    group_sizes = [5, 7, 9, 11]
    assert sum(group_sizes) == m
    group_outs = [
        group_in @ group_w.T
        for group_in, group_w in zip(inp_ref.split(group_sizes), w_ref.unbind())
    ]
    out_ref = torch.cat(group_outs, dim=0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    w = multidevice_test.shard_tensor(w_ref, w_tv)
    offsets = torch.cumsum(torch.tensor(group_sizes), 0, dtype=torch.int32).cuda()
    (out,) = fd.execute([inp, w, offsets])

    torch.testing.assert_close(out, multidevice_test.shard_tensor(out_ref, out_tv))


@pytest.mark.mpi
def test_row_parallel_grouped_mm(multidevice_test):
    d = multidevice_test.size
    g = 4
    k = 16 * d
    n = 64

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, k], dtype=DataType.BFloat16, contiguity=True)
        w_tv = fd.define_tensor([g, n, k], dtype=DataType.BFloat16, contiguity=True)
        w_t = fd.ops.permute(w_tv, [0, 2, 1])
        offsets_tv = fd.define_tensor([g], dtype=DataType.Int32, contiguity=True)
        out_tv = fd.ops.grouped_mm(inp_tv, w_t, offsets_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp_tv, w_tv, offsets_tv]:
            t.set_device_mesh(mesh)

        inp_tv.outer_split(-1, d)
        inp_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        w_tv.outer_split(-1, d)
        w_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    m = 32
    inp_ref = torch.randint(-2, 3, (m, k), dtype=torch.bfloat16)
    w_ref = torch.randint(-2, 3, (g, n, k), dtype=torch.bfloat16)
    group_sizes = [5, 7, 9, 11]
    assert sum(group_sizes) == m
    group_outs = [
        group_in @ group_w.T
        for group_in, group_w in zip(inp_ref.split(group_sizes), w_ref.unbind())
    ]
    out_ref = torch.cat(group_outs, dim=0)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    w = multidevice_test.shard_tensor(w_ref, w_tv)
    offsets = torch.cumsum(torch.tensor(group_sizes), 0, dtype=torch.int32).cuda()
    (out,) = fd.execute([inp, w, offsets])

    torch.testing.assert_close(
        out,
        multidevice_test.shard_tensor(out_ref, out_tv),
    )


@pytest.mark.mpi
def test_issue4729(multidevice_test):
    d = multidevice_test.size

    with FusionDefinition() as fd:
        x_tv = fd.define_tensor([1, 1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        y_tv = fd.define_tensor([1, 1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        w_tv = fd.define_tensor([-1, d * 3], dtype=DataType.BFloat16, contiguity=True)
        x = fd.ops.cast(x_tv, DataType.Float)
        y = fd.ops.cast(y_tv, DataType.Float)
        xy = fd.ops.mul(x, y)
        xy = fd.ops.cast(xy, DataType.BFloat16)
        z_tv = fd.ops.linear(xy, w_tv)
        fd.add_output(z_tv)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [x_tv, y_tv, w_tv]:
            t.set_device_mesh(mesh)
            t.outer_split(-1, d)
            t.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    x_ref = torch.randint(-2, 3, (1, 1, d * 3), dtype=torch.bfloat16)
    y_ref = torch.randint(-2, 3, (1, 1, d * 3), dtype=torch.bfloat16)
    w_ref = torch.randint(-2, 3, (2, d * 3), dtype=torch.bfloat16)
    z_ref = torch.nn.functional.linear(x_ref * y_ref, w_ref)

    x = multidevice_test.shard_tensor(x_ref, x_tv)
    y = multidevice_test.shard_tensor(y_ref, y_tv)
    w = multidevice_test.shard_tensor(w_ref, w_tv)
    (z,) = fd.execute([x, y, w])

    torch.testing.assert_close(
        z, multidevice_test.shard_tensor(z_ref, fd.fusion.outputs()[0])
    )


@pytest.mark.mpi
def test_sequence_parallel_linear(multidevice_test):
    d = multidevice_test.size
    b, s, e = 2, 1024, 768
    assert (
        s % d == 0
    ), f"Sequence length {s} must be divisible by the number of devices {d}"
    assert e % d == 0, f"Hidden size {e} must be divisible by the number of devices {d}"

    with FusionDefinition() as fd:
        inp = fd.define_tensor(shape=[-1, -1, -1], contiguity=True)  # [b, s // d, e]
        weight = fd.define_tensor(shape=[-1, -1], contiguity=True)  # [e // d, e]
        bias = fd.define_tensor(shape=[-1], contiguity=True)  # [e // d]
        out = fd.ops.linear(inp, weight, bias)
        fd.add_output(out)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in [inp, weight, bias, out]:
            t.set_device_mesh(mesh)

        inp.outer_split(1, d)
        inp.axis(1).parallelize(nvfuser.ParallelType.mesh_x)
        weight.outer_split(0, d)
        weight.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        bias.outer_split(0, d)
        bias.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        out.outer_split(1, d)
        out.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(b, s, e)
    weight_ref = torch.randn(e, e)
    bias_ref = torch.randn(e)

    inp = multidevice_test.shard_tensor(inp_ref, inp)
    weight = multidevice_test.shard_tensor(weight_ref, weight)
    bias = multidevice_test.shard_tensor(bias_ref, bias)
    (out,) = fd.execute([inp, weight, bias])

    # [b, s, d*e]
    out_ref = torch.nn.functional.linear(inp_ref, weight_ref, bias_ref)
    expected_out = multidevice_test.shard_tensor(out_ref, fd.fusion.outputs()[0])

    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-2)


@pytest.mark.mpi
def test_data_and_tensor_parallel_mlp(multidevice_test):
    # Build a GPT-3 style MLP with data and tensor parallelism.
    d = multidevice_test.size
    tp_size = 2

    if d % tp_size != 0:
        pytest.skip(f"Number of devices ({d}) must be divisible by tp_size ({tp_size})")

    dp_size = d // tp_size
    rank = multidevice_test.rank

    # mesh_y (dim 0) for data parallelism, mesh_x (dim 1) for tensor parallelism
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d).reshape(dp_size, tp_size))

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    batch_per_rank = 7
    b = dp_size * batch_per_rank
    s, e = 5, 3

    inp_ref = torch.testing.make_tensor(b, s, e, dtype=torch.int32, device="cpu").to(
        torch.float
    )
    up_w_ref = torch.testing.make_tensor(4 * e, e, dtype=torch.int32, device="cpu").to(
        torch.float
    )
    down_w_ref = torch.testing.make_tensor(
        e, 4 * e, dtype=torch.int32, device="cpu"
    ).to(torch.float)

    with FusionDefinition() as fd:
        inp = fd.define_tensor([-1, -1, e], contiguity=True)
        up_w = fd.define_tensor([4 * e, e], contiguity=True)
        down_w = fd.define_tensor([e, 4 * e], contiguity=True)
        up_out = fd.ops.linear(inp, up_w)
        down_out = fd.ops.linear(up_out, down_w)
        fd.add_output(down_out)

        for t in [inp, up_w, down_w]:
            t.set_device_mesh(mesh)

        inp.outer_split(0, dp_size)
        inp.axis(0).parallelize(nvfuser.ParallelType.mesh_y)
        up_w.outer_split(0, tp_size)
        up_w.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        down_w.outer_split(-1, tp_size)
        down_w.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    # TODO: Use shard_tensor when it's ready.
    inp = inp_ref[dp_rank * batch_per_rank : (dp_rank + 1) * batch_per_rank].cuda()
    hidden_per_rank = (4 * e) // tp_size
    up_w = up_w_ref[tp_rank * hidden_per_rank : (tp_rank + 1) * hidden_per_rank].cuda()
    down_w = down_w_ref[
        :, tp_rank * hidden_per_rank : (tp_rank + 1) * hidden_per_rank
    ].cuda()

    (out,) = fd.execute([inp, up_w, down_w])

    up_out_ref = torch.nn.functional.linear(inp_ref, up_w_ref, None)
    down_out_ref = torch.nn.functional.linear(up_out_ref, down_w_ref, None)

    expected_out = down_out_ref[
        dp_rank * batch_per_rank : (dp_rank + 1) * batch_per_rank
    ]

    torch.testing.assert_close(out.cpu(), expected_out)
