# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from enum import Enum, auto
from torch.nn.attention import SDPBackend

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition
from python.direct_utils import is_pre_ampere


@pytest.mark.mpi
def test_sizes_and_ranks(multidevice_test):
    size, rank, local_size, local_rank = (
        multidevice_test.size,
        multidevice_test.rank,
        multidevice_test.local_size,
        multidevice_test.local_rank,
    )
    assert size > 0
    assert rank >= 0 and rank < size
    assert local_size > 0
    assert local_rank >= 0 and local_rank < local_size


@pytest.mark.mpi
def test_pointwise(multidevice_test):
    num_devices = multidevice_test.size

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((-1, -1), contiguity=False, dtype=DataType.Float)
        tv1 = fd.ops.relu(inp_tv)
        tv2 = fd.ops.add(tv1, tv1)
        fd.add_output(tv2)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(num_devices))
        for tv in [inp_tv, tv1, tv2]:
            tv.set_device_mesh(mesh)

        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(num_devices, 4)
    out_ref = inp_ref.relu() * 2

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_transpose(multidevice_test):
    d = multidevice_test.size
    cp_size = 2
    if d % (cp_size * cp_size) != 0:
        pytest.skip(
            f"We only support even split, so {d} has to be divisible by {cp_size * cp_size} for {cp_size=}."
        )
    dp_size = d // (cp_size * cp_size)

    c = 128
    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor(
            (-1, c, -1, -1, cp_size), contiguity=True, dtype=DataType.BFloat16
        )
        out_tv = fd.ops.set(inp_tv)
        fd.add_output(out_tv)

        mesh = nvfuser.multidevice.DeviceMesh(
            torch.arange(d).reshape(dp_size, cp_size, cp_size)
        )
        for tv in [inp_tv, out_tv]:
            tv.set_device_mesh(mesh)

        inp_tv.axis(4).parallelize(nvfuser.ParallelType.mesh_y)
        inp_tv.outer_split(3, cp_size)
        inp_tv.axis(3).parallelize(nvfuser.ParallelType.mesh_x)
        inp_tv.outer_split(0, dp_size)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_z)

        out_tv.axis(4).parallelize(nvfuser.ParallelType.mesh_y)
        out_tv.outer_split(3, cp_size)
        out_tv.axis(3).parallelize(nvfuser.ParallelType.mesh_x)
        out_tv.outer_split(0, dp_size)
        out_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_z)
        out_tv.set_allocation_domain(
            (
                out_tv.axis(3),
                out_tv.axis(0),
                out_tv.axis(1),
                out_tv.axis(2),
                out_tv.axis(4),
                out_tv.axis(5),
                out_tv.axis(6),
            ),
            True,
        )

    b = dp_size * 3
    s = cp_size * 5
    inp_ref = torch.randn(b, c, s, s, cp_size, dtype=torch.bfloat16)
    out_ref = inp_ref

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    fd.execute([inp])


class QkvFormat(Enum):
    BHSE = auto()
    BSHE = auto()


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.parametrize("qkv_format", [QkvFormat.BHSE, QkvFormat.BSHE])
@pytest.mark.mpi
def test_sdpa(multidevice_test, qkv_format: QkvFormat):
    d, b, s, h, e = multidevice_test.size, 2, 1024, 12, 768

    if h % d != 0:
        pytest.skip(f"We only support even split, so {h} has to be divisible by {d}.")

    def _definition(fd: FusionDefinition, qkv_format: QkvFormat) -> None:
        match qkv_format:
            case QkvFormat.BHSE:
                stride_order = [4, 3, 2, 1, 0]
            case QkvFormat.BSHE:
                stride_order = [4, 3, 1, 2, 0]

        q, k, v, out_grad = [
            fd.define_tensor(
                shape=[d, b, h // d, s, e // h],
                contiguity=True,
                dtype=DataType.BFloat16,
                stride_order=stride_order,
            )
            for _ in range(4)
        ]

        # TODO(#3123): support sharded dropout and change this to a
        # positive probability.
        dropout_p = fd.define_scalar(0.0, dtype=DataType.Double)
        is_causal = fd.define_scalar(True, dtype=DataType.Bool)
        attn, logsumexp, seed, offset = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal
        )

        q_grad, k_grad, v_grad = fd.ops.sdpfa_bwd(
            out_grad,
            q,
            k,
            v,
            attn,
            logsumexp,
            dropout_p,
            is_causal,
            seed,
            offset,
            scale=None,
        )

        fd.add_output(attn)
        for grad in [q_grad, k_grad, v_grad]:
            fd.add_output(grad)

    def _multidevice_schedule(fd: FusionDefinition) -> None:
        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
        for t in fd.fusion.inputs():
            t.set_device_mesh(mesh)
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    def make_unsharded_tensor() -> torch.Tensor:
        return torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cuda")

    q, k, v = [make_unsharded_tensor().requires_grad_() for _ in range(3)]
    out_grad = make_unsharded_tensor()

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=None
        )
        expected_out.backward(out_grad)
        expected_q_grad, expected_k_grad, expected_v_grad = q.grad, k.grad, v.grad

    rank = multidevice_test.rank

    # Head-parallelize Q, K, V or the attention output of an SDPA.
    def head_parallelize(t: torch.Tensor) -> torch.Tensor:
        assert t.shape == torch.Size([b, h, s, e // h])
        # The input `t` may require gradient. We .detach() so the new `t`
        # doesn't require gradient.
        t = t.detach().view([b, d, h // d, s, e // h]).narrow(1, rank, 1)
        match qkv_format:
            case QkvFormat.BHSE:
                return t.transpose(0, 1).contiguous()
            case QkvFormat.BSHE:
                return t.permute(1, 0, 3, 2, 4).contiguous().transpose(2, 3)

    with FusionDefinition() as fd:
        _definition(fd, qkv_format)
        _multidevice_schedule(fd)

    out, q_grad, k_grad, v_grad = fd.execute(
        [
            head_parallelize(q).requires_grad_(),
            head_parallelize(k).requires_grad_(),
            head_parallelize(v).requires_grad_(),
            head_parallelize(out_grad),
        ]
    )

    def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
        match qkv_format:
            case QkvFormat.BHSE:
                assert actual.is_contiguous()
            case QkvFormat.BSHE:
                assert actual.transpose(2, 3).is_contiguous()

        # Use the default rtol for bfloat16 and a relaxed atol.
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)

    assert_close(out, head_parallelize(expected_out))
    assert_close(q_grad, head_parallelize(expected_q_grad))
    assert_close(k_grad, head_parallelize(expected_k_grad))
    assert_close(v_grad, head_parallelize(expected_v_grad))


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.parametrize("qkv_format", [QkvFormat.BHSE, QkvFormat.BSHE])
@pytest.mark.mpi
def test_sdpa_loop_split(multidevice_test, qkv_format: QkvFormat):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    h, e = 12, 768
    if h % d != 0:
        pytest.skip(f"We only support even split, so {h} has to be divisible by {d}.")

    def _definition(fd: FusionDefinition, qkv_format: QkvFormat) -> None:
        match qkv_format:
            case QkvFormat.BHSE:
                stride_order = [3, 2, 1, 0]
            case QkvFormat.BSHE:
                stride_order = [3, 1, 2, 0]

        q, k, v, out_grad = [
            fd.define_tensor(
                shape=[-1, h, -1, e // h],
                dtype=DataType.BFloat16,
                stride_order=stride_order,
            )
            for _ in range(4)
        ]

        # TODO(#3123): support sharded dropout and change this to a
        # positive probability.
        dropout_p = fd.define_scalar(0.0, dtype=DataType.Double)
        is_causal = fd.define_scalar(True, dtype=DataType.Bool)
        attn, logsumexp, seed, offset = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal
        )

        q_grad, k_grad, v_grad = fd.ops.sdpfa_bwd(
            out_grad,
            q,
            k,
            v,
            attn,
            logsumexp,
            dropout_p,
            is_causal,
            seed,
            offset,
            scale=None,
        )

        fd.add_output(attn)
        for grad in [q_grad, k_grad, v_grad]:
            fd.add_output(grad)

    def _multidevice_schedule(fd: FusionDefinition) -> None:
        for t in fd.fusion.inputs():
            t.set_device_mesh(mesh)
            t.outer_split(1, d)
            t.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 1024

    def make_unsharded_tensor() -> torch.Tensor:
        return torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cpu")

    q, k, v = [make_unsharded_tensor().requires_grad_() for _ in range(3)]
    out_grad = make_unsharded_tensor()
    sharded_q, sharded_k, sharded_v, sharded_out_grad = [
        multidevice_test.shard_tensor_1d(t, 1, mesh) for t in [q, k, v, out_grad]
    ]

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=None
        )
        expected_out.backward(out_grad)
        expected_q_grad, expected_k_grad, expected_v_grad = q.grad, k.grad, v.grad

    with FusionDefinition() as fd:
        _definition(fd, qkv_format)
        _multidevice_schedule(fd)

    def reformat_tensor(t: torch.Tensor) -> torch.Tensor:
        match qkv_format:
            case QkvFormat.BHSE:
                return t
            case QkvFormat.BSHE:
                return t.transpose(1, 2).contiguous().transpose(1, 2)

    attn, q_grad, k_grad, v_grad = fd.execute(
        [
            reformat_tensor(sharded_q).requires_grad_(),
            reformat_tensor(sharded_k).requires_grad_(),
            reformat_tensor(sharded_v).requires_grad_(),
            reformat_tensor(sharded_out_grad),
        ]
    )

    def assert_close(actual, expected):
        match qkv_format:
            case QkvFormat.BHSE:
                assert actual.is_contiguous()
            case QkvFormat.BSHE:
                assert actual.transpose(1, 2).is_contiguous()

        # Use the default rtol for bfloat16 and a relaxed atol.
        torch.testing.assert_close(
            actual,
            multidevice_test.shard_tensor_1d(expected, 1, mesh),
            rtol=1.6e-2,
            atol=1e-2,
        )

    for actual, expected in zip(
        [attn, q_grad, k_grad, v_grad],
        [expected_out, expected_q_grad, expected_k_grad, expected_v_grad],
    ):
        assert_close(actual, expected)


@pytest.mark.mpi
def test_privatize_squeeze(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((-1, -1, -1), dtype=DataType.BFloat16)
        tv1 = fd.ops.broadcast(inp_tv, [True, False, False, False])
        tv2 = fd.ops.squeeze(tv1, dims=[0])
        tv3 = fd.ops.cast(tv2, dtype=DataType.Float)
        tv4 = fd.ops.sum(tv3, dims=[0])
        tv5 = fd.ops.sum(tv3, dims=[1])
        fd.add_output(tv4)
        fd.add_output(tv5)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(0, d)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(d * 3, 5, 6, dtype=torch.bfloat16)
    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)

    out1, out2 = fd.execute([inp])
    torch.testing.assert_close(out1.cpu(), inp_ref.to(torch.float).sum(0))
    torch.testing.assert_close(out2, inp.to(torch.float).sum(1))


@pytest.mark.mpi
def test_inner_reduction(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((-1, -1), dtype=DataType.Float)
        out = fd.ops.sum(inp_tv, [1])
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(0, d)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.ones(d * 3, 5)

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out, inp.sum(1))


@pytest.mark.mpi
def test_insert_resharding_after(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor((-1, -1), contiguity=True)
        out = fd.ops.relu(inp_tv)
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(0, d)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

        out.set_device_mesh(mesh)

    inp_ref = torch.randn(d * 3, 5)
    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)

    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), inp_ref.relu())


@pytest.mark.mpi
def test_welford(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    b, s, e = 1, 2048, 12288

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor(shape=[b, s, e], contiguity=True)
        var, mean = fd.ops.var_mean(inp_tv, dims=[2], correction=0, keepdim=False)
        fd.add_output(var)
        fd.add_output(mean)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(1, d)
        inp_tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    inp_ref = torch.randn(b, s, e)
    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)

    var, mean = fd.execute([inp])

    torch.testing.assert_close(var, inp.var(2), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(mean, inp.mean(2), rtol=1e-3, atol=1e-3)


@pytest.mark.mpi
def test_binary(multidevice_test):
    d = multidevice_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    with FusionDefinition() as fd:
        x_tv = fd.define_tensor((-1, -1), contiguity=True, dtype=DataType.Half)
        y_tv = fd.define_tensor((-1, -1), contiguity=True, dtype=DataType.Half)
        z = fd.ops.add(x_tv, y_tv)
        fd.add_output(z)

        x_tv.set_device_mesh(mesh)
        y_tv.set_device_mesh(mesh)
        y_tv.outer_split(0, d)
        y_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    x_ref = torch.randn(d * 2, 3, dtype=torch.float16)
    y_ref = torch.randn(d * 2, 3, dtype=torch.float16)
    z_ref = x_ref.float() + y_ref.float()

    x = multidevice_test.shard_tensor(x_ref, x_tv)
    y = multidevice_test.shard_tensor(y_ref, y_tv)
    (z,) = fd.execute([x, y])

    torch.testing.assert_close(z, multidevice_test.shard_tensor_1d(z_ref, 0, mesh))


@pytest.mark.mpi
def test_reduction_with_2d_mesh(multidevice_test):
    d = multidevice_test.size
    tp_size = 2

    # Skip if d is not divisible by tp_size
    if d % tp_size != 0:
        pytest.skip(f"Number of devices ({d}) must be divisible by tp_size ({tp_size})")

    dp_size = d // tp_size

    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d).reshape(dp_size, tp_size))

    with FusionDefinition() as fd:
        inp_tv = fd.define_tensor([-1, -1], dtype=DataType.Float, contiguity=True)
        out = fd.ops.sum(inp_tv, [1])
        fd.add_output(out)

        inp_tv.set_device_mesh(mesh)
        inp_tv.outer_split(0, dp_size)
        inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_y)
        inp_tv.outer_split(-1, tp_size)
        inp_tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

    rank = multidevice_test.rank
    rows_per_rank, cols_per_rank = 2, 3
    rows, cols = dp_size * rows_per_rank, tp_size * cols_per_rank

    inp_ref = torch.arange(rows * cols, dtype=torch.float).reshape(rows, cols)
    out_ref = inp_ref.sum([-1])

    inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
    (out,) = fd.execute([inp])

    dp_rank = rank // tp_size
    torch.testing.assert_close(
        out.cpu(), out_ref[dp_rank * rows_per_rank : (dp_rank + 1) * rows_per_rank]
    )
