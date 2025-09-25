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
def test_sizes_and_ranks(multidevice_direct_test):
    size, rank, local_size, local_rank = (
        multidevice_direct_test.size,
        multidevice_direct_test.rank,
        multidevice_direct_test.local_size,
        multidevice_direct_test.local_rank,
    )
    assert size > 0
    assert rank >= 0 and rank < size
    assert local_size > 0
    assert local_rank >= 0 and local_rank < local_size


@pytest.mark.mpi
def test_pointwise(multidevice_direct_test):
    num_devices = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(num_devices))

    def _definition(fd: FusionDefinition):
        t0 = fd.define_tensor((-1, -1), contiguity=False, dtype=DataType.Float)
        t1 = fd.ops.relu(t0)
        t2 = fd.ops.add(t1, t1)
        fd.add_output(t2)

    def _multidevice_schedule(fd: FusionDefinition):
        for t in fd.fusion.vals():
            if t.is_tensor():
                t.set_device_mesh(mesh)

        for t in fd.fusion.inputs():
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded_input = torch.randn(num_devices, 4)
    sharded_input = multidevice_direct_test.shard_tensor(unsharded_input, 0, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    (output,) = fd.execute([sharded_input])
    (output_sharding,) = fd.fec.get_output_shardings()
    torch.testing.assert_close(output.cpu(), unsharded_input.relu() * 2)
    assert output_sharding.axis_sharded_on(nvfuser.ParallelType.mesh_x) == -1


class QkvFormat(Enum):
    BHSE = auto()
    BSHE = auto()


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.parametrize("qkv_format", [QkvFormat.BHSE, QkvFormat.BSHE])
@pytest.mark.mpi
def test_sdpa(multidevice_direct_test, qkv_format: QkvFormat):
    d, b, s, h, e = multidevice_direct_test.size, 2, 1024, 12, 768

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
        attn, log_sumexp, seed, offset = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p, is_causal, scale=None
        )

        q_grad, k_grad, v_grad = fd.ops.sdpfa_bwd(
            out_grad,
            q,
            k,
            v,
            attn,
            log_sumexp,
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

    torch.cuda.set_device(multidevice_direct_test.local_rank)

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

    rank = multidevice_direct_test.rank

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
def test_sdpa_loop_split(multidevice_direct_test, qkv_format: QkvFormat):
    d = multidevice_direct_test.size
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
        attn, log_sumexp, seed, offset = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p, is_causal, scale=None
        )

        q_grad, k_grad, v_grad = fd.ops.sdpfa_bwd(
            out_grad,
            q,
            k,
            v,
            attn,
            log_sumexp,
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
            t.split(1, d, inner_split=False)
            t.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    b, s = 2, 1024

    torch.cuda.set_device(multidevice_direct_test.local_rank)

    def make_unsharded_tensor() -> torch.Tensor:
        return torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cpu")

    q, k, v = [make_unsharded_tensor().requires_grad_() for _ in range(3)]
    out_grad = make_unsharded_tensor()
    sharded_q, sharded_k, sharded_v, sharded_out_grad = [
        multidevice_direct_test.shard_tensor(t, 1, mesh) for t in [q, k, v, out_grad]
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
            multidevice_direct_test.shard_tensor(expected, 1, mesh),
            rtol=1.6e-2,
            atol=1e-2,
        )

    for actual, expected in zip(
        [attn, q_grad, k_grad, v_grad],
        [expected_out, expected_q_grad, expected_k_grad, expected_v_grad],
    ):
        assert_close(actual, expected)


@pytest.mark.mpi
def test_privatize_squeeze(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor((-1, -1, -1), dtype=DataType.BFloat16)
        tv1 = fd.ops.broadcast(inp, [True, False, False, False])
        tv2 = fd.ops.squeeze(tv1, dims=[0])
        tv3 = fd.ops.cast(tv2, dtype=DataType.Float)
        tv4 = fd.ops.sum(tv3, dims=[0])
        tv5 = fd.ops.sum(tv3, dims=[1])
        fd.add_output(tv4)
        fd.add_output(tv5)

    def _multidevice_schedule(fd: FusionDefinition):
        for t in fd.fusion.inputs():
            t.set_device_mesh(mesh)
            t.split(0, d, inner_split=False)
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)

    unsharded = torch.randn(d * 3, 5, 6, dtype=torch.bfloat16)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 0, mesh)
    out1, out2 = fd.execute([sharded])
    torch.testing.assert_close(out1.cpu(), unsharded.to(torch.float).sum(0))
    torch.testing.assert_close(out2, sharded.to(torch.float).sum(1))


@pytest.mark.mpi
def test_inner_reduction(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    torch.cuda.set_device(multidevice_direct_test.local_rank)

    def _definition(fd: FusionDefinition) -> None:
        inp = fd.define_tensor((-1, -1), dtype=DataType.Float)
        sum_inner = fd.ops.sum(inp, [1])
        fd.add_output(sum_inner)

    def _multidevice_schedule(fd: FusionDefinition) -> None:
        for t in fd.fusion.inputs():
            t.set_device_mesh(mesh)
            t.split(0, d, inner_split=False)
            t.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    unsharded = torch.ones(d * 3, 5)
    sharded = multidevice_direct_test.shard_tensor(unsharded, 0, mesh)

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)
    ref_out = sharded.sum(1)
    (out,) = fd.execute([sharded])
    assert torch.allclose(ref_out, out)
