# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from enum import Enum, auto
from torch.nn.attention import SDPBackend

import nvfuser
from nvfuser import DataType, FusionDefinition
from nvfuser.testing.utils import is_pre_ampere


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
    mesh = nvfuser.DeviceMesh(range(num_devices))

    class Model(FusionDefinition):
        def definition(self):
            self.t0 = self.define_tensor(
                (-1, -1), contiguity=False, dtype=DataType.Float
            )
            self.t1 = self.ops.relu(self.t0)
            self.t2 = self.ops.add(self.t1, self.t1)
            self.add_output(self.t2)

        def multidevice_schedule(self):
            self.sched._set_device_mesh(self.t0, mesh)
            self.sched._set_device_mesh(self.t1, mesh)
            self.sched._set_device_mesh(self.t2, mesh)
            self.sched.parallelize(self.t0, 0, nvfuser.ParallelType.mesh_x)

    unsharded_input = torch.randn(num_devices, 4)
    sharded_input = multidevice_test.shard_tensor(unsharded_input, 0, mesh)

    fd = Model()
    (output,), (output_sharding,) = fd.execute([sharded_input])
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
def test_sdpa(multidevice_test, qkv_format: QkvFormat):
    d, b, s, h, e = multidevice_test.size, 2, 1024, 12, 768

    if h % d != 0:
        pytest.skip(f"We only support even split, so {h} has to be divisible by {d}.")

    class Model(FusionDefinition):
        def __init__(self, qkv_format: QkvFormat):
            super().__init__()
            self._qkv_format = qkv_format

        def definition(self) -> None:
            match self._qkv_format:
                case QkvFormat.BHSE:
                    stride_order = [4, 3, 2, 1, 0]
                case QkvFormat.BSHE:
                    stride_order = [4, 3, 1, 2, 0]

            self.q, self.k, self.v, self.out_grad = [
                self.define_tensor(
                    shape=[d, b, h // d, s, e // h],
                    contiguity=True,
                    dtype=DataType.BFloat16,
                    stride_order=stride_order,
                )
                for _ in range(4)
            ]

            # TODO(#3123): support sharded dropout and change this to a
            # positive probability.
            dropout_p = self.define_scalar(0.0, dtype=DataType.Double)
            is_causal = self.define_scalar(True, dtype=DataType.Bool)
            attn, log_sumexp, seed, offset = self.ops.sdpfa_fwd(
                self.q, self.k, self.v, dropout_p, is_causal, scale=None
            )

            q_grad, k_grad, v_grad = self.ops.sdpfa_bwd(
                self.out_grad,
                self.q,
                self.k,
                self.v,
                attn,
                log_sumexp,
                dropout_p,
                is_causal,
                seed,
                offset,
                scale=None,
            )

            self.add_output(attn)
            for grad in [q_grad, k_grad, v_grad]:
                self.add_output(grad)

        def multidevice_schedule(self) -> None:
            mesh = nvfuser.DeviceMesh(range(d))
            for t in [self.q, self.k, self.v, self.out_grad]:
                self.sched._set_device_mesh(t, mesh)
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(multidevice_test.local_rank)

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

    fd = Model(qkv_format)
    outs, _ = fd.execute(
        [
            head_parallelize(q).requires_grad_(),
            head_parallelize(k).requires_grad_(),
            head_parallelize(v).requires_grad_(),
            head_parallelize(out_grad),
        ]
    )
    out, q_grad, k_grad, v_grad = outs

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
    mesh = nvfuser.DeviceMesh(range(d))
    h, e = 12, 768
    if h % d != 0:
        pytest.skip(f"We only support even split, so {h} has to be divisible by {d}.")

    class Model(FusionDefinition):
        def __init__(self, qkv_format: QkvFormat):
            super().__init__()
            self._qkv_format = qkv_format

        def definition(self) -> None:
            match self._qkv_format:
                case QkvFormat.BHSE:
                    stride_order = [3, 2, 1, 0]
                case QkvFormat.BSHE:
                    stride_order = [3, 1, 2, 0]

            self.q, self.k, self.v, self.out_grad = [
                self.define_tensor(
                    shape=[-1, h, -1, e // h],
                    dtype=DataType.BFloat16,
                    stride_order=stride_order,
                )
                for _ in range(4)
            ]

            # TODO(#3123): support sharded dropout and change this to a
            # positive probability.
            dropout_p = self.define_scalar(0.0, dtype=DataType.Double)
            is_causal = self.define_scalar(True, dtype=DataType.Bool)
            self.attn, self.log_sumexp, self.seed, self.offset = self.ops.sdpfa_fwd(
                self.q, self.k, self.v, dropout_p, is_causal, scale=None
            )

            self.q_grad, self.k_grad, self.v_grad = self.ops.sdpfa_bwd(
                self.out_grad,
                self.q,
                self.k,
                self.v,
                self.attn,
                self.log_sumexp,
                dropout_p,
                is_causal,
                self.seed,
                self.offset,
                scale=None,
            )

            self.add_output(self.attn)
            for grad in [self.q_grad, self.k_grad, self.v_grad]:
                self.add_output(grad)

        def multidevice_schedule(self) -> None:
            input_tvs = [self.q, self.k, self.v, self.out_grad]
            output_tvs = [
                self.attn,
                self.log_sumexp,
                self.q_grad,
                self.k_grad,
                self.v_grad,
            ]
            non_sharded_tvs = [self.seed, self.offset]

            for t in input_tvs + output_tvs + non_sharded_tvs:
                self.sched._set_device_mesh(t, mesh)

            # Shard input tensorviews
            for t in input_tvs:
                self.sched.split(t, 1, d, False)
                self.sched.parallelize(t, 1, nvfuser.ParallelType.mesh_x)
                if self._qkv_format == QkvFormat.BSHE:
                    # The loop domain is: {i{B}, i{DIDx}, i{H//D}, i{S}, i{E//H}}
                    # Reorder i{S} in the allocation domain for BHSE: {i{DIDx}, i{B}, i{S}, i{H//D}, i{E//H}}
                    self.sched.reorder(t, {2: 3, 3: 2})
                self.sched.set_allocation_as_loop(t)

            # Propagate sharding to output tvs
            self.sched.transform_like(self.q, output_tvs)
            self.sched.parallelize_like(
                self.q, -1, output_tvs, {nvfuser.ParallelType.mesh_x}
            )

            # Set allocation as loop for output tvs
            for t in output_tvs:
                self.sched.set_allocation_as_loop(t)

    b, s = 2, 1024

    torch.cuda.set_device(multidevice_test.local_rank)

    def make_unsharded_tensor() -> torch.Tensor:
        return torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cpu")

    q, k, v = [make_unsharded_tensor().requires_grad_() for _ in range(3)]
    out_grad = make_unsharded_tensor()
    sharded_q, sharded_k, sharded_v, sharded_out_grad = [
        multidevice_test.shard_tensor(t, 1, mesh) for t in [q, k, v, out_grad]
    ]

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=None
        )
        expected_out.backward(out_grad)
        expected_q_grad, expected_k_grad, expected_v_grad = q.grad, k.grad, v.grad

    fd = Model(qkv_format)

    def reformat_tensor(t: torch.Tensor) -> torch.Tensor:
        match qkv_format:
            case QkvFormat.BHSE:
                return t
            case QkvFormat.BSHE:
                return t.transpose(1, 2).contiguous().transpose(1, 2)

    (attn, q_grad, k_grad, v_grad), _ = fd.execute(
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
            multidevice_test.shard_tensor(expected, 1, mesh),
            rtol=1.6e-2,
            atol=1e-2,
        )

    for actual, expected in zip(
        [attn, q_grad, k_grad, v_grad],
        [expected_out, expected_q_grad, expected_k_grad, expected_v_grad],
    ):
        assert_close(actual, expected)
