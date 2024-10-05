# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch.nn.attention import SDPBackend

import mpi_fixtures
import nvfuser
import utils
from nvfuser import DataType, FusionDefinition


mpi_test = mpi_fixtures.mpi_test


@pytest.mark.mpi
def test_sizes_and_ranks(mpi_test):
    size, rank, local_size, local_rank = (
        mpi_test.size,
        mpi_test.rank,
        mpi_test.local_size,
        mpi_test.local_rank,
    )
    assert size > 0
    assert rank >= 0 and rank < size
    assert local_size > 0
    assert local_rank >= 0 and local_rank < local_size


@pytest.mark.mpi
def test_pointwise(mpi_test):
    num_devices = mpi_test.size
    rank = mpi_test.rank

    torch.cuda.set_device(mpi_test.local_rank)

    # Just so each rank receives the same unsharded input.
    torch.manual_seed(0)
    unsharded_input = torch.randn(num_devices, 4, device="cuda")
    sharded_input = unsharded_input[rank : rank + 1]

    class Model(FusionDefinition):
        def definition(self):
            self.t0 = self.define_tensor(
                (-1, -1), contiguity=(False, False), dtype=DataType.Float
            )
            self.t1 = self.ops.relu(self.t0)
            self.t2 = self.ops.add(self.t1, self.t1)
            self.add_output(self.t2)

        def multidevice_schedule(self):
            mesh = self.sched._create_device_mesh(range(num_devices))
            self.sched._set_device_mesh(self.t0, mesh)
            self.sched._set_device_mesh(self.t1, mesh)
            self.sched._set_device_mesh(self.t2, mesh)
            self.sched.parallelize(self.t0, 0, nvfuser.ParallelType.mesh_x)

    fd = Model()
    outputs = fd.execute([sharded_input])
    torch.testing.assert_close(outputs[0], unsharded_input.relu() * 2)


@pytest.mark.mpi
def test_linear(mpi_test):
    class Model(FusionDefinition):
        def __init__(self, num_devices, batch, sequence, hidden):
            super().__init__()
            self._num_devices = num_devices
            self._batch = batch
            self._sequence = sequence
            self._hidden = hidden

        def definition(self):
            d, b, s, h = self._num_devices, self._batch, self._sequence, self._hidden
            self.inp = self.define_tensor([b, s, h])
            self.weight = self.define_tensor([d, h, h], contiguity=[True, True, True])
            self.bias = self.define_tensor([d, h], contiguity=[True, True])
            out = self.ops.linear(self.inp, self.weight, self.bias)
            self.add_output(out)

        def multidevice_schedule(self):
            mesh = self.sched._create_device_mesh(range(self._num_devices))
            for t in [self.inp, self.weight, self.bias]:
                self.sched._set_device_mesh(t, mesh)
            for t in [self.weight, self.bias]:
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    d = mpi_test.size
    rank = mpi_test.rank

    torch.cuda.set_device(mpi_test.local_rank)

    b, s, h = 2, 1024, 768
    torch.manual_seed(0)
    inp_tensor = torch.randn(b, s, h, device="cuda")
    unsharded_weight_tensor = torch.randn(d * h, h, device="cuda")
    weight_tensor = unsharded_weight_tensor.view([d, h, h])[rank : rank + 1]
    unsharded_bias_tensor = torch.randn(d * h, device="cuda")
    bias_tensor = unsharded_bias_tensor.view([d, h])[rank : rank + 1]

    fd = Model(d, b, s, h)
    out_tensors = fd.execute([inp_tensor, weight_tensor, bias_tensor])

    # [b, s, d*h]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor, unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = unsharded_out_tensor.view([b, s, d, h]).permute(2, 0, 1, 3)[
        rank : rank + 1
    ]
    # rtol is the same as the default for fp32. atol is slightly increased.
    torch.testing.assert_close(
        out_tensors[0], expected_out_tensor, rtol=1.3e-6, atol=1e-4
    )


@pytest.mark.skipif(
    utils.is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_sdpa(mpi_test):
    d, b, s, a, h = mpi_test.size, 2, 1024, 12, 768

    if a % d != 0:
        pytest.skip(f"We only support even split, so {a} has to be divisible by {d}.")

    class Model(FusionDefinition):
        def definition(self) -> None:
            self.q = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=[True if d > 1 else None, True, True, True, True],
                dtype=DataType.BFloat16,
            )
            self.k = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=[True if d > 1 else None, True, True, True, True],
                dtype=DataType.BFloat16,
            )
            self.v = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=[True if d > 1 else None, True, True, True, True],
                dtype=DataType.BFloat16,
            )
            dropout_p = self.define_scalar(0.0, dtype=DataType.Double)
            is_causal = self.define_scalar(True, dtype=DataType.Bool)
            sdpa_result = self.ops.sdpfa_fwd(
                self.q, self.k, self.v, dropout_p, is_causal, scale=None
            )
            attn = sdpa_result[0]
            self.add_output(attn)

        def multidevice_schedule(self) -> None:
            mesh = self.sched._create_device_mesh(range(d))
            for t in [self.q, self.k, self.v]:
                self.sched._set_device_mesh(t, mesh)
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(mpi_test.local_rank)
    torch.manual_seed(0)
    q, k, v = [
        torch.randn(b, a, s, h // a, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ) and torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )

    rank = mpi_test.rank

    # Sequence-parallelize Q, K, V or the attention output of an SDPA.
    def sequence_parallelize(t: torch.Tensor) -> torch.Tensor:
        assert t.shape == torch.Size([b, a, s, h // a])
        return t.view([b, d, a // d, s, h // a]).transpose(0, 1)[rank : rank + 1]

    fd = Model()
    attn = fd.execute(
        [sequence_parallelize(q), sequence_parallelize(k), sequence_parallelize(v)]
    )[0]
    # Use the default rtol for bfloat16 and a relaxed atol.
    torch.testing.assert_close(
        attn, sequence_parallelize(expected_attn), rtol=1.6e-2, atol=1e-3
    )
