# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

import mpi_fixtures
import nvfuser
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

    fn = Model()
    outputs = fn.execute([sharded_input])
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

    fn = Model(d, b, s, h)
    out_tensors = fn.execute([inp_tensor, weight_tensor, bias_tensor])

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


def transformer_forward_fusion(fd: FusionDefinition) -> None:
    # MHA dropout.rng_offset
    S0 = fd.define_scalar(None, dtype=DataType.Int)
    # MHA dropout.rng_seed
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    # MLP dropout.rng_offset
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    # MLP dropout.rng_seed
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    # x: input
    T4 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
    )
    # layer_norm0.weight
    T5 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # layer_norm0.bias
    T6 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # MHA linear0.weight
    T7 = fd.define_tensor(
        shape=[36864, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
    )
    # MHA linear0.bias
    T8 = fd.define_tensor(
        shape=[36864],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # MHA linear1.weight
    T9 = fd.define_tensor(
        shape=[12288, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
    )
    # MHA linear1.bias
    T10 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # layer_norm1.weight
    T11 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # layer_norm1.bias
    T12 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # MLP linear0.weight
    T13 = fd.define_tensor(
        shape=[49152, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
    )
    # MLP linear0.bias
    T14 = fd.define_tensor(
        shape=[49152],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    # MLP linear1.weight
    T15 = fd.define_tensor(
        shape=[12288, 49152],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
    )
    # MLP linear1.bias
    T16 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
    )
    S17 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S18 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T23 = fd.ops.uniform(
        S17,
        S18,
        shape=[1, 2048, 12288],
        rng_seed=S1,
        rng_offset=S0,
        dtype=DataType.BFloat16,
    )
    S24 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S25 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T30 = fd.ops.uniform(
        S24,
        S25,
        shape=[1, 2048, 12288],
        rng_seed=S3,
        rng_offset=S2,
        dtype=DataType.BFloat16,
    )
    T31 = fd.ops.cast(T4, dtype=DataType.Float)
    S32 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T33 = fd.ops.lt(T23, S32)
    S34 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T35 = fd.ops.lt(T30, S34)
    T36, T37 = fd.ops.var_mean(T31, dims=[2], correction=0, keepdim=False)
    T42 = fd.ops.broadcast_in_dim(T36, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    T47 = fd.ops.broadcast_in_dim(T37, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    S48 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T49 = fd.ops.add(T42, S48)
    T54 = fd.ops.broadcast_in_dim(T47, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2])
    T55 = fd.ops.rsqrt(T49)
    T56 = fd.ops.sub(T31, T54)
    T61 = fd.ops.broadcast_in_dim(T55, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2])
    T62 = fd.ops.mul(T56, T61)
    T67 = fd.ops.broadcast_in_dim(T5, shape=[1, 2048, 12288], broadcast_dims=[2])
    T68 = fd.ops.cast(T67, dtype=DataType.Float)
    T69 = fd.ops.mul(T62, T68)
    T74 = fd.ops.broadcast_in_dim(T6, shape=[1, 2048, 12288], broadcast_dims=[2])
    T75 = fd.ops.cast(T74, dtype=DataType.Float)
    T76 = fd.ops.add(T69, T75)
    T77 = fd.ops.cast(T76, dtype=DataType.BFloat16)
    T78 = fd.ops.linear(T77, T7, T8)
    T91 = fd.ops.slice(
        T78, start_indices=[0, 0, 0], end_indices=[1, 2048, 12288], strides=[1, 1, 1]
    )
    T104 = fd.ops.slice(
        T78,
        start_indices=[0, 0, 12288],
        end_indices=[1, 2048, 24576],
        strides=[1, 1, 1],
    )
    T117 = fd.ops.slice(
        T78,
        start_indices=[0, 0, 24576],
        end_indices=[1, 2048, 36864],
        strides=[1, 1, 1],
    )
    T123 = fd.ops.reshape(T104, new_shape=[1, 2048, 96, 128])
    T124 = fd.ops.permute(T123, dims=[0, 2, 1, 3])
    T130 = fd.ops.reshape(T91, new_shape=[1, 2048, 96, 128])
    T131 = fd.ops.permute(T130, dims=[0, 2, 1, 3])
    T137 = fd.ops.reshape(T117, new_shape=[1, 2048, 96, 128])
    T138 = fd.ops.permute(T137, dims=[0, 2, 1, 3])
    S139 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S140 = fd.define_scalar(True, dtype=DataType.Bool)
    T141, T142, T143, T144 = fd.ops.sdpfa_fwd(T131, T124, T138, S139, S140, None)
    T145 = fd.ops.permute(T141, dims=[0, 2, 1, 3])
    T146 = fd.ops.stride_order(T145, stride_order=[3, 2, 1, 0])
    T151 = fd.ops.reshape(T146, new_shape=[1, 2048, 12288])
    T152 = fd.ops.linear(T151, T9, T10)
    T153 = fd.ops.cast(T152, dtype=DataType.Float)
    T154 = fd.ops.cast(T33, dtype=DataType.Float)
    T155 = fd.ops.mul(T153, T154)
    S156 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T157 = fd.ops.mul(T155, S156)
    T158 = fd.ops.add(T31, T157)
    T159, T160 = fd.ops.var_mean(T158, dims=[2], correction=0, keepdim=False)
    T165 = fd.ops.broadcast_in_dim(T159, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    T170 = fd.ops.broadcast_in_dim(T160, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    S171 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T172 = fd.ops.add(T165, S171)
    T177 = fd.ops.broadcast_in_dim(
        T170, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2]
    )
    T178 = fd.ops.rsqrt(T172)
    T179 = fd.ops.sub(T158, T177)
    T184 = fd.ops.broadcast_in_dim(
        T178, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2]
    )
    T185 = fd.ops.mul(T179, T184)
    T190 = fd.ops.broadcast_in_dim(T11, shape=[1, 2048, 12288], broadcast_dims=[2])
    T191 = fd.ops.cast(T190, dtype=DataType.Float)
    T192 = fd.ops.mul(T185, T191)
    T197 = fd.ops.broadcast_in_dim(T12, shape=[1, 2048, 12288], broadcast_dims=[2])
    T198 = fd.ops.cast(T197, dtype=DataType.Float)
    T199 = fd.ops.add(T192, T198)
    T200 = fd.ops.cast(T199, dtype=DataType.BFloat16)
    T201 = fd.ops.linear(T200, T13, T14)
    T202 = fd.ops.cast(T201, dtype=DataType.Float)
    T203 = fd.ops.mul(T202, T202)
    T204 = fd.ops.mul(T203, T202)
    S205 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T206 = fd.ops.mul(S205, T204)
    T207 = fd.ops.add(T202, T206)
    S208 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T209 = fd.ops.mul(S208, T207)
    T210 = fd.ops.tanh(T209)
    S211 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T212 = fd.ops.mul(S211, T202)
    S213 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T214 = fd.ops.add(S213, T210)
    T215 = fd.ops.mul(T212, T214)
    T216 = fd.ops.cast(T215, dtype=DataType.BFloat16)
    T217 = fd.ops.linear(T216, T15, T16)
    T218 = fd.ops.cast(T217, dtype=DataType.Float)
    T219 = fd.ops.cast(T35, dtype=DataType.Float)
    T220 = fd.ops.mul(T218, T219)
    S221 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T222 = fd.ops.mul(T220, S221)
    T223 = fd.ops.add(T158, T222)
    T224 = fd.ops.cast(T223, dtype=DataType.BFloat16)
    # See the caller of this FusionDefinition for what these outputs mean.
    fd.add_output(T37)
    fd.add_output(T55)
    fd.add_output(T78)
    fd.add_output(T141)
    fd.add_output(T142)
    fd.add_output(T143)
    fd.add_output(T144)
    fd.add_output(T158)
    fd.add_output(T160)
    fd.add_output(T178)
    fd.add_output(T224)


@pytest.mark.mpi
def test_transformer_forward(mpi_test):
    torch.cuda.set_device(mpi_test.local_rank)

    ins = [
        29,
        2142642406458297,
        30,
        2142642406458297,
        torch.randn(25165824, dtype=torch.bfloat16, device="cuda").as_strided(
            (1, 2048, 12288), (25165824, 12288, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
        torch.randn(452984832, dtype=torch.bfloat16, device="cuda").as_strided(
            (36864, 12288), (12288, 1)
        ),
        torch.randn(36864, dtype=torch.bfloat16, device="cuda").as_strided(
            (36864,), (1,)
        ),
        torch.randn(150994944, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288, 12288), (12288, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
        torch.randn(603979776, dtype=torch.bfloat16, device="cuda").as_strided(
            (49152, 12288), (12288, 1)
        ),
        torch.randn(49152, dtype=torch.bfloat16, device="cuda").as_strided(
            (49152,), (1,)
        ),
        torch.randn(603979776, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288, 49152), (49152, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda").as_strided(
            (12288,), (1,)
        ),
    ]

    with FusionDefinition() as fd:
        transformer_forward_fusion(fd)

    outs = fd.execute(ins)
    (
        layernorm0_avg,
        layernorm0_invstd,
        mha_linear0,
        sdpa_out,
        sdpa_logsum_exp,
        sdpa_seed,
        sdpa_offset,
        mha_dropout,
        layernorm1_avg,
        layernorm1_invstd,
        output,
    ) = outs

    def assert_shape_is(t: torch.Tensor, sizes: list[int]) -> None:
        assert t.shape == torch.Size(sizes)

    assert_shape_is(layernorm0_avg, [1, 2048])
    assert_shape_is(layernorm0_invstd, [1, 2048, 1])
    assert_shape_is(mha_linear0, [1, 2048, 36864])
    assert_shape_is(sdpa_out, [1, 96, 2048, 128])
    assert_shape_is(sdpa_logsum_exp, [1, 96, 2048])
    assert_shape_is(sdpa_seed, [])
    assert_shape_is(sdpa_offset, [])
    assert_shape_is(mha_dropout, [1, 2048, 12288])
    assert_shape_is(layernorm1_avg, [1, 2048])
    assert_shape_is(layernorm1_invstd, [1, 2048, 1])
    assert_shape_is(output, [1, 2048, 12288])
