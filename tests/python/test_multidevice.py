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
            d, b, s, e = self._num_devices, self._batch, self._sequence, self._hidden
            self.inp = self.define_tensor([b, s, e])
            self.weight = self.define_tensor([d, e, e], contiguity=[True, True, True])
            self.bias = self.define_tensor([d, e], contiguity=[True, True])
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

    b, s, e = 2, 1024, 768
    torch.manual_seed(0)
    inp_tensor = torch.randn(b, s, e, device="cuda")
    unsharded_weight_tensor = torch.randn(d * e, e, device="cuda")
    weight_tensor = unsharded_weight_tensor.view([d, e, e])[rank : rank + 1]
    unsharded_bias_tensor = torch.randn(d * e, device="cuda")
    bias_tensor = unsharded_bias_tensor.view([d, e])[rank : rank + 1]

    fd = Model(d, b, s, e)
    out_tensors = fd.execute([inp_tensor, weight_tensor, bias_tensor])

    # [b, s, d*e]
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor, unsharded_weight_tensor, unsharded_bias_tensor
    )
    expected_out_tensor = unsharded_out_tensor.view([b, s, d, e]).permute(2, 0, 1, 3)[
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
    d, b, s, h, e = mpi_test.size, 2, 1024, 12, 768

    if h % d != 0:
        pytest.skip(f"We only support even split, so {h} has to be divisible by {d}.")

    class Model(FusionDefinition):
        def definition(self) -> None:
            self.q = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=True,
                dtype=DataType.BFloat16,
            )
            self.k = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=True,
                dtype=DataType.BFloat16,
            )
            self.v = self.define_tensor(
                shape=[d, -1, -1, -1, -1],
                contiguity=True,
                dtype=DataType.BFloat16,
            )
            # TODO(#3123): support sharded dropout and change this to a
            # positive probability.
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
        torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )

    rank = mpi_test.rank

    # Head-parallelize Q, K, V or the attention output of an SDPA.
    def head_parallelize(t: torch.Tensor) -> torch.Tensor:
        assert t.shape == torch.Size([b, h, s, e // h])
        return t.view([b, d, h // d, s, e // h]).transpose(0, 1)[rank : rank + 1]

    fd = Model()
    attn = fd.execute([head_parallelize(q), head_parallelize(k), head_parallelize(v)])[
        0
    ]
    # Use the default rtol for bfloat16 and a relaxed atol.
    torch.testing.assert_close(
        attn, head_parallelize(expected_attn), rtol=1.6e-2, atol=1e-3
    )


# This is ported over from https://github.com/NVIDIA/Fuser/blob/0d33366bf69393bfcc626e28d15dc830609aae2d/benchmarks/python/test_transformer.py#L306. The major changes are:
# 1. Replace magic values with variables for flexibility and readability.
# 2. Split device dimensions and parallelize them.
# 3. Decompose the second linear layer in MLP so the matmul result can be allreduced.
# 4. Rename the receiving variables for `define_scalar`s and `define_tensor`s for readability.
class TransformerForwardFusion(FusionDefinition):
    def __init__(self, num_devices, batch, sequence, head, hidden):
        super().__init__()
        self._num_devices = num_devices
        self._batch = batch
        self._sequence = sequence
        self._head = head
        self._hidden = hidden

    def definition(self) -> None:
        # Same notations as in test_multidevice_transformer.cpp.
        d, b, s, h, e = (
            self._num_devices,
            self._batch,
            self._sequence,
            self._head,
            self._hidden,
        )

        mha_dropout_rng_offset = self.define_scalar(None, dtype=DataType.Int)
        mha_dropout_rng_seed = self.define_scalar(None, dtype=DataType.Int)
        mlp_dropout_rng_offset = self.define_scalar(None, dtype=DataType.Int)
        mlp_dropout_rng_seed = self.define_scalar(None, dtype=DataType.Int)
        self.input = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm0_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm0_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear0_weight = self.define_tensor(
            shape=[e * 3, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear0_bias = self.define_tensor(
            shape=[e * 3],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear1_weight = self.define_tensor(
            shape=[e, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear0_weight = self.define_tensor(
            shape=[d, e * 4 // d, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear0_bias = self.define_tensor(
            shape=[d, e * 4 // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear1_weight = self.define_tensor(
            shape=[d, e, e * 4 // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        S17 = self.define_scalar(0.00000, dtype=DataType.Double)
        S18 = self.define_scalar(1.00000, dtype=DataType.Double)
        T23 = self.ops.uniform(
            S17,
            S18,
            shape=[b, s, e],
            rng_seed=mha_dropout_rng_seed,
            rng_offset=mha_dropout_rng_offset,
            dtype=DataType.BFloat16,
        )
        S24 = self.define_scalar(0.00000, dtype=DataType.Double)
        S25 = self.define_scalar(1.00000, dtype=DataType.Double)
        T30 = self.ops.uniform(
            S24,
            S25,
            shape=[b, s, e],
            rng_seed=mlp_dropout_rng_seed,
            rng_offset=mlp_dropout_rng_offset,
            dtype=DataType.BFloat16,
        )
        T31 = self.ops.cast(self.input, dtype=DataType.Float)
        S32 = self.define_scalar(0.900000, dtype=DataType.Double)
        T33 = self.ops.lt(T23, S32)
        S34 = self.define_scalar(0.900000, dtype=DataType.Double)
        T35 = self.ops.lt(T30, S34)
        T36, T37 = self.ops.var_mean(T31, dims=[2], correction=0, keepdim=False)
        T42 = self.ops.broadcast_in_dim(T36, shape=[b, s, 1], broadcast_dims=[0, 1])
        T47 = self.ops.broadcast_in_dim(T37, shape=[b, s, 1], broadcast_dims=[0, 1])
        S48 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T49 = self.ops.add(T42, S48)
        T54 = self.ops.broadcast_in_dim(T47, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T55 = self.ops.rsqrt(T49)
        T56 = self.ops.sub(T31, T54)
        T61 = self.ops.broadcast_in_dim(T55, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T62 = self.ops.mul(T56, T61)
        T67 = self.ops.broadcast_in_dim(
            self.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T68 = self.ops.cast(T67, dtype=DataType.Float)
        T69 = self.ops.mul(T62, T68)
        T74 = self.ops.broadcast_in_dim(
            self.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T75 = self.ops.cast(T74, dtype=DataType.Float)
        T76 = self.ops.add(T69, T75)
        T77 = self.ops.cast(T76, dtype=DataType.BFloat16)
        T78 = self.ops.linear(T77, self.mha_linear0_weight, self.mha_linear0_bias)
        T91 = self.ops.slice(
            T78,
            start_indices=[0, 0, 0],
            end_indices=[b, s, e],
            strides=[1, 1, 1],
        )
        T104 = self.ops.slice(
            T78,
            start_indices=[0, 0, e],
            end_indices=[b, s, e * 2],
            strides=[1, 1, 1],
        )
        T117 = self.ops.slice(
            T78,
            start_indices=[0, 0, e * 2],
            end_indices=[b, s, e * 3],
            strides=[1, 1, 1],
        )
        T123 = self.ops.reshape(T104, new_shape=[b, s, h, e // h])
        T124 = self.ops.permute(T123, dims=[0, 2, 1, 3])
        T130 = self.ops.reshape(T91, new_shape=[b, s, h, e // h])
        T131 = self.ops.permute(T130, dims=[0, 2, 1, 3])
        T137 = self.ops.reshape(T117, new_shape=[b, s, h, e // h])
        T138 = self.ops.permute(T137, dims=[0, 2, 1, 3])
        S139 = self.define_scalar(0.100000, dtype=DataType.Double)
        S140 = self.define_scalar(True, dtype=DataType.Bool)
        T141, T142, T143, T144 = self.ops.sdpfa_fwd(T131, T124, T138, S139, S140, None)
        T145 = self.ops.permute(T141, dims=[0, 2, 1, 3])
        T146 = self.ops.stride_order(T145, stride_order=[3, 2, 1, 0])
        T151 = self.ops.reshape(T146, new_shape=[b, s, e])
        T152 = self.ops.linear(T151, self.mha_linear1_weight, self.mha_linear1_bias)
        T153 = self.ops.cast(T152, dtype=DataType.Float)
        T154 = self.ops.cast(T33, dtype=DataType.Float)
        T155 = self.ops.mul(T153, T154)
        S156 = self.define_scalar(1.11111, dtype=DataType.Double)
        T157 = self.ops.mul(T155, S156)
        T158 = self.ops.add(T31, T157)
        T159, T160 = self.ops.var_mean(T158, dims=[2], correction=0, keepdim=False)
        T165 = self.ops.broadcast_in_dim(T159, shape=[b, s, 1], broadcast_dims=[0, 1])
        T170 = self.ops.broadcast_in_dim(T160, shape=[b, s, 1], broadcast_dims=[0, 1])
        S171 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T172 = self.ops.add(T165, S171)
        T177 = self.ops.broadcast_in_dim(
            T170, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T178 = self.ops.rsqrt(T172)
        T179 = self.ops.sub(T158, T177)
        T184 = self.ops.broadcast_in_dim(
            T178, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T185 = self.ops.mul(T179, T184)
        T190 = self.ops.broadcast_in_dim(
            self.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T191 = self.ops.cast(T190, dtype=DataType.Float)
        T192 = self.ops.mul(T185, T191)
        T197 = self.ops.broadcast_in_dim(
            self.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T198 = self.ops.cast(T197, dtype=DataType.Float)
        T199 = self.ops.add(T192, T198)
        T200 = self.ops.cast(T199, dtype=DataType.BFloat16)
        T201 = self.ops.linear(T200, self.mlp_linear0_weight, self.mlp_linear0_bias)
        T202 = self.ops.cast(T201, dtype=DataType.Float)
        T203 = self.ops.mul(T202, T202)
        T204 = self.ops.mul(T203, T202)
        S205 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T206 = self.ops.mul(S205, T204)
        T207 = self.ops.add(T202, T206)
        S208 = self.define_scalar(0.797885, dtype=DataType.Double)
        T209 = self.ops.mul(S208, T207)
        T210 = self.ops.tanh(T209)
        S211 = self.define_scalar(0.500000, dtype=DataType.Double)
        T212 = self.ops.mul(S211, T202)
        S213 = self.define_scalar(1.00000, dtype=DataType.Double)
        T214 = self.ops.add(S213, T210)
        T215 = self.ops.mul(T212, T214)
        T216 = self.ops.cast(T215, dtype=DataType.BFloat16)
        # TODO(#3125): nvFuser is missing an API to construct a sharded linear
        # like this. Therefore, I decomposed it by hand.
        # T217 = self.ops.linear(T216, self.mlp_linear1_weight, self.mlp_linear1_bias)
        # [b,s,e]        [d,b,s,4h/d]        [d,e,4h/d]                  [e]
        T217_local_matmul = self.ops.matmul(
            T216,
            self.ops.broadcast_in_dim(
                self.ops.permute(self.mlp_linear1_weight, [0, 2, 1]),
                [d, 1, e * 4 // d, e],
                [0, 2, 3],
            ),
        )
        T217_matmul = self.ops.sum(T217_local_matmul, [0])
        T217_biasadd = self.ops.add(
            T217_matmul,
            self.ops.broadcast_in_dim(self.mlp_linear1_bias, [1, 1, e], [2]),
        )
        T217 = self.ops.cast(T217_biasadd, dtype=DataType.BFloat16)
        T218 = self.ops.cast(T217, dtype=DataType.Float)
        T219 = self.ops.cast(T35, dtype=DataType.Float)
        T220 = self.ops.mul(T218, T219)
        S221 = self.define_scalar(1.11111, dtype=DataType.Double)
        T222 = self.ops.mul(T220, S221)
        T223 = self.ops.add(T158, T222)
        T224 = self.ops.cast(T223, dtype=DataType.BFloat16)
        # See the caller of this FusionDefinition for what these outputs mean.
        self.add_output(T37)
        self.add_output(T55)
        self.add_output(T78)
        self.add_output(T141)
        self.add_output(T142)
        self.add_output(T143)
        self.add_output(T144)
        self.add_output(T158)
        self.add_output(T160)
        self.add_output(T178)
        self.add_output(T224)

    def multidevice_schedule(self):
        mesh = self.sched._create_device_mesh(range(self._num_devices))
        for in_tv in [
            self.input,
            self.layernorm0_weight,
            self.layernorm0_bias,
            self.mha_linear0_weight,
            self.mha_linear0_bias,
            self.mha_linear1_weight,
            self.mha_linear1_bias,
            self.layernorm1_weight,
            self.layernorm1_bias,
            self.mlp_linear0_weight,
            self.mlp_linear0_bias,
            self.mlp_linear1_weight,
            self.mlp_linear1_bias,
        ]:
            self.sched._set_device_mesh(in_tv, mesh)

        self.sched.parallelize(self.mlp_linear0_weight, 0, nvfuser.ParallelType.mesh_x)
        self.sched.parallelize(self.mlp_linear0_bias, 0, nvfuser.ParallelType.mesh_x)
        self.sched.parallelize(self.mlp_linear1_weight, 0, nvfuser.ParallelType.mesh_x)


@pytest.mark.skipif(
    utils.is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_forward(mpi_test):
    d = mpi_test.size
    rank = mpi_test.rank

    b, s, h, e = 2, 2048, 96, 12288

    if e * 4 % d != 0:
        pytest.skip(
            f"We only support even split, so {e} * 4 has to be divisible by {d}."
        )

    torch.cuda.set_device(mpi_test.local_rank)

    # To reduce memory footprint, create unsharded data on CPU and copy only
    # the needed slice to GPU.
    mlp_linear0_weight = torch.randn(d, e * 4 // d, e, dtype=torch.bfloat16)
    mlp_linear0_bias = torch.randn(d, e * 4 // d, dtype=torch.bfloat16)
    mlp_linear1_weight = torch.randn(d, e, e * 4 // d, dtype=torch.bfloat16)
    # See TransformerForwardFusion.definition for the meanings of these
    # arguments. They are passed in in the same order as the `define_scalar`s
    # and `define_tensor`s.
    ins = [
        29,
        2142642406458297,
        30,
        2142642406458297,
        torch.randn(b, s, e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e * 3, e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e * 3, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
        mlp_linear0_weight[rank : rank + 1].cuda(),
        mlp_linear0_bias[rank : rank + 1].cuda(),
        mlp_linear1_weight[rank : rank + 1].cuda(),
        torch.randn(e, dtype=torch.bfloat16, device="cuda"),
    ]

    fd = TransformerForwardFusion(d, b, s, h, e)

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

    assert_shape_is(layernorm0_avg, [b, s])
    assert_shape_is(layernorm0_invstd, [b, s, 1])
    assert_shape_is(mha_linear0, [b, s, e * 3])
    assert_shape_is(sdpa_out, [b, h, s, e // h])
    assert_shape_is(sdpa_logsum_exp, [b, h, s])
    assert_shape_is(sdpa_seed, [])
    assert_shape_is(sdpa_offset, [])
    assert_shape_is(mha_dropout, [b, s, e])
    assert_shape_is(layernorm1_avg, [b, s])
    assert_shape_is(layernorm1_invstd, [b, s, 1])
    assert_shape_is(output, [b, s, e])
