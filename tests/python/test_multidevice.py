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


# The following two benchmarks micro-benchmarks the forward pass and the
# backprop of a sharded Transformer block used in GPT-3.
#
# The single-GPU nvFusions are
# dumped from Thunder. To regenerate the nvFusions and the inputs, run the
# following:
#
# 1. `git clone https://github.com/Lightning-AI/lightning-thunder.git`
# 2. `git fetch origin wjy/sharded`
# 3. `git checkout wjy/sharded`
#    This branch adds the GPT-3 block benchmark, turns on certain knobs so the
#    entire Transformer block fits into one nvFusion, and prints out the repro.
# 4. `pytest thunder/benchmarks/targets.py -k 'test_nanogpt_block[backward-thunder]' -s`
#
# In stdout, you'll find the forward nvFusion executed once followed by the
# backward nvFusion executed many times.
#
# For future reference, the nvFusions below are generated with Thunder version
# https://github.com/Lightning-AI/lightning-thunder/commit/30e4aa1e67005c58219d7f06b46836eedb74b27a.
# The Thunder traces are
# https://gist.github.com/wujingyue/b111aa8b8d92067fc6004f5d0488dd27.
#
# Based on the single-GPU nvFusions, more changes are applied to generate the multi-GPU nvFusions.
# 1. Replace magic values with variables for flexibility and readability.
# 2. Split device dimensions and parallelize them.
# 3. Decompose the second linear layer in MLP so the matmul result can be allreduced.
# 4. Rename the inputs and outputs for readability.
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

        self.inp = self.define_tensor(
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
            shape=[d, e * 3 // d, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear0_bias = self.define_tensor(
            shape=[d, e * 3 // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear1_weight = self.define_tensor(
            shape=[d, e, e // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        mha_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        mha_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
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
        mlp_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        mlp_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
        T17 = self.ops.cast(self.inp, dtype=DataType.Float)
        T18, layernorm0_mean = self.ops.var_mean(
            T17, dims=[2], correction=0, keepdim=False
        )
        T24 = self.ops.broadcast_in_dim(T18, shape=[b, s, 1], broadcast_dims=[0, 1])
        T29 = self.ops.broadcast_in_dim(
            layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        S30 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T31 = self.ops.add(T24, S30)
        layernorm0_rstd = self.ops.rsqrt(T31)
        T37 = self.ops.broadcast_in_dim(T29, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T38 = self.ops.sub(T17, T37)
        T43 = self.ops.broadcast_in_dim(
            layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T44 = self.ops.mul(T38, T43)
        T49 = self.ops.broadcast_in_dim(
            self.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T50 = self.ops.cast(T49, dtype=DataType.Float)
        T51 = self.ops.mul(T44, T50)
        T56 = self.ops.broadcast_in_dim(
            self.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T57 = self.ops.cast(T56, dtype=DataType.Float)
        T58 = self.ops.add(T51, T57)
        T59 = self.ops.cast(T58, dtype=DataType.BFloat16)
        mha_linear0_out = self.ops.linear(
            T59, self.mha_linear0_weight, self.mha_linear0_bias
        )
        T73 = self.ops.slice(
            mha_linear0_out,
            start_indices=[0, 0, 0, 0],
            end_indices=[d, b, s, e // d],
        )
        T86 = self.ops.slice(
            mha_linear0_out,
            start_indices=[0, 0, 0, e // d],
            end_indices=[d, b, s, e * 2 // d],
        )
        T99 = self.ops.slice(
            mha_linear0_out,
            start_indices=[0, 0, 0, e * 2 // d],
            end_indices=[d, b, s, e * 3 // d],
        )
        T105 = self.ops.reshape(T86, new_shape=[d, b, s, h // d, e // h])
        T106 = self.ops.permute(T105, dims=[0, 1, 3, 2, 4])
        T112 = self.ops.reshape(T73, new_shape=[d, b, s, h // d, e // h])
        T113 = self.ops.permute(T112, dims=[0, 1, 3, 2, 4])
        T119 = self.ops.reshape(T99, new_shape=[d, b, s, h // d, e // h])
        T120 = self.ops.permute(T119, dims=[0, 1, 3, 2, 4])
        S121 = self.define_scalar(0.100000, dtype=DataType.Double)
        S122 = self.define_scalar(True, dtype=DataType.Bool)
        sdpa_out, sdpa_logsum_exp, sdpa_seed, sdpa_offset = self.ops.sdpfa_fwd(
            T113, T106, T120, S121, S122, None
        )
        T127 = self.ops.permute(sdpa_out, dims=[0, 1, 3, 2, 4])
        T128 = self.ops.stride_order(T127, stride_order=[4, 3, 2, 1, 0])
        T133 = self.ops.reshape(T128, new_shape=[d, b, s, e // d])
        # TODO(#3125): nvFuser is missing an API to construct a sharded linear
        # like this. Therefore, I decomposed it by hand.
        # mha_linear1_out = self.ops.linear(T133, self.mha_linear1_weight, self.mha_linear1_bias)
        #    [b,s,e]                 [d,b,s,e/d]        [d,e,e/d]                 [e]
        mha_linear1_local_matmul = self.ops.matmul(
            T133,
            self.ops.broadcast_in_dim(
                self.ops.permute(self.mha_linear1_weight, [0, 2, 1]),
                [d, 1, e // d, e],
                [0, 2, 3],
            ),
        )
        mha_linear1_matmul = self.ops.sum(mha_linear1_local_matmul, [0])  # allreduce
        mha_linear1_biasadd = self.ops.add(
            mha_linear1_matmul,
            self.ops.broadcast_in_dim(self.mha_linear1_bias, [1, 1, e], [2]),
        )
        mha_linear1_out = self.ops.cast(mha_linear1_biasadd, dtype=DataType.BFloat16)
        S135 = self.define_scalar(0.00000, dtype=DataType.Double)
        S136 = self.define_scalar(1.00000, dtype=DataType.Double)
        T141 = self.ops.uniform(
            S135,
            S136,
            shape=[b, s, e],
            rng_seed=mha_dropout_seed,
            rng_offset=mha_dropout_offset,
            dtype=DataType.BFloat16,
        )
        S142 = self.define_scalar(0.900000, dtype=DataType.Double)
        T143 = self.ops.lt(T141, S142)
        T144 = self.ops.cast(mha_linear1_out, dtype=DataType.Float)
        T145 = self.ops.cast(T143, dtype=DataType.Float)
        T146 = self.ops.mul(T144, T145)
        S147 = self.define_scalar(1.11111, dtype=DataType.Double)
        T148 = self.ops.mul(T146, S147)
        T149 = self.ops.add(T17, T148)
        T150, layernorm1_mean = self.ops.var_mean(
            T149, dims=[2], correction=0, keepdim=False
        )
        T156 = self.ops.broadcast_in_dim(T150, shape=[b, s, 1], broadcast_dims=[0, 1])
        T161 = self.ops.broadcast_in_dim(
            layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        S162 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T163 = self.ops.add(T156, S162)
        layernorm1_rstd = self.ops.rsqrt(T163)
        T169 = self.ops.broadcast_in_dim(
            T161, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T170 = self.ops.sub(T149, T169)
        T175 = self.ops.broadcast_in_dim(
            layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T176 = self.ops.mul(T170, T175)
        T181 = self.ops.broadcast_in_dim(
            self.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T182 = self.ops.cast(T181, dtype=DataType.Float)
        T183 = self.ops.mul(T176, T182)
        T188 = self.ops.broadcast_in_dim(
            self.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T189 = self.ops.cast(T188, dtype=DataType.Float)
        T190 = self.ops.add(T183, T189)
        T191 = self.ops.cast(T190, dtype=DataType.BFloat16)
        mlp_linear0_out = self.ops.linear(
            T191, self.mlp_linear0_weight, self.mlp_linear0_bias
        )
        T193 = self.ops.cast(mlp_linear0_out, dtype=DataType.Float)
        T194 = self.ops.mul(T193, T193)
        T195 = self.ops.mul(T194, T193)
        S196 = self.define_scalar(0.500000, dtype=DataType.Double)
        T197 = self.ops.mul(S196, T193)
        S198 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T199 = self.ops.mul(S198, T195)
        T200 = self.ops.add(T193, T199)
        S201 = self.define_scalar(0.797885, dtype=DataType.Double)
        T202 = self.ops.mul(S201, T200)
        T203 = self.ops.tanh(T202)
        S204 = self.define_scalar(1.00000, dtype=DataType.Double)
        T205 = self.ops.add(S204, T203)
        T206 = self.ops.mul(T197, T205)
        T207 = self.ops.cast(T206, dtype=DataType.BFloat16)
        # TODO(#3125): same as mha_linear1.
        # T208 = self.ops.linear(T207, self.mlp_linear1_weight, self.mlp_linear1_bias)
        # [b,s,e]        [d,b,s,4h/d]        [d,e,4h/d]                  [e]
        T208_local_matmul = self.ops.matmul(
            T207,
            self.ops.broadcast_in_dim(
                self.ops.permute(self.mlp_linear1_weight, [0, 2, 1]),
                [d, 1, e * 4 // d, e],
                [0, 2, 3],
            ),
        )
        T208_matmul = self.ops.sum(T208_local_matmul, [0])
        T208_biasadd = self.ops.add(
            T208_matmul,
            self.ops.broadcast_in_dim(self.mlp_linear1_bias, [1, 1, e], [2]),
        )
        T208 = self.ops.cast(T208_biasadd, dtype=DataType.BFloat16)
        S209 = self.define_scalar(0.00000, dtype=DataType.Double)
        S210 = self.define_scalar(1.00000, dtype=DataType.Double)
        T215 = self.ops.uniform(
            S209,
            S210,
            shape=[b, s, e],
            rng_seed=mlp_dropout_seed,
            rng_offset=mlp_dropout_offset,
            dtype=DataType.BFloat16,
        )
        S216 = self.define_scalar(0.900000, dtype=DataType.Double)
        T217 = self.ops.lt(T215, S216)
        T218 = self.ops.cast(T208, dtype=DataType.Float)
        T219 = self.ops.cast(T217, dtype=DataType.Float)
        T220 = self.ops.mul(T218, T219)
        S221 = self.define_scalar(1.11111, dtype=DataType.Double)
        T222 = self.ops.mul(T220, S221)
        T223 = self.ops.add(T149, T222)
        out = self.ops.cast(T223, dtype=DataType.BFloat16)

        self.add_output(layernorm0_mean)
        self.add_output(layernorm0_rstd)
        self.add_output(mha_linear0_out)
        self.add_output(sdpa_out)
        self.add_output(sdpa_logsum_exp)
        self.add_output(sdpa_seed)
        self.add_output(sdpa_offset)
        self.add_output(mha_linear1_out)
        self.add_output(layernorm1_mean)
        self.add_output(layernorm1_rstd)
        self.add_output(mlp_linear0_out)
        self.add_output(out)

    def multidevice_schedule(self):
        mesh = self.sched._create_device_mesh(range(self._num_devices))
        # Assign the mesh to inputs and weights. nvFuser will propagate it to
        # downstream tensors.
        for in_tv in [
            self.inp,
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

        # Parallelize the device dimension of certain weights. nvFuser will try
        # to propagate shardings to downstream tensors.
        for in_tv in [
            self.mha_linear0_weight,
            self.mha_linear0_bias,
            self.mha_linear1_weight,
            self.mlp_linear0_weight,
            self.mlp_linear0_bias,
            self.mlp_linear1_weight,
        ]:
            self.sched.parallelize(in_tv, 0, nvfuser.ParallelType.mesh_x)


# TODO(#2962): validate the numbers as well. Currently, the numbers are off
# by a lot, making comparison infeasible.
def _assert_shape_dtype(
    t: torch.Tensor, expected_sizes: list[int], expected_dtype: torch.dtype
) -> None:
    assert t.shape == torch.Size(expected_sizes)
    assert t.dtype == expected_dtype


@pytest.mark.skipif(
    utils.is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_forward(mpi_test, benchmark):
    d = mpi_test.size
    rank = mpi_test.rank

    b, s, h, e = 1, 2048, 96, 12288

    assert (
        e % h == 0
    ), f"The hidden size ({e}) has to be divisible by the number of heads ({h})."

    if h % d != 0:
        pytest.skip(
            f"We only support even DID split, so the number of heads ({h}) has to be divisible by the number of GPUs ({d})."
        )

    assert e * 4 % d == 0, (
        "This is required to evenly DID split MLP. This condition is implied "
        "by the previous two checks; a fail would indicate a programming "
        "error. So I use `assert` instead of `pytest.skip`."
    )

    torch.cuda.set_device(mpi_test.local_rank)

    # To reduce memory footprint, create unsharded data on CPU and copy only
    # the needed slice to GPU.
    mha_linear0_weight = torch.randn(d, e * 3 // d, e, dtype=torch.bfloat16)
    mha_linear0_bias = torch.randn(d, e * 3 // d, dtype=torch.bfloat16)
    mha_linear1_weight = torch.randn(d, e, e // d, dtype=torch.bfloat16)
    mha_linear1_bias = torch.randn(e, dtype=torch.bfloat16, device="cuda")
    mlp_linear0_weight = torch.randn(d, e * 4 // d, e, dtype=torch.bfloat16)
    mlp_linear0_bias = torch.randn(d, e * 4 // d, dtype=torch.bfloat16)
    mlp_linear1_weight = torch.randn(d, e, e * 4 // d, dtype=torch.bfloat16)
    mlp_linear1_bias = torch.randn(e, dtype=torch.bfloat16, device="cuda")
    # See TransformerForwardFusion.definition for the meanings of these
    # arguments. They are passed in in the same order as the `define_scalar`s
    # and `define_tensor`s.
    ins = [
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        mha_linear0_weight[rank : rank + 1].cuda(),
        mha_linear0_bias[rank : rank + 1].cuda(),
        mha_linear1_weight[rank : rank + 1].cuda(),
        mha_linear1_bias,
        29,
        8338718769759788,
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        mlp_linear0_weight[rank : rank + 1].cuda(),
        mlp_linear0_bias[rank : rank + 1].cuda(),
        mlp_linear1_weight[rank : rank + 1].cuda(),
        mlp_linear1_bias,
        30,
        8338718769759788,
    ]

    fd = TransformerForwardFusion(d, b, s, h, e)

    def benchmark_fn(profile):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        outs = fd.execute(ins)
        torch.cuda.synchronize()

        if profile:
            torch.cuda.cudart().cudaProfilerStop()

        return outs

    # Warm up and validate.
    outs = benchmark_fn(False)
    (
        layernorm0_mean,
        layernorm0_rstd,
        mha_linear0_out,
        sdpa_out,
        sdpa_logsum_exp,
        sdpa_seed,
        sdpa_offset,
        mha_linear1_out,
        layernorm1_mean,
        layernorm1_rstd,
        mlp_linear0_out,
        out,
    ) = outs

    _assert_shape_dtype(layernorm0_mean, [b, s], torch.float32)
    _assert_shape_dtype(layernorm0_rstd, [b, s, 1], torch.float32)
    _assert_shape_dtype(mha_linear0_out, [1, b, s, e * 3 // d], torch.bfloat16)
    _assert_shape_dtype(sdpa_out, [1, b, h // d, s, e // h], torch.bfloat16)
    _assert_shape_dtype(sdpa_logsum_exp, [1, b, h // d, s], torch.float32)
    _assert_shape_dtype(sdpa_seed, [], torch.int64)
    _assert_shape_dtype(sdpa_offset, [], torch.int64)
    _assert_shape_dtype(mha_linear1_out, [b, s, e], torch.bfloat16)
    _assert_shape_dtype(layernorm1_mean, [b, s], torch.float32)
    _assert_shape_dtype(layernorm1_rstd, [b, s, 1], torch.float32)
    _assert_shape_dtype(mlp_linear0_out, [1, b, s, e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(out, [b, s, e], torch.bfloat16)

    # Benchmark and profile. The profile can be collected and displayed using
    # `nsys`. See instructions in test_transformer_engine.py.
    benchmark.pedantic(benchmark_fn, args=(True,), rounds=5)


class TransformerBackwardFusion(FusionDefinition):
    def __init__(self, num_devices, batch, sequence, head, hidden):
        super().__init__()
        self._num_devices = num_devices
        self._batch = batch
        self._sequence = sequence
        self._head = head
        self._hidden = hidden

    def definition(self) -> None:
        b, s, h, e = self._batch, self._sequence, self._head, self._hidden
        self.sdpa_out = self.define_tensor(
            shape=[b, h, s, e // h],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=[3, 1, 2, 0],
        )
        self.mha_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        self.mha_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
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
        self.ln1_mean = self.define_tensor(
            shape=[b, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.inp = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.ln1_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.ln1_rstd = self.define_tensor(
            shape=[b, s, 1],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.ln1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear0_weight = self.define_tensor(
            shape=[e * 4, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear0_bias = self.define_tensor(
            shape=[e * 4],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        self.mlp_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
        self.out_grad = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear1_weight = self.define_tensor(
            shape=[e, e * 4],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.ln0_mean = self.define_tensor(
            shape=[b, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.ln0_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.ln0_rstd = self.define_tensor(
            shape=[b, s, 1],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.ln0_bias = self.define_tensor(
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
        self.mha_log_sumexp = self.define_tensor(
            shape=[b, h, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.mha_sdpa_seed = self.define_tensor(
            shape=[], dtype=DataType.Int, is_cpu=True
        )
        self.mha_sdpa_offset = self.define_tensor(
            shape=[], dtype=DataType.Int, is_cpu=True
        )
        T25 = self.ops.permute(self.sdpa_out, dims=[0, 2, 1, 3])
        T26 = self.ops.stride_order(T25, stride_order=[3, 2, 1, 0])
        S27 = self.define_scalar(0.00000, dtype=DataType.Double)
        S28 = self.define_scalar(1.00000, dtype=DataType.Double)
        T33 = self.ops.uniform(
            S27,
            S28,
            shape=[b, s, e],
            rng_seed=self.mha_dropout_seed,
            rng_offset=self.mha_dropout_offset,
            dtype=DataType.BFloat16,
        )
        T38 = self.ops.reshape(T26, new_shape=[b, s, e])
        S39 = self.define_scalar(0.900000, dtype=DataType.Double)
        T40 = self.ops.lt(T33, S39)
        T41 = self.ops.linear(T38, self.mha_linear1_weight, self.mha_linear1_bias)
        T42 = self.ops.cast(T40, dtype=DataType.Float)
        T43 = self.ops.cast(T41, dtype=DataType.Float)
        T44 = self.ops.mul(T43, T42)
        T49 = self.ops.broadcast_in_dim(
            self.ln1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        S50 = self.define_scalar(1.11111, dtype=DataType.Double)
        T51 = self.ops.mul(T44, S50)
        T52 = self.ops.cast(self.inp, dtype=DataType.Float)
        T57 = self.ops.broadcast_in_dim(T49, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T58 = self.ops.add(T52, T51)
        T63 = self.ops.broadcast_in_dim(
            self.ln1_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T68 = self.ops.broadcast_in_dim(
            self.ln1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T69 = self.ops.sub(T58, T57)
        T74 = self.ops.broadcast_in_dim(
            self.ln1_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T75 = self.ops.cast(T63, dtype=DataType.Float)
        T76 = self.ops.mul(T69, T68)
        T77 = self.ops.cast(T74, dtype=DataType.Float)
        T78 = self.ops.mul(T76, T75)
        T79 = self.ops.add(T78, T77)
        T80 = self.ops.cast(T79, dtype=DataType.BFloat16)
        T81 = self.ops.linear(T80, self.mlp_linear0_weight, self.mlp_linear0_bias)
        S82 = self.define_scalar(0.00000, dtype=DataType.Double)
        S83 = self.define_scalar(1.00000, dtype=DataType.Double)
        T88 = self.ops.uniform(
            S82,
            S83,
            shape=[b, s, e],
            rng_seed=self.mlp_dropout_seed,
            rng_offset=self.mlp_dropout_offset,
            dtype=DataType.BFloat16,
        )
        T89 = self.ops.cast(T81, dtype=DataType.Float)
        T90 = self.ops.cast(self.out_grad, dtype=DataType.Float)
        S91 = self.define_scalar(0.900000, dtype=DataType.Double)
        T92 = self.ops.lt(T88, S91)
        T93 = self.ops.mul(T89, T89)
        S94 = self.define_scalar(1.11111, dtype=DataType.Double)
        T95 = self.ops.mul(S94, T90)
        T96 = self.ops.cast(T92, dtype=DataType.Float)
        T97 = self.ops.mul(T93, T89)
        T98 = self.ops.mul(T96, T95)
        S99 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T100 = self.ops.mul(S99, T97)
        T101 = self.ops.cast(T98, dtype=DataType.BFloat16)
        T102 = self.ops.add(T89, T100)
        T106 = self.ops.reshape(T101, new_shape=[b * s, e])
        S107 = self.define_scalar(0.797885, dtype=DataType.Double)
        T108 = self.ops.mul(S107, T102)
        T109 = self.ops.matmul(T106, self.mlp_linear1_weight)
        T110 = self.ops.tanh(T108)
        T115 = self.ops.reshape(T109, new_shape=[b, s, e * 4])
        T116 = self.ops.mul(T110, T110)
        T117 = self.ops.cast(T115, dtype=DataType.Float)
        S118 = self.define_scalar(0.500000, dtype=DataType.Double)
        T119 = self.ops.mul(S118, T89)
        S120 = self.define_scalar(1.00000, dtype=DataType.Double)
        T121 = self.ops.sub(S120, T116)
        T122 = self.ops.mul(T119, T117)
        T123 = self.ops.mul(T122, T121)
        S124 = self.define_scalar(1.00000, dtype=DataType.Double)
        T125 = self.ops.add(S124, T110)
        S126 = self.define_scalar(0.797885, dtype=DataType.Double)
        T127 = self.ops.mul(S126, T123)
        T128 = self.ops.mul(T125, T117)
        S129 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T130 = self.ops.mul(S129, T127)
        S131 = self.define_scalar(0.500000, dtype=DataType.Double)
        T132 = self.ops.mul(S131, T128)
        T133 = self.ops.mul(T89, T130)
        T134 = self.ops.mul(T93, T130)
        T135 = self.ops.add(T127, T132)
        T136 = self.ops.mul(T89, T133)
        T137 = self.ops.add(T135, T134)
        T138 = self.ops.add(T137, T136)
        T139 = self.ops.add(T138, T136)
        T140 = self.ops.cast(T139, dtype=DataType.BFloat16)
        T144 = self.ops.reshape(T140, new_shape=[b * s, e * 4])
        T145 = self.ops.matmul(T144, self.mlp_linear0_weight)
        T150 = self.ops.reshape(T145, new_shape=[b, s, e])
        T151 = self.ops.cast(T150, dtype=DataType.Float)
        T152 = self.ops.mul(T75, T151)
        T153 = self.ops.mul(T69, T152)
        T154 = self.ops.sum(T153, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T159 = self.ops.broadcast_in_dim(T154, shape=[b, s, 1], broadcast_dims=[1])
        S160 = self.define_scalar(3.00000, dtype=DataType.Double)
        T161 = self.ops.pow(self.ln1_rstd, S160)
        S162 = self.define_scalar(-0.500000, dtype=DataType.Double)
        T163 = self.ops.mul(S162, T159)
        T164 = self.ops.mul(T68, T152)
        T165 = self.ops.mul(T163, T161)
        T166 = self.ops.neg(T164)
        T167 = self.ops.sum(T165, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T168 = self.ops.sum(T166, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T172 = self.ops.broadcast_in_dim(T167, shape=[b, s], broadcast_dims=[1])
        T177 = self.ops.broadcast_in_dim(T168, shape=[b, s, 1], broadcast_dims=[1])
        T182 = self.ops.broadcast_in_dim(
            self.ln1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T187 = self.ops.broadcast_in_dim(T172, shape=[b, s, 1], broadcast_dims=[0, 1])
        T188 = self.ops.sum(T177, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T193 = self.ops.broadcast_in_dim(
            T182, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T198 = self.ops.broadcast_in_dim(
            T187, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T202 = self.ops.broadcast_in_dim(T188, shape=[b, s], broadcast_dims=[1])
        T203 = self.ops.sub(T58, T193)
        S204 = self.define_scalar(2.00000, dtype=DataType.Double)
        T205 = self.ops.mul(S204, T198)
        T210 = self.ops.broadcast_in_dim(T202, shape=[b, s, 1], broadcast_dims=[0, 1])
        T211 = self.ops.mul(T205, T203)
        T216 = self.ops.broadcast_in_dim(
            T210, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        S217 = self.define_scalar(e, dtype=DataType.Double)
        S218 = self.ops.reciprocal(S217)
        T219 = self.ops.mul(T211, S218)
        S220 = self.define_scalar(1 / e, dtype=DataType.Double)
        T221 = self.ops.mul(S220, T216)
        T226 = self.ops.broadcast_in_dim(
            self.ln0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T227 = self.ops.add(T221, T219)
        T232 = self.ops.broadcast_in_dim(
            T226, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T233 = self.ops.add(T164, T227)
        T238 = self.ops.broadcast_in_dim(
            self.ln0_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T243 = self.ops.broadcast_in_dim(
            self.ln0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T244 = self.ops.sub(T52, T232)
        T245 = self.ops.add(T90, T233)
        T250 = self.ops.broadcast_in_dim(
            self.ln0_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T251 = self.ops.cast(T238, dtype=DataType.Float)
        T252 = self.ops.mul(T244, T243)
        S253 = self.define_scalar(1.11111, dtype=DataType.Double)
        T254 = self.ops.mul(S253, T245)
        T255 = self.ops.cast(T250, dtype=DataType.Float)
        T256 = self.ops.mul(T252, T251)
        T257 = self.ops.mul(T42, T254)
        T258 = self.ops.add(T256, T255)
        T259 = self.ops.cast(T257, dtype=DataType.BFloat16)
        T260 = self.ops.cast(T258, dtype=DataType.BFloat16)
        T264 = self.ops.reshape(T259, new_shape=[b * s, e])
        T265 = self.ops.linear(T260, self.mha_linear0_weight, self.mha_linear0_bias)
        T266 = self.ops.matmul(T264, self.mha_linear1_weight)
        T279 = self.ops.slice(
            T265,
            start_indices=[0, 0, e * 2],
            end_indices=[b, s, e * 3],
            strides=[1, 1, 1],
        )
        T292 = self.ops.slice(
            T265,
            start_indices=[0, 0, e],
            end_indices=[b, s, e * 2],
            strides=[1, 1, 1],
        )
        T305 = self.ops.slice(
            T265,
            start_indices=[0, 0, 0],
            end_indices=[b, s, e],
            strides=[1, 1, 1],
        )
        T310 = self.ops.reshape(T266, new_shape=[b, s, e])
        T316 = self.ops.reshape(T279, new_shape=[b, s, h, e // h])
        T322 = self.ops.reshape(T292, new_shape=[b, s, h, e // h])
        T328 = self.ops.reshape(T305, new_shape=[b, s, h, e // h])
        T334 = self.ops.reshape(T310, new_shape=[b, s, h, e // h])
        T335 = self.ops.permute(T316, dims=[0, 2, 1, 3])
        T336 = self.ops.permute(T322, dims=[0, 2, 1, 3])
        T337 = self.ops.permute(T328, dims=[0, 2, 1, 3])
        T338 = self.ops.permute(T334, dims=[0, 2, 1, 3])
        S339 = self.define_scalar(0.100000, dtype=DataType.Double)
        S340 = self.define_scalar(True, dtype=DataType.Bool)
        T341, T342, T343 = self.ops.sdpfa_bwd(
            T338,
            T337,
            T336,
            T335,
            self.sdpa_out,
            self.mha_log_sumexp,
            S339,
            S340,
            self.mha_sdpa_seed,
            self.mha_sdpa_offset,
            None,
        )
        T344 = self.ops.permute(T343, dims=[0, 2, 1, 3])
        T345 = self.ops.permute(T342, dims=[0, 2, 1, 3])
        T346 = self.ops.permute(T341, dims=[0, 2, 1, 3])
        T351 = self.ops.reshape(T344, new_shape=[b, s, e])
        T356 = self.ops.reshape(T345, new_shape=[b, s, e])
        T361 = self.ops.reshape(T346, new_shape=[b, s, e])
        T362 = self.ops.cat([T361, T356, T351], dim=2)
        T366 = self.ops.reshape(T362, new_shape=[b * s, e * 3])
        T367 = self.ops.matmul(T366, self.mha_linear0_weight)
        T372 = self.ops.reshape(T367, new_shape=[b, s, e])
        T373 = self.ops.cast(T372, dtype=DataType.Float)
        T374 = self.ops.mul(T251, T373)
        T375 = self.ops.mul(T244, T374)
        T376 = self.ops.sum(T375, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T381 = self.ops.broadcast_in_dim(T376, shape=[b, s, 1], broadcast_dims=[1])
        S382 = self.define_scalar(3.00000, dtype=DataType.Double)
        T383 = self.ops.pow(self.ln0_rstd, S382)
        S384 = self.define_scalar(-0.500000, dtype=DataType.Double)
        T385 = self.ops.mul(S384, T381)
        T386 = self.ops.mul(T243, T374)
        T387 = self.ops.mul(T385, T383)
        T388 = self.ops.neg(T386)
        T389 = self.ops.sum(T387, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T390 = self.ops.sum(T388, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T394 = self.ops.broadcast_in_dim(T389, shape=[b, s], broadcast_dims=[1])
        T399 = self.ops.broadcast_in_dim(T390, shape=[b, s, 1], broadcast_dims=[1])
        T404 = self.ops.broadcast_in_dim(
            self.ln0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T409 = self.ops.broadcast_in_dim(T394, shape=[b, s, 1], broadcast_dims=[0, 1])
        T410 = self.ops.sum(T399, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T415 = self.ops.broadcast_in_dim(
            T404, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T420 = self.ops.broadcast_in_dim(
            T409, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T424 = self.ops.broadcast_in_dim(T410, shape=[b, s], broadcast_dims=[1])
        T425 = self.ops.sub(T52, T415)
        S426 = self.define_scalar(2.00000, dtype=DataType.Double)
        T427 = self.ops.mul(S426, T420)
        T432 = self.ops.broadcast_in_dim(T424, shape=[b, s, 1], broadcast_dims=[0, 1])
        T433 = self.ops.mul(T427, T425)
        T438 = self.ops.broadcast_in_dim(
            T432, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        S439 = self.define_scalar(e, dtype=DataType.Double)
        S440 = self.ops.reciprocal(S439)
        T441 = self.ops.mul(T433, S440)
        S442 = self.define_scalar(1 / e, dtype=DataType.Double)
        T443 = self.ops.mul(S442, T438)
        T444 = self.ops.add(T443, T441)
        T445 = self.ops.mul(T119, T125)
        T446 = self.ops.add(T386, T444)
        T447 = self.ops.mul(T252, T373)
        T448 = self.ops.cast(T362, dtype=DataType.Float)
        T449 = self.ops.mul(T76, T151)
        T450 = self.ops.cast(T445, dtype=DataType.BFloat16)
        T451 = self.ops.add(T245, T446)
        T452 = self.ops.sum(T447, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T453 = self.ops.sum(T373, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T454 = self.ops.sum(T448, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T458 = self.ops.reshape(T260, new_shape=[b * s, e])
        T459 = self.ops.permute(T366, dims=[1, 0])
        T460 = self.ops.sum(T257, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T464 = self.ops.reshape(T38, new_shape=[b * s, e])
        T465 = self.ops.permute(T264, dims=[1, 0])
        T466 = self.ops.sum(T449, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T467 = self.ops.sum(T151, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T468 = self.ops.sum(T139, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T472 = self.ops.reshape(T80, new_shape=[b * s, e])
        T473 = self.ops.permute(T144, dims=[1, 0])
        T474 = self.ops.sum(T98, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T478 = self.ops.reshape(T450, new_shape=[b * s, e * 4])
        T479 = self.ops.permute(T106, dims=[1, 0])
        inp_grad = self.ops.cast(T451, dtype=DataType.BFloat16)
        ln0_weight = self.ops.cast(T452, dtype=DataType.BFloat16)
        ln0_bias = self.ops.cast(T453, dtype=DataType.BFloat16)
        mha_linear0_bias_grad = self.ops.cast(T454, dtype=DataType.BFloat16)
        mha_linear0_weight_grad = self.ops.matmul(T459, T458)
        mha_linear1_bias_grad = self.ops.cast(T460, dtype=DataType.BFloat16)
        mha_linear1_weight_grad = self.ops.matmul(T465, T464)
        ln1_weight = self.ops.cast(T466, dtype=DataType.BFloat16)
        ln1_bias = self.ops.cast(T467, dtype=DataType.BFloat16)
        mlp_linear0_bias_grad = self.ops.cast(T468, dtype=DataType.BFloat16)
        mlp_linear0_weight_grad = self.ops.matmul(T473, T472)
        mlp_linear1_bias_grad = self.ops.cast(T474, dtype=DataType.BFloat16)
        mlp_linear1_weight_grad = self.ops.matmul(T479, T478)
        self.add_output(mlp_linear1_weight_grad)
        self.add_output(mlp_linear1_bias_grad)
        self.add_output(mlp_linear0_weight_grad)
        self.add_output(mlp_linear0_bias_grad)
        self.add_output(ln1_bias)
        self.add_output(ln1_weight)
        self.add_output(mha_linear1_weight_grad)
        self.add_output(mha_linear1_bias_grad)
        self.add_output(mha_linear0_weight_grad)
        self.add_output(mha_linear0_bias_grad)
        self.add_output(ln0_bias)
        self.add_output(ln0_weight)
        self.add_output(inp_grad)

    def multidevice_schedule(self):
        mesh = self.sched._create_device_mesh(range(self._num_devices))
        for in_tv in [
            self.sdpa_out,
            self.mha_linear1_weight,
            self.mha_linear1_bias,
            self.ln1_mean,
            self.inp,
            self.ln1_weight,
            self.ln1_rstd,
            self.ln1_bias,
            self.mlp_linear0_weight,
            self.mlp_linear0_bias,
            self.out_grad,
            self.mlp_linear1_weight,
            self.ln0_mean,
            self.ln0_weight,
            self.ln0_rstd,
            self.ln0_bias,
            self.mha_linear0_weight,
            self.mha_linear0_bias,
            self.mha_log_sumexp,
        ]:
            self.sched._set_device_mesh(in_tv, mesh)


@pytest.mark.skipif(
    utils.is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(mpi_test):
    d = mpi_test.size
    rank = mpi_test.rank

    b, s, h, e = 2, 2048, 96, 12288

    # FIXME: skip if certain dimensions are not divisible by d.

    torch.cuda.set_device(mpi_test.local_rank)

    ins = [
        torch.randn(b * s * e, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (b, h, s, e // h), (s * e, e // h, e, 1)
        ),
        29,
        2644496055549444,
        torch.testing.make_tensor((e, e), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e * 4, e), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e * 4,), dtype=torch.bfloat16, device="cuda:0"),
        30,
        2644496055549444,
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e, e * 4), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e * 3, e), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((e * 3,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((b, h, s), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
    ]

    fd = TransformerBackwardFusion(d, b, s, h, e)

    outs = fd.execute(ins)
    (
        mlp_linear1_weight_grad,
        mlp_linear1_bias_grad,
        mlp_linear0_weight_grad,
        mlp_linear0_bias_grad,
        ln1_bias,
        ln1_weight,
        mha_linear1_weight_grad,
        mha_linear1_bias_grad,
        mha_linear0_weight_grad,
        mha_linear0_bias_grad,
        ln0_bias,
        ln0_weight,
        inp_grad,
    ) = outs
    _assert_shape_dtype(mlp_linear1_weight_grad, [e, e * 4], torch.bfloat16)
    _assert_shape_dtype(mlp_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_weight_grad, [e * 4, e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_bias_grad, [e * 4], torch.bfloat16)
    _assert_shape_dtype(ln1_bias, [e], torch.bfloat16)
    _assert_shape_dtype(ln1_weight, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_weight_grad, [e, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_weight_grad, [e * 3, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_bias_grad, [e * 3], torch.bfloat16)
    _assert_shape_dtype(ln0_bias, [e], torch.bfloat16)
    _assert_shape_dtype(ln0_weight, [e], torch.bfloat16)
    _assert_shape_dtype(inp_grad, [b, s, e], torch.bfloat16)
