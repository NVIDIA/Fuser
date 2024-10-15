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


def _sharded_linear_all_reduce(
    fd: FusionDefinition,
    inp: nvfuser.Tensor,
    weight: nvfuser.Tensor,
    bias: nvfuser.Tensor,
    d: int,
    b: int,
    s: int,
    e_in: int,
    e_out: int,
) -> nvfuser.Tensor:
    assert inp.ndim == 4
    assert weight.ndim == 3
    assert bias.ndim == 1

    # TODO(#3125): nvFuser is missing an API to construct a sharded linear
    # like this. Therefore, I decomposed it by hand.
    #
    # inp: [d,b,s,e_in]
    # weight: [d,e_out,e_in]
    # bias: [e_out]
    # out: [b,s,e_out] = ops.linear(inp, weight, bias)
    if d == 1:
        # Fast path where allreduce isn't needed.
        return fd.ops.linear(fd.ops.squeeze(inp, [0]), fd.ops.squeeze(weight, [0]), bias)

    local_matmul = fd.ops.matmul(
        inp,
        fd.ops.broadcast_in_dim(
            fd.ops.permute(weight, [0, 2, 1]),
            [d, 1, e_in, e_out],
            [0, 2, 3],
        ),
    )
    matmul = fd.ops.sum(local_matmul, [0])  # allreduce
    biasadd = fd.ops.add(
        matmul,
        fd.ops.broadcast_in_dim(bias, [1, 1, e_out], [2]),
    )
    return fd.ops.cast(biasadd, dtype=DataType.BFloat16)


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
        mha_linear1_out = _sharded_linear_all_reduce(
            self,
            T133,
            self.mha_linear1_weight,
            self.mha_linear1_bias,
            d,
            b,
            s,
            e // d,
            e,
        )
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
        T208 = _sharded_linear_all_reduce(
            self,
            T207,
            self.mlp_linear1_weight,
            self.mlp_linear1_bias,
            d,
            b,
            s,
            e * 4 // d,
            e,
        )
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

    # TODO(#2962): validate the numbers as well. Currently, the numbers are off
    # by a lot, making comparison infeasible.
    def assert_shape_dtype(
        t: torch.Tensor, expected_sizes: list[int], expected_dtype: torch.dtype
    ) -> None:
        assert t.shape == torch.Size(expected_sizes)
        assert t.dtype == expected_dtype

    assert_shape_dtype(layernorm0_mean, [b, s], torch.float32)
    assert_shape_dtype(layernorm0_rstd, [b, s, 1], torch.float32)
    assert_shape_dtype(mha_linear0_out, [1, b, s, e * 3 // d], torch.bfloat16)
    assert_shape_dtype(sdpa_out, [1, b, h // d, s, e // h], torch.bfloat16)
    assert_shape_dtype(sdpa_logsum_exp, [1, b, h // d, s], torch.float32)
    assert_shape_dtype(sdpa_seed, [], torch.int64)
    assert_shape_dtype(sdpa_offset, [], torch.int64)
    assert_shape_dtype(mha_linear1_out, [b, s, e], torch.bfloat16)
    assert_shape_dtype(layernorm1_mean, [b, s], torch.float32)
    assert_shape_dtype(layernorm1_rstd, [b, s, 1], torch.float32)
    assert_shape_dtype(mlp_linear0_out, [1, b, s, e * 4 // d], torch.bfloat16)
    assert_shape_dtype(out, [b, s, e], torch.bfloat16)

    # Benchmark and profile. The profile can be collected and displayed using
    # `nsys`. See instructions in test_transformer_engine.py.
    benchmark.pedantic(benchmark_fn, args=(True,), rounds=5)
