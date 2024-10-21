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
            self.q, self.k, self.v, self.out_grad = [
                self.define_tensor(
                    shape=[d, b, h // d, s, e // h],
                    contiguity=True,
                    dtype=DataType.BFloat16,
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
            mesh = self.sched._create_device_mesh(range(d))
            for t in [self.q, self.k, self.v, self.out_grad]:
                self.sched._set_device_mesh(t, mesh)
                self.sched.parallelize(t, 0, nvfuser.ParallelType.mesh_x)

    torch.cuda.set_device(mpi_test.local_rank)
    q, k, v = [
        torch.randn(
            b, h, s, e // h, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        for _ in range(3)
    ]
    out_grad = torch.randn(b, h, s, e // h, dtype=torch.bfloat16, device="cuda")

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=None
        )
        expected_out.backward(out_grad)
        expected_q_grad, expected_k_grad, expected_v_grad = q.grad, k.grad, v.grad

    rank = mpi_test.rank

    # Head-parallelize Q, K, V or the attention output of an SDPA.
    def head_parallelize(t: torch.Tensor) -> torch.Tensor:
        assert t.shape == torch.Size([b, h, s, e // h])
        return (
            t.view([b, d, h // d, s, e // h])
            .transpose(0, 1)
            .contiguous()[rank : rank + 1]
        )

    fd = Model()
    outs = fd.execute(
        [
            head_parallelize(q),
            head_parallelize(k),
            head_parallelize(v),
            head_parallelize(out_grad),
        ]
    )
    out, q_grad, k_grad, v_grad = outs

    def assert_close(x, y):
        # Use the default rtol for bfloat16 and a relaxed atol.
        torch.testing.assert_close(x, y, rtol=1.6e-2, atol=1e-2)

    assert_close(out, head_parallelize(expected_out))
    assert_close(q_grad, head_parallelize(expected_q_grad))
    assert_close(k_grad, head_parallelize(expected_k_grad))
    assert_close(v_grad, head_parallelize(expected_v_grad))


def get_benchmark_fn(func, /, profile: bool):
    def wrapper(*args, **kwargs):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return result

    return wrapper


# Returns two functors, one with profiler off and the other with profiler on.
def get_benchmark_fns(func):
    return get_benchmark_fn(func, profile=False), get_benchmark_fn(func, profile=True)


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

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

    # Warm up and validate.
    outs = warmup_fn()
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
    benchmark.pedantic(benchmark_fn, rounds=5)


# All tensors are replicated to all devices at this moment; future PRs will try
# to shard them.
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

        mlp_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        mlp_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
        self.mlp_linear0_out = self.define_tensor(
            shape=[b, s, e * 4],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
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
        mha_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        mha_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
        self.mha_linear1_out = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear0_weight = self.define_tensor(
            shape=[e * 4, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_mean = self.define_tensor(
            shape=[b, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.inp = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_rstd = self.define_tensor(
            shape=[b, s, 1],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.mha_linear1_weight = self.define_tensor(
            shape=[e, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear0_out = self.define_tensor(
            shape=[b, s, e * 3],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.sdpa_out = self.define_tensor(
            shape=[b, h, s, e // h],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=[3, 1, 2, 0],
        )
        self.sdpa_log_sumexp = self.define_tensor(
            shape=[b, h, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        mha_sdpa_seed = self.define_tensor(shape=[], dtype=DataType.Int, is_cpu=True)
        mha_sdpa_offset = self.define_tensor(shape=[], dtype=DataType.Int, is_cpu=True)
        self.mha_linear0_weight = self.define_tensor(
            shape=[e * 3, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm0_weight = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm0_mean = self.define_tensor(
            shape=[b, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.layernorm0_rstd = self.define_tensor(
            shape=[b, s, 1],
            contiguity=True,
            dtype=DataType.Float,
        )
        self.layernorm0_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.layernorm1_bias = self.define_tensor(
            shape=[e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        S25 = self.define_scalar(0.00000, dtype=DataType.Double)
        S26 = self.define_scalar(1.00000, dtype=DataType.Double)
        T31 = self.ops.uniform(
            S25,
            S26,
            shape=[b, s, e],
            rng_seed=mlp_dropout_seed,
            rng_offset=mlp_dropout_offset,
            dtype=DataType.BFloat16,
        )
        T32 = self.ops.cast(self.mlp_linear0_out, dtype=DataType.Float)
        T33 = self.ops.cast(self.out_grad, dtype=DataType.Float)
        S34 = self.define_scalar(0.900000, dtype=DataType.Double)
        T35 = self.ops.lt(T31, S34)
        T36 = self.ops.mul(T32, T32)
        S37 = self.define_scalar(1.11111, dtype=DataType.Double)
        T38 = self.ops.mul(S37, T33)
        T39 = self.ops.cast(T35, dtype=DataType.Float)
        T40 = self.ops.mul(T36, T32)
        T41 = self.ops.mul(T39, T38)
        S42 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T43 = self.ops.mul(S42, T40)
        T44 = self.ops.cast(T41, dtype=DataType.BFloat16)
        T45 = self.ops.add(T32, T43)
        T49 = self.ops.reshape(T44, new_shape=[b * s, e])
        S50 = self.define_scalar(0.797885, dtype=DataType.Double)
        T51 = self.ops.mul(S50, T45)
        T52 = self.ops.matmul(T49, self.mlp_linear1_weight)
        T53 = self.ops.tanh(T51)
        T58 = self.ops.reshape(T52, new_shape=[b, s, e * 4])
        T59 = self.ops.mul(T53, T53)
        T60 = self.ops.cast(T58, dtype=DataType.Float)
        S61 = self.define_scalar(0.500000, dtype=DataType.Double)
        T62 = self.ops.mul(S61, T32)
        S63 = self.define_scalar(1.00000, dtype=DataType.Double)
        T64 = self.ops.sub(S63, T59)
        T65 = self.ops.mul(T62, T60)
        T66 = self.ops.mul(T65, T64)
        S67 = self.define_scalar(1.00000, dtype=DataType.Double)
        T68 = self.ops.add(S67, T53)
        S69 = self.define_scalar(0.797885, dtype=DataType.Double)
        T70 = self.ops.mul(S69, T66)
        T71 = self.ops.mul(T68, T60)
        S72 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T73 = self.ops.mul(S72, T70)
        S74 = self.define_scalar(0.500000, dtype=DataType.Double)
        T75 = self.ops.mul(S74, T71)
        T76 = self.ops.mul(T32, T73)
        T77 = self.ops.mul(T36, T73)
        T78 = self.ops.add(T70, T75)
        T79 = self.ops.mul(T32, T76)
        T80 = self.ops.add(T78, T77)
        T81 = self.ops.add(T80, T79)
        T82 = self.ops.add(T81, T79)
        S83 = self.define_scalar(0.00000, dtype=DataType.Double)
        S84 = self.define_scalar(1.00000, dtype=DataType.Double)
        T89 = self.ops.uniform(
            S83,
            S84,
            shape=[b, s, e],
            rng_seed=mha_dropout_seed,
            rng_offset=mha_dropout_offset,
            dtype=DataType.BFloat16,
        )
        T90 = self.ops.cast(T82, dtype=DataType.BFloat16)
        S91 = self.define_scalar(0.900000, dtype=DataType.Double)
        T92 = self.ops.lt(T89, S91)
        T96 = self.ops.reshape(T90, new_shape=[b * s, e * 4])
        T97 = self.ops.cast(T92, dtype=DataType.Float)
        T98 = self.ops.cast(self.mha_linear1_out, dtype=DataType.Float)
        T99 = self.ops.matmul(T96, self.mlp_linear0_weight)
        T100 = self.ops.mul(T98, T97)
        T105 = self.ops.reshape(T99, new_shape=[b, s, e])
        T110 = self.ops.broadcast_in_dim(
            self.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T115 = self.ops.broadcast_in_dim(
            self.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        S116 = self.define_scalar(1.11111, dtype=DataType.Double)
        T117 = self.ops.mul(T100, S116)
        T118 = self.ops.cast(self.inp, dtype=DataType.Float)
        T119 = self.ops.cast(T105, dtype=DataType.Float)
        T120 = self.ops.cast(T110, dtype=DataType.Float)
        T125 = self.ops.broadcast_in_dim(
            T115, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T126 = self.ops.add(T118, T117)
        T127 = self.ops.mul(T120, T119)
        T128 = self.ops.sub(T126, T125)
        T129 = self.ops.mul(T128, T127)
        T130 = self.ops.sum(T129, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T135 = self.ops.broadcast_in_dim(T130, shape=[b, s, 1], broadcast_dims=[1])
        T140 = self.ops.broadcast_in_dim(
            self.layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        S141 = self.define_scalar(3.00000, dtype=DataType.Double)
        T142 = self.ops.pow(self.layernorm1_rstd, S141)
        S143 = self.define_scalar(-0.500000, dtype=DataType.Double)
        T144 = self.ops.mul(S143, T135)
        T145 = self.ops.mul(T140, T127)
        T146 = self.ops.mul(T144, T142)
        T147 = self.ops.neg(T145)
        T148 = self.ops.sum(T146, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T149 = self.ops.sum(T147, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T153 = self.ops.broadcast_in_dim(T148, shape=[b, s], broadcast_dims=[1])
        T158 = self.ops.broadcast_in_dim(T149, shape=[b, s, 1], broadcast_dims=[1])
        T163 = self.ops.broadcast_in_dim(
            self.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T168 = self.ops.broadcast_in_dim(T153, shape=[b, s, 1], broadcast_dims=[0, 1])
        T169 = self.ops.sum(T158, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T174 = self.ops.broadcast_in_dim(
            T163, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T179 = self.ops.broadcast_in_dim(
            T168, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T183 = self.ops.broadcast_in_dim(T169, shape=[b, s], broadcast_dims=[1])
        T184 = self.ops.sub(T126, T174)
        S185 = self.define_scalar(2.00000, dtype=DataType.Double)
        T186 = self.ops.mul(S185, T179)
        T191 = self.ops.broadcast_in_dim(T183, shape=[b, s, 1], broadcast_dims=[0, 1])
        T192 = self.ops.mul(T186, T184)
        T197 = self.ops.broadcast_in_dim(
            T191, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        S198 = self.define_scalar(e, dtype=DataType.Double)
        S199 = self.ops.reciprocal(S198)
        T200 = self.ops.mul(T192, S199)
        S201 = self.define_scalar(1 / e, dtype=DataType.Double)
        T202 = self.ops.mul(S201, T197)
        T203 = self.ops.add(T202, T200)
        T204 = self.ops.add(T145, T203)
        T205 = self.ops.add(T33, T204)
        S206 = self.define_scalar(1.11111, dtype=DataType.Double)
        T207 = self.ops.mul(S206, T205)
        T208 = self.ops.mul(T97, T207)
        T209 = self.ops.cast(T208, dtype=DataType.BFloat16)
        T213 = self.ops.reshape(T209, new_shape=[b * s, e])
        T214 = self.ops.matmul(T213, self.mha_linear1_weight)
        T227 = self.ops.slice(
            self.mha_linear0_out,
            start_indices=[0, 0, e * 2],
            end_indices=[b, s, e * 3],
        )
        T240 = self.ops.slice(
            self.mha_linear0_out,
            start_indices=[0, 0, e],
            end_indices=[b, s, e * 2],
        )
        T253 = self.ops.slice(
            self.mha_linear0_out,
            start_indices=[0, 0, 0],
            end_indices=[b, s, e],
        )
        T258 = self.ops.reshape(T214, new_shape=[b, s, e])
        T264 = self.ops.reshape(T227, new_shape=[b, s, h, e // h])
        T270 = self.ops.reshape(T240, new_shape=[b, s, h, e // h])
        T276 = self.ops.reshape(T253, new_shape=[b, s, h, e // h])
        T282 = self.ops.reshape(T258, new_shape=[b, s, h, e // h])
        T283 = self.ops.permute(T264, dims=[0, 2, 1, 3])
        T284 = self.ops.permute(T270, dims=[0, 2, 1, 3])
        T285 = self.ops.permute(T276, dims=[0, 2, 1, 3])
        T286 = self.ops.permute(T282, dims=[0, 2, 1, 3])
        S287 = self.define_scalar(0.100000, dtype=DataType.Double)
        S288 = self.define_scalar(True, dtype=DataType.Bool)
        T289, T290, T291 = self.ops.sdpfa_bwd(
            T286,
            T285,
            T284,
            T283,
            self.sdpa_out,
            self.sdpa_log_sumexp,
            S287,
            S288,
            mha_sdpa_seed,
            mha_sdpa_offset,
            None,
        )
        T292 = self.ops.permute(T291, dims=[0, 2, 1, 3])
        T293 = self.ops.permute(T290, dims=[0, 2, 1, 3])
        T294 = self.ops.permute(T289, dims=[0, 2, 1, 3])
        T299 = self.ops.reshape(T292, new_shape=[b, s, e])
        T304 = self.ops.reshape(T293, new_shape=[b, s, e])
        T309 = self.ops.reshape(T294, new_shape=[b, s, e])
        T310 = self.ops.cat([T309, T304, T299], dim=2)
        T314 = self.ops.reshape(T310, new_shape=[b * s, e * 3])
        T315 = self.ops.matmul(T314, self.mha_linear0_weight)
        T320 = self.ops.reshape(T315, new_shape=[b, s, e])
        T325 = self.ops.broadcast_in_dim(
            self.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
        )
        T330 = self.ops.broadcast_in_dim(
            self.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T331 = self.ops.cast(T320, dtype=DataType.Float)
        T332 = self.ops.cast(T325, dtype=DataType.Float)
        T337 = self.ops.broadcast_in_dim(
            T330, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T338 = self.ops.mul(T332, T331)
        T339 = self.ops.sub(T118, T337)
        T340 = self.ops.mul(T339, T338)
        T341 = self.ops.sum(T340, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T346 = self.ops.broadcast_in_dim(T341, shape=[b, s, 1], broadcast_dims=[1])
        T351 = self.ops.broadcast_in_dim(
            self.layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        S352 = self.define_scalar(3.00000, dtype=DataType.Double)
        T353 = self.ops.pow(self.layernorm0_rstd, S352)
        S354 = self.define_scalar(-0.500000, dtype=DataType.Double)
        T355 = self.ops.mul(S354, T346)
        T356 = self.ops.mul(T351, T338)
        T357 = self.ops.mul(T355, T353)
        T358 = self.ops.neg(T356)
        T359 = self.ops.sum(T357, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T360 = self.ops.sum(T358, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T364 = self.ops.broadcast_in_dim(T359, shape=[b, s], broadcast_dims=[1])
        T369 = self.ops.broadcast_in_dim(T360, shape=[b, s, 1], broadcast_dims=[1])
        T374 = self.ops.broadcast_in_dim(
            self.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
        )
        T379 = self.ops.broadcast_in_dim(T364, shape=[b, s, 1], broadcast_dims=[0, 1])
        T380 = self.ops.sum(T369, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T385 = self.ops.broadcast_in_dim(
            T374, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T390 = self.ops.broadcast_in_dim(
            T379, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T394 = self.ops.broadcast_in_dim(T380, shape=[b, s], broadcast_dims=[1])
        T395 = self.ops.sub(T118, T385)
        S396 = self.define_scalar(2.00000, dtype=DataType.Double)
        T397 = self.ops.mul(S396, T390)
        T402 = self.ops.broadcast_in_dim(T394, shape=[b, s, 1], broadcast_dims=[0, 1])
        T403 = self.ops.mul(T397, T395)
        T408 = self.ops.broadcast_in_dim(
            T402, shape=[b, s, e], broadcast_dims=[0, 1, 2]
        )
        T413 = self.ops.broadcast_in_dim(
            self.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T414 = self.ops.mul(T339, T351)
        T419 = self.ops.broadcast_in_dim(
            self.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2]
        )
        T420 = self.ops.mul(T128, T140)
        S421 = self.define_scalar(e, dtype=DataType.Double)
        S422 = self.ops.reciprocal(S421)
        T423 = self.ops.mul(T403, S422)
        S424 = self.define_scalar(1 / e, dtype=DataType.Double)
        T425 = self.ops.mul(S424, T408)
        T426 = self.ops.cast(T413, dtype=DataType.Float)
        T427 = self.ops.mul(T414, T332)
        T428 = self.ops.permute(self.sdpa_out, dims=[0, 2, 1, 3])
        T429 = self.ops.cast(T419, dtype=DataType.Float)
        T430 = self.ops.mul(T420, T120)
        T431 = self.ops.add(T425, T423)
        T432 = self.ops.add(T427, T426)
        T433 = self.ops.stride_order(T428, stride_order=[3, 2, 1, 0])
        T434 = self.ops.add(T430, T429)
        T435 = self.ops.mul(T62, T68)
        T436 = self.ops.add(T356, T431)
        T437 = self.ops.mul(T414, T331)
        T438 = self.ops.cast(T310, dtype=DataType.Float)
        T439 = self.ops.cast(T432, dtype=DataType.BFloat16)
        T444 = self.ops.reshape(T433, new_shape=[b, s, e])
        T445 = self.ops.mul(T420, T119)
        T446 = self.ops.cast(T434, dtype=DataType.BFloat16)
        T447 = self.ops.cast(T435, dtype=DataType.BFloat16)
        T448 = self.ops.add(T205, T436)
        T449 = self.ops.sum(T437, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T450 = self.ops.sum(T331, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T451 = self.ops.sum(T438, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T455 = self.ops.reshape(T439, new_shape=[b * s, e])
        T456 = self.ops.permute(T314, dims=[1, 0])
        T457 = self.ops.sum(T208, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T461 = self.ops.reshape(T444, new_shape=[b * s, e])
        T462 = self.ops.permute(T213, dims=[1, 0])
        T463 = self.ops.sum(T445, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T464 = self.ops.sum(T119, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T465 = self.ops.sum(T82, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T469 = self.ops.reshape(T446, new_shape=[b * s, e])
        T470 = self.ops.permute(T96, dims=[1, 0])
        T471 = self.ops.sum(T41, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T475 = self.ops.reshape(T447, new_shape=[b * s, e * 4])
        T476 = self.ops.permute(T49, dims=[1, 0])
        inp_grad = self.ops.cast(T448, dtype=DataType.BFloat16)
        layernorm0_weight_grad = self.ops.cast(T449, dtype=DataType.BFloat16)
        layernorm0_bias_grad = self.ops.cast(T450, dtype=DataType.BFloat16)
        mha_linear0_bias_grad = self.ops.cast(T451, dtype=DataType.BFloat16)
        mha_linear0_weight_grad = self.ops.matmul(T456, T455)
        mha_linear1_bias_grad = self.ops.cast(T457, dtype=DataType.BFloat16)
        mha_linear1_weight_grad = self.ops.matmul(T462, T461)
        layernorm1_weight_grad = self.ops.cast(T463, dtype=DataType.BFloat16)
        layernorm1_bias_grad = self.ops.cast(T464, dtype=DataType.BFloat16)
        mlp_linear0_bias_grad = self.ops.cast(T465, dtype=DataType.BFloat16)
        mlp_linear0_weight_grad = self.ops.matmul(T470, T469)
        mlp_linear1_bias_grad = self.ops.cast(T471, dtype=DataType.BFloat16)
        mlp_linear1_weight_grad = self.ops.matmul(T476, T475)
        self.add_output(mlp_linear1_weight_grad)
        self.add_output(mlp_linear1_bias_grad)
        self.add_output(mlp_linear0_weight_grad)
        self.add_output(mlp_linear0_bias_grad)
        self.add_output(layernorm1_bias_grad)
        self.add_output(layernorm1_weight_grad)
        self.add_output(mha_linear1_weight_grad)
        self.add_output(mha_linear1_bias_grad)
        self.add_output(mha_linear0_weight_grad)
        self.add_output(mha_linear0_bias_grad)
        self.add_output(layernorm0_bias_grad)
        self.add_output(layernorm0_weight_grad)
        self.add_output(inp_grad)

    def multidevice_schedule(self):
        mesh = self.sched._create_device_mesh(range(self._num_devices))
        for in_tv in [
            self.mlp_linear0_out,
            self.out_grad,
            self.mlp_linear1_weight,
            self.mha_linear1_out,
            self.mlp_linear0_weight,
            self.layernorm1_weight,
            self.layernorm1_mean,
            self.inp,
            self.layernorm1_rstd,
            self.mha_linear1_weight,
            self.mha_linear0_out,
            self.sdpa_out,
            self.sdpa_log_sumexp,
            self.mha_linear0_weight,
            self.layernorm0_weight,
            self.layernorm0_mean,
            self.layernorm0_rstd,
            self.layernorm0_bias,
            self.layernorm1_bias,
        ]:
            self.sched._set_device_mesh(in_tv, mesh)


@pytest.mark.skipif(
    utils.is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(mpi_test, benchmark):
    d = mpi_test.size
    rank = mpi_test.rank

    # I made the batch size 2 to harden the test. It caught several mistakes
    # where I forgot to multiply a dimension size by `b`. When the sharded
    # implementation is ready, I'll reset the batch size to one, to reflect the
    # benchmark workload.
    b, s, h, e = 2, 2048, 96, 12288

    torch.cuda.set_device(mpi_test.local_rank)

    ins = [
        30,
        2722423872872113,
        torch.testing.make_tensor((b, s, e * 4), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e, e * 4), dtype=torch.bfloat16, device="cuda"),
        29,
        2722423872872113,
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e * 4, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((e, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, e * 3), dtype=torch.bfloat16, device="cuda"),
        torch.randn(b * s * e, dtype=torch.bfloat16, device="cuda").as_strided(
            (b, h, s, e // h), (s * e, e // h, e, 1)
        ),
        torch.testing.make_tensor((b, h, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
        torch.testing.make_tensor((e * 3, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
    ]

    fd = TransformerBackwardFusion(d, b, s, h, e)

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

    outs = warmup_fn()
    (
        mlp_linear1_weight_grad,
        mlp_linear1_bias_grad,
        mlp_linear0_weight_grad,
        mlp_linear0_bias_grad,
        layernorm1_bias_grad,
        layernorm1_weight_grad,
        mha_linear1_weight_grad,
        mha_linear1_bias_grad,
        mha_linear0_weight_grad,
        mha_linear0_bias_grad,
        layernorm0_bias_grad,
        layernorm0_weight_grad,
        inp_grad,
    ) = outs
    _assert_shape_dtype(mlp_linear1_weight_grad, [e, e * 4], torch.bfloat16)
    _assert_shape_dtype(mlp_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_weight_grad, [e * 4, e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_bias_grad, [e * 4], torch.bfloat16)
    _assert_shape_dtype(layernorm1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm1_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_weight_grad, [e, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_weight_grad, [e * 3, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_bias_grad, [e * 3], torch.bfloat16)
    _assert_shape_dtype(layernorm0_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm0_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(inp_grad, [b, s, e], torch.bfloat16)

    benchmark.pedantic(benchmark_fn, rounds=5)
