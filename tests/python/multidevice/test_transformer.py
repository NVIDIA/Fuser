# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from sympy import E
import torch
import torch.nn.functional as F

import nvfuser
from nvfuser import DataType, FusionDefinition
from python.utils import create_sdpa_rng_tensors, define_sdpa_rng_state, is_pre_ampere
from benchmark_utils import get_benchmark_fns


@pytest.mark.mpi
def test_grouped_mlp(multidevice_test):
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (prop.major, prop.minor) != (9, 0):
        pytest.skip("at::_grouped_mm only supports sm90.")

    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))
    g = 4
    k = 16
    n = 16 * d

    class Model(FusionDefinition):
        def definition(self):
            self.inp = self.define_tensor(
                [-1, k], dtype=DataType.BFloat16, contiguity=True
            )
            self.gate_w = self.define_tensor(
                [g, k, n], dtype=DataType.BFloat16, contiguity=True
            )
            self.up_w = self.define_tensor(
                [g, k, n], dtype=DataType.BFloat16, contiguity=True
            )
            self.down_w = self.define_tensor(
                [g, n, k], dtype=DataType.BFloat16, contiguity=True
            )
            self.offsets = self.define_tensor(
                [g], dtype=DataType.Int32, contiguity=True
            )

            gate_out = self.ops.grouped_mm(self.inp, self.gate_w, self.offsets)
            gate_out = self.ops.cast(gate_out, DataType.Float)

            up_out = self.ops.grouped_mm(self.inp, self.up_w, self.offsets)

            mul_out = self.ops.mul(self.ops.silu(gate_out), up_out)
            mul_out = self.ops.cast(mul_out, DataType.BFloat16)

            out = self.ops.grouped_mm(mul_out, self.down_w, self.offsets)

            self.add_output(out)

        def multidevice_schedule(self):
            for t in [self.inp, self.gate_w, self.up_w, self.down_w, self.offsets]:
                self.sched._set_device_mesh(t, mesh)

            for w in [self.gate_w, self.up_w]:
                self.sched.split(w, -1, d, False)
                self.sched.parallelize(w, -2, nvfuser.ParallelType.mesh_x)

            self.sched.split(self.down_w, -2, d, False)
            self.sched.parallelize(self.down_w, -3, nvfuser.ParallelType.mesh_x)

    m = 32
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    gate_w = torch.randn(g, k, n, dtype=torch.bfloat16)
    up_w = torch.randn(g, k, n, dtype=torch.bfloat16)
    down_w = torch.randn(g, n, k, dtype=torch.bfloat16)
    sharded_gate_w = multidevice_test.shard_tensor(gate_w, -1, mesh)
    sharded_up_w = multidevice_test.shard_tensor(up_w, -1, mesh)
    sharded_down_w = multidevice_test.shard_tensor(down_w, -2, mesh)
    assert m % g == 0
    group_sizes = [m // g] * g
    offsets = torch.cumsum(torch.tensor(group_sizes), 0, dtype=torch.int32).cuda()

    group_outs = [
        (F.silu(group_in.cpu() @ group_gate_w) * (group_in.cpu() @ group_up_w))
        @ group_down_w
        for group_in, group_gate_w, group_up_w, group_down_w in zip(
            inp.split(group_sizes), gate_w.unbind(), up_w.unbind(), down_w.unbind()
        )
    ]
    expected_out = torch.cat(group_outs, dim=0)

    fd = Model()
    (out,), _ = fd.execute([inp, sharded_gate_w, sharded_up_w, sharded_down_w, offsets])

    # Unfortunately, I couldn't come up with meaningful thresholds to pass the
    # comparison even with one GPU. I manually examined the results. They are
    # not completely off, which is good.
    #
    # I tried several easy things:
    # 1. run the reference implementation on GPU,
    # 2. upcast tensors to `float` here and there in the reference implementation.
    #
    # None of them significantly reduce the error. It could be a problem in the
    # grouped gemm kernel.
    torch.testing.assert_close(out.cpu(), expected_out, rtol=1.0, atol=float("inf"))


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
# https://github.com/Lightning-AI/lightning-thunder/commit/953a91477cec792b6e694650cd2466b871af812d.
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
        b, s, h, e = (
            self._batch,
            self._sequence,
            self._head,
            self._hidden,
        )
        self.inp = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
        self.layernorm0_weight = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.layernorm0_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.mha_linear0_weight = self.define_tensor(shape=[e * 3, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        self.mha_linear0_bias = self.define_tensor(shape=[e * 3], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.mha_linear1_weight = self.define_tensor(shape=[e, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        self.mha_linear1_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.layernorm1_weight = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.layernorm1_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.mlp_linear0_weight = self.define_tensor(shape=[e * 4, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        self.mlp_linear0_bias = self.define_tensor(shape=[e * 4], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        self.mlp_linear1_weight = self.define_tensor(shape=[e, e * 4], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        self.mlp_linear1_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
        
        T13 = self.ops.cast(self.inp, dtype=DataType.Float)
        T14, layernorm0_mean = self.ops.var_mean(T13, dims=[2], correction=0, keepdim=False)
        T20 = self.ops.broadcast_in_dim(T14, shape=[b, s, 1], broadcast_dims=[0, 1])
        T25 = self.ops.broadcast_in_dim(layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
        S26 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T27 = self.ops.add(T20, S26)
        layernorm0_rstd = self.ops.rsqrt(T27)
        T33 = self.ops.broadcast_in_dim(T25, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T34 = self.ops.sub(T13, T33)
        T39 = self.ops.broadcast_in_dim(layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T40 = self.ops.mul(T34, T39)
        T45 = self.ops.broadcast_in_dim(self.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2])
        T46 = self.ops.cast(T45, dtype=DataType.Float)
        T47 = self.ops.mul(T40, T46)
        T52 = self.ops.broadcast_in_dim(self.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2])
        T53 = self.ops.cast(T52, dtype=DataType.Float)
        T54 = self.ops.add(T47, T53)
        T55 = self.ops.cast(T54, dtype=DataType.BFloat16)
        mha_linear0_out = self.ops.linear(T55, self.mha_linear0_weight, self.mha_linear0_bias)
        
        # Reshape before slice to avoid slicing a tensor along sharded dimension.
        # This is different from the single-GPU definition obtained from Thunder.
        T57 = self.ops.reshape(mha_linear0_out, new_shape=[b, s, h, 3*e//h])
        T69 = self.ops.slice(T57, start_indices=[0, 0, 0, 0], end_indices=[b, s, h, e//h])
        T82 = self.ops.slice(T57, start_indices=[0, 0, 0, e//h], end_indices=[b, s, h, 2*e//h])
        T95 = self.ops.slice(T57, start_indices=[0, 0, 0, 2*e//h], end_indices=[b, s, h, 3*e//h])
        
        T102 = self.ops.permute(T82, dims=[0, 2, 1, 3])
        T109 = self.ops.permute(T69, dims=[0, 2, 1, 3])
        T116 = self.ops.permute(T95, dims=[0, 2, 1, 3])
        
        S117 = self.define_scalar(0.100000, dtype=DataType.Double)
        S118 = self.define_scalar(True, dtype=DataType.Bool)
        sdpa_out, sdpa_logsum_exp, sdpa_seed, sdpa_offset = self.ops.sdpfa_fwd(T109, T102, T116, S117, S118, None)
        T123 = self.ops.permute(sdpa_out, dims=[0, 2, 1, 3])
        T124 = self.ops.stride_order(T123, stride_order=[3, 2, 1, 0])
        T129 = self.ops.reshape(T124, new_shape=[b, s, e])
        mha_linear1_out = self.ops.linear(T129, self.mha_linear1_weight, self.mha_linear1_bias)
        S131 = self.define_scalar(0.00000, dtype=DataType.Double)
        S132 = self.define_scalar(1.00000, dtype=DataType.Double)
        T137 = self.ops.uniform(S131, S132, shape=[b, s, e], dtype=DataType.BFloat16)
        S138 = self.define_scalar(0.900000, dtype=DataType.Double)
        mha_dropout_mask = self.ops.lt(T137, S138)
        T140 = self.ops.cast(mha_linear1_out, dtype=DataType.Float)
        T141 = self.ops.cast(mha_dropout_mask, dtype=DataType.Float)
        T142 = self.ops.mul(T140, T141)
        S143 = self.define_scalar(1.11111, dtype=DataType.Double)
        T144 = self.ops.mul(T142, S143)
        T145 = self.ops.add(T13, T144)
        T146, layernorm1_mean = self.ops.var_mean(T145, dims=[2], correction=0, keepdim=False)
        T152 = self.ops.broadcast_in_dim(T146, shape=[b, s, 1], broadcast_dims=[0, 1])
        T157 = self.ops.broadcast_in_dim(layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
        S158 = self.define_scalar(1.00000e-05, dtype=DataType.Double)
        T159 = self.ops.add(T152, S158)
        layernorm1_rstd = self.ops.rsqrt(T159)
        T165 = self.ops.broadcast_in_dim(T157, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T166 = self.ops.sub(T145, T165)
        T171 = self.ops.broadcast_in_dim(layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2])
        T172 = self.ops.mul(T166, T171)
        T177 = self.ops.broadcast_in_dim(self.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2])
        T178 = self.ops.cast(T177, dtype=DataType.Float)
        T179 = self.ops.mul(T172, T178)
        T184 = self.ops.broadcast_in_dim(self.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2])
        T185 = self.ops.cast(T184, dtype=DataType.Float)
        T186 = self.ops.add(T179, T185)
        T187 = self.ops.cast(T186, dtype=DataType.BFloat16)
        mlp_linear0_out = self.ops.linear(T187, self.mlp_linear0_weight, self.mlp_linear0_bias)
        T189 = self.ops.cast(mlp_linear0_out, dtype=DataType.Float)
        T190 = self.ops.mul(T189, T189)
        T191 = self.ops.mul(T190, T189)
        S192 = self.define_scalar(0.500000, dtype=DataType.Double)
        T193 = self.ops.mul(S192, T189)
        S194 = self.define_scalar(0.0447150, dtype=DataType.Double)
        T195 = self.ops.mul(S194, T191)
        T196 = self.ops.add(T189, T195)
        S197 = self.define_scalar(0.797885, dtype=DataType.Double)
        T198 = self.ops.mul(S197, T196)
        T199 = self.ops.tanh(T198)
        S200 = self.define_scalar(1.00000, dtype=DataType.Double)
        T201 = self.ops.add(S200, T199)
        T202 = self.ops.mul(T193, T201)
        T203 = self.ops.cast(T202, dtype=DataType.BFloat16)
        mlp_linear1_out = self.ops.linear(T203, self.mlp_linear1_weight, self.mlp_linear1_bias)
        S205 = self.define_scalar(0.00000, dtype=DataType.Double)
        S206 = self.define_scalar(1.00000, dtype=DataType.Double)
        T211 = self.ops.uniform(S205, S206, shape=[b, s, e], dtype=DataType.BFloat16)
        S212 = self.define_scalar(0.900000, dtype=DataType.Double)
        mlp_dropout_mask = self.ops.lt(T211, S212)
        T214 = self.ops.cast(mlp_linear1_out, dtype=DataType.Float)
        T215 = self.ops.cast(mlp_dropout_mask, dtype=DataType.Float)
        T216 = self.ops.mul(T214, T215)
        S217 = self.define_scalar(1.11111, dtype=DataType.Double)
        T218 = self.ops.mul(T216, S217)
        T219 = self.ops.add(T145, T218)
        out = self.ops.cast(T219, dtype=DataType.BFloat16)
        self.add_output(layernorm0_mean)
        self.add_output(layernorm0_rstd)
        self.add_output(mha_linear0_out)
        self.add_output(sdpa_out)
        self.add_output(sdpa_logsum_exp)
        self.add_output(sdpa_seed)
        self.add_output(sdpa_offset)
        self.add_output(mha_linear1_out)
        self.add_output(mha_dropout_mask)
        self.add_output(layernorm1_mean)
        self.add_output(layernorm1_rstd)
        self.add_output(mlp_linear0_out)
        self.add_output(mlp_dropout_mask)
        self.add_output(out)
        

    def multidevice_schedule(self):
        mesh = nvfuser.DeviceMesh(range(self._num_devices))
        for tv in [
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
            self.sched._set_device_mesh(tv, mesh)

        for tv in [
            self.mha_linear0_weight,
            self.mha_linear0_bias,
            self.mlp_linear0_weight,
            self.mlp_linear0_bias,
        ]:
            self.sched.split(tv, 0, self._num_devices, False)
            self.sched.parallelize(tv, 0, nvfuser.ParallelType.mesh_x)


        for tv in [
            self.mha_linear1_weight,
            self.mlp_linear1_weight,
        ]:
            self.sched.split(tv, -1, self._num_devices, False)
            self.sched.parallelize(tv, -2, nvfuser.ParallelType.mesh_x)


# TODO(#2962): validate the numbers as well. Currently, the numbers are off
# by a lot, making comparison infeasible.
def _assert_shape_dtype(
    t: torch.Tensor, expected_sizes: list[int], expected_dtype: torch.dtype
) -> None:
    assert t.shape == torch.Size(expected_sizes)
    assert t.dtype == expected_dtype


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_forward(multidevice_test, benchmark):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    b, s, h, e = 16, 128, 12, 768

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

    torch.cuda.set_device(multidevice_test.local_rank)

    # To reduce memory footprint, create unsharded data on CPU and copy only
    # the needed slice to GPU.
    mha_linear0_weight = torch.testing.make_tensor(
        e * 3, e, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear0_bias = torch.testing.make_tensor(
        e * 3, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear1_weight = torch.testing.make_tensor(
        e, e, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear0_weight = torch.testing.make_tensor(
        e * 4, e, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear0_bias = torch.testing.make_tensor(
        e * 4, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear1_weight = torch.testing.make_tensor(
        e, e * 4, dtype=torch.bfloat16, device="cpu"
    )
    
    # See TransformerForwardFusion.definition for the meanings of these
    # arguments. They are passed in in the same order as the `define_scalar`s
    # and `define_tensor`s.
    ins = [
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        multidevice_test.shard_tensor(mha_linear0_weight, 0, mesh),
        multidevice_test.shard_tensor(mha_linear0_bias, 0, mesh),
        multidevice_test.shard_tensor(mha_linear1_weight, -1, mesh),
        torch.testing.make_tensor(e, dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        multidevice_test.shard_tensor(mlp_linear0_weight, 0, mesh),
        multidevice_test.shard_tensor(mlp_linear0_bias, 0, mesh),
        multidevice_test.shard_tensor(mlp_linear1_weight, -1, mesh),
        torch.testing.make_tensor(e, dtype=torch.bfloat16, device="cuda"),
    ]

    fd = TransformerForwardFusion(d, b, s, h, e)

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

    # Warm up and validate.
    outs, _ = warmup_fn()
    (
        layernorm0_mean,
        layernorm0_rstd,
        mha_linear0_out,
        sdpa_out,
        sdpa_logsum_exp,
        sdpa_seed,
        sdpa_offset,
        mha_linear1_out,
        mha_dropout_mask,
        layernorm1_mean,
        layernorm1_rstd,
        mlp_linear0_out,
        mlp_dropout_mask,
        out,
    ) = outs

    _assert_shape_dtype(layernorm0_mean, [b, s], torch.float32)
    _assert_shape_dtype(layernorm0_rstd, [b, s, 1], torch.float32)
    _assert_shape_dtype(mha_linear0_out, [b, s, e * 3 // d], torch.bfloat16)
    _assert_shape_dtype(sdpa_out, [b, h // d, s, e // h], torch.bfloat16)
    _assert_shape_dtype(sdpa_logsum_exp, [b, h // d, s], torch.float32)
    ref_philox_seed, ref_philox_offset = create_sdpa_rng_tensors()
    _assert_shape_dtype(sdpa_seed, ref_philox_seed.shape, ref_philox_seed.dtype)
    _assert_shape_dtype(sdpa_offset, ref_philox_offset.shape, ref_philox_offset.dtype)
    _assert_shape_dtype(mha_linear1_out, [b, s, e], torch.bfloat16)
    _assert_shape_dtype(mha_dropout_mask, [b, s, e], torch.bool)
    _assert_shape_dtype(layernorm1_mean, [b, s], torch.float32)
    _assert_shape_dtype(layernorm1_rstd, [b, s, 1], torch.float32)
    _assert_shape_dtype(mlp_linear0_out, [b, s, e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(mlp_dropout_mask, [b, s, e], torch.bool)
    _assert_shape_dtype(out, [b, s, e], torch.bfloat16)

    # # Benchmark and profile. The profile can be collected and displayed using
    # # `nsys`. See instructions in test_transformer_engine.py.
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
        d, b, s, h, e = (
            self._num_devices,
            self._batch,
            self._sequence,
            self._head,
            self._hidden,
        )

        mlp_dropout_offset = self.define_scalar(None, dtype=DataType.Int)
        mlp_dropout_seed = self.define_scalar(None, dtype=DataType.Int)
        self.mlp_linear0_out = self.define_tensor(
            shape=[d, b, s, e * 4 // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.out_grad = self.define_tensor(
            shape=[b, s, e],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mlp_linear1_weight = self.define_tensor(
            shape=[d, e, e * 4 // d],
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
            shape=[d, e * 4 // d, e],
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
            shape=[d, e, e // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.mha_linear0_out = self.define_tensor(
            shape=[d, b, s, e * 3 // d],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        self.sdpa_out = self.define_tensor(
            shape=[d, b, h // d, s, e // h],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=[4, 3, 1, 2, 0],
        )
        self.sdpa_log_sumexp = self.define_tensor(
            shape=[d, b, h // d, s],
            contiguity=True,
            dtype=DataType.Float,
        )
        mha_sdpa_seed, mha_sdpa_offset = define_sdpa_rng_state(self)
        self.mha_linear0_weight = self.define_tensor(
            shape=[d, e * 3 // d, e],
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
        T58 = self.ops.reshape(T52, new_shape=[d, b, s, e * 4 // d])
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
        T96 = self.ops.reshape(T90, new_shape=[d, b * s, e * 4 // d])
        T97 = self.ops.cast(T92, dtype=DataType.Float)
        T98 = self.ops.cast(self.mha_linear1_out, dtype=DataType.Float)
        T99_local = self.ops.matmul(T96, self.mlp_linear0_weight)
        T99 = self.ops.sum(T99_local, [0])  # allreduce
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
            start_indices=[0, 0, 0, e * 2 // d],
            end_indices=[d, b, s, e * 3 // d],
        )
        T240 = self.ops.slice(
            self.mha_linear0_out,
            start_indices=[0, 0, 0, e // d],
            end_indices=[d, b, s, e * 2 // d],
        )
        T253 = self.ops.slice(
            self.mha_linear0_out,
            start_indices=[0, 0, 0, 0],
            end_indices=[d, b, s, e // d],
        )
        T258 = self.ops.reshape(T214, new_shape=[d, b, s, e // d])
        T264 = self.ops.reshape(T227, new_shape=[d, b, s, h // d, e // h])
        T270 = self.ops.reshape(T240, new_shape=[d, b, s, h // d, e // h])
        T276 = self.ops.reshape(T253, new_shape=[d, b, s, h // d, e // h])
        T282 = self.ops.reshape(T258, new_shape=[d, b, s, h // d, e // h])
        T283 = self.ops.permute(T264, dims=[0, 1, 3, 2, 4])
        T284 = self.ops.permute(T270, dims=[0, 1, 3, 2, 4])
        T285 = self.ops.permute(T276, dims=[0, 1, 3, 2, 4])
        T286 = self.ops.permute(T282, dims=[0, 1, 3, 2, 4])
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
        T292 = self.ops.permute(T291, dims=[0, 1, 3, 2, 4])
        T293 = self.ops.permute(T290, dims=[0, 1, 3, 2, 4])
        T294 = self.ops.permute(T289, dims=[0, 1, 3, 2, 4])
        T299 = self.ops.reshape(T292, new_shape=[d, b, s, e // d])
        T304 = self.ops.reshape(T293, new_shape=[d, b, s, e // d])
        T309 = self.ops.reshape(T294, new_shape=[d, b, s, e // d])
        T310 = self.ops.cat([T309, T304, T299], dim=3)
        T314 = self.ops.reshape(T310, new_shape=[d, b * s, e * 3 // d])
        T315_local = self.ops.matmul(T314, self.mha_linear0_weight)
        T315 = self.ops.sum(T315_local, [0])  # allreduce
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
        T428 = self.ops.permute(self.sdpa_out, dims=[0, 1, 3, 2, 4])
        T429 = self.ops.cast(T419, dtype=DataType.Float)
        T430 = self.ops.mul(T420, T120)
        T431 = self.ops.add(T425, T423)
        T432 = self.ops.add(T427, T426)
        T433 = self.ops.stride_order(T428, stride_order=[4, 3, 2, 1, 0])
        T434 = self.ops.add(T430, T429)
        T435 = self.ops.mul(T62, T68)
        T436 = self.ops.add(T356, T431)
        T437 = self.ops.mul(T414, T331)
        T438 = self.ops.cast(T310, dtype=DataType.Float)
        T439 = self.ops.cast(T432, dtype=DataType.BFloat16)
        T444 = self.ops.reshape(T433, new_shape=[d, b, s, e // d])
        T445 = self.ops.mul(T420, T119)
        T446 = self.ops.cast(T434, dtype=DataType.BFloat16)
        T447 = self.ops.cast(T435, dtype=DataType.BFloat16)
        T448 = self.ops.add(T205, T436)
        T449 = self.ops.sum(T437, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T450 = self.ops.sum(T331, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T451 = self.ops.sum(T438, dims=[1, 2], keepdim=False, dtype=DataType.Null)
        T455 = self.ops.reshape(T439, new_shape=[b * s, e])
        T456 = self.ops.permute(T314, dims=[0, 2, 1])
        T457 = self.ops.sum(T208, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T461 = self.ops.reshape(T444, new_shape=[d, b * s, e // d])
        T462 = self.ops.permute(T213, dims=[1, 0])
        T463 = self.ops.sum(T445, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T464 = self.ops.sum(T119, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T465 = self.ops.sum(T82, dims=[1, 2], keepdim=False, dtype=DataType.Null)
        T469 = self.ops.reshape(T446, new_shape=[b * s, e])
        T470 = self.ops.permute(T96, dims=[0, 2, 1])
        T471 = self.ops.sum(T41, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T475 = self.ops.reshape(T447, new_shape=[d, b * s, e * 4 // d])
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
        mesh = nvfuser.DeviceMesh(range(self._num_devices))
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

        for in_tv in [
            self.mlp_linear0_out,
            self.mlp_linear1_weight,
            self.mlp_linear0_weight,
            self.mha_linear1_weight,
            self.mha_linear0_out,
            self.sdpa_out,
            self.sdpa_log_sumexp,
            self.mha_linear0_weight,
        ]:
            self.sched.parallelize(in_tv, 0, nvfuser.ParallelType.mesh_x)


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(multidevice_test, benchmark):
    d = multidevice_test.size
    rank = multidevice_test.rank

    b, s, h, e = 1, 2048, 96, 12288

    torch.cuda.set_device(multidevice_test.local_rank)

    mlp_linear0_out = torch.testing.make_tensor(
        d, b, s, e * 4 // d, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear1_weight = torch.testing.make_tensor(
        d, e, e * 4 // d, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear0_weight = torch.testing.make_tensor(
        d, e * 4 // d, e, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear1_weight = torch.testing.make_tensor(
        d, e, e // d, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear0_out = torch.testing.make_tensor(
        d, b, s, e * 3 // d, dtype=torch.bfloat16, device="cpu"
    )
    sdpa_out = torch.testing.make_tensor(
        d, b, s, h // d, e // h, dtype=torch.bfloat16, device="cpu"
    ).permute(0, 1, 3, 2, 4)
    sdpa_log_sumexp = torch.testing.make_tensor(
        d, b, h // d, s, dtype=torch.float32, device="cpu"
    )
    mha_linear0_weight = torch.testing.make_tensor(
        d, e * 3 // d, e, dtype=torch.bfloat16, device="cpu"
    )
    sdpa_philox_seed, sdpa_philox_offset = create_sdpa_rng_tensors()
    ins = [
        30,
        2722423872872113,
        mlp_linear0_out[rank : rank + 1].cuda(),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        mlp_linear1_weight[rank : rank + 1].cuda(),
        29,
        2722423872872113,
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        mlp_linear0_weight[rank : rank + 1].cuda(),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        mha_linear1_weight[rank : rank + 1].cuda(),
        mha_linear0_out[rank : rank + 1].cuda(),
        sdpa_out[rank : rank + 1].cuda(),
        sdpa_log_sumexp[rank : rank + 1].cuda(),
        sdpa_philox_seed,
        sdpa_philox_offset,
        mha_linear0_weight[rank : rank + 1].cuda(),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
    ]

    fd = TransformerBackwardFusion(d, b, s, h, e)

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

    outs, _ = warmup_fn()
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
    _assert_shape_dtype(mlp_linear1_weight_grad, [1, e, e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(mlp_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_weight_grad, [1, e * 4 // d, e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_bias_grad, [1, e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(layernorm1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm1_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_weight_grad, [1, e, e // d], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_weight_grad, [1, e * 3 // d, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_bias_grad, [1, e * 3 // d], torch.bfloat16)
    _assert_shape_dtype(layernorm0_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm0_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(inp_grad, [b, s, e], torch.bfloat16)

    benchmark.pedantic(benchmark_fn, rounds=5)
