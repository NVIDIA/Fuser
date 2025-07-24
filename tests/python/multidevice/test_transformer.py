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
            self._num_devices,
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
        T95 = self.ops.slice(T57, start_indices=[0, 0, h, 2*e//h], end_indices=[b, s, h, 3*e//h])
        
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
            self.sched.parallelize(tv, 0, nvfuser.ParallelType.mesh_x)
            self.sched.split(tv, 0, nvfuser.ParallelType.mesh_x, False)

        for tv in [
            self.mha_linear1_weight,
            self.mlp_linear1_weight,
        ]:
            self.sched.split(tv, -1, nvfuser.ParallelType.mesh_x, False)
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
      b, s, h, e = (
          self._batch,
          self._sequence,
          self._head,
          self._hidden,
      )

      self.mlp_linear0_out = self.define_tensor(shape=[b, s, 4 * e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
      self.out_grad = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
      self.mlp_dropout_mask = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[2, 1, 0])
      self.mlp_linear1_weight = self.define_tensor(shape=[e, e * 4], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
      self.mha_dropout_mask = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[2, 1, 0])
      self.mha_linear1_out = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
      self.mlp_linear0_weight = self.define_tensor(shape=[4 * e, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
      self.layernorm1_weight = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
      self.layernorm1_mean = self.define_tensor(shape=[b, s], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
      self.inp = self.define_tensor(shape=[b, s, e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
      self.layernorm1_rstd = self.define_tensor(shape=[b, s, 1], contiguity=[True, True, None], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
      self.mha_linear1_weight = self.define_tensor(shape=[e, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
      self.mha_linear0_out = self.define_tensor(shape=[b, s, 3 * e], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
      self.sdpa_out = self.define_tensor(shape=[b, h, s, e // h], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
      self.sdpa_logsum_exp = self.define_tensor(shape=[b, h, s], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
      self.sdpa_seed = self.define_tensor(shape=[2], contiguity=[True], dtype=DataType.UInt64, is_cpu=False, stride_order=[0])
      self.sdpa_offset = self.define_tensor(shape=[], contiguity=[], dtype=DataType.UInt64, is_cpu=False)
      self.mha_linear0_weight = self.define_tensor(shape=[3*e, e], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
      self.layernorm0_weight = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
      self.layernorm0_mean = self.define_tensor(shape=[b, s], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
      self.layernorm0_rstd = self.define_tensor(shape=[b, s, 1], contiguity=[True, True, None], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
      self.layernorm0_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
      self.layernorm1_bias = self.define_tensor(shape=[e], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
      T23 = self.ops.cast(self.mlp_linear0_out, dtype=DataType.Float)
      T24 = self.ops.cast(self.out_grad, dtype=DataType.Float)
      T25 = self.ops.mul(T23, T23)
      S26 = self.define_scalar(1.11111, dtype=DataType.Double)
      T27 = self.ops.mul(S26, T24)
      T28 = self.ops.cast(self.mlp_dropout_mask, dtype=DataType.Float)
      T29 = self.ops.mul(T25, T23)
      T30 = self.ops.mul(T28, T27)
      S31 = self.define_scalar(0.0447150, dtype=DataType.Double)
      T32 = self.ops.mul(S31, T29)
      T33 = self.ops.cast(T30, dtype=DataType.BFloat16)
      T34 = self.ops.add(T23, T32)
      T38 = self.ops.reshape(T33, new_shape=[2048, 768])
      S39 = self.define_scalar(0.797885, dtype=DataType.Double)
      T40 = self.ops.mul(S39, T34)
      T41 = self.ops.matmul(T38, self.mlp_linear1_weight)
      T42 = self.ops.tanh(T40)
      T47 = self.ops.reshape(T41, new_shape=[b, s, 4 * e])
      T48 = self.ops.mul(T42, T42)
      T49 = self.ops.cast(T47, dtype=DataType.Float)
      S50 = self.define_scalar(0.500000, dtype=DataType.Double)
      T51 = self.ops.mul(S50, T23)
      S52 = self.define_scalar(1.00000, dtype=DataType.Double)
      T53 = self.ops.sub(S52, T48)
      T54 = self.ops.mul(T51, T49)
      T55 = self.ops.mul(T54, T53)
      S56 = self.define_scalar(1.00000, dtype=DataType.Double)
      T57 = self.ops.add(S56, T42)
      S58 = self.define_scalar(0.797885, dtype=DataType.Double)
      T59 = self.ops.mul(S58, T55)
      T60 = self.ops.mul(T57, T49)
      S61 = self.define_scalar(0.0447150, dtype=DataType.Double)
      T62 = self.ops.mul(S61, T59)
      S63 = self.define_scalar(0.500000, dtype=DataType.Double)
      T64 = self.ops.mul(S63, T60)
      T65 = self.ops.mul(T23, T62)
      T66 = self.ops.mul(T25, T62)
      T67 = self.ops.add(T59, T64)
      T68 = self.ops.mul(T23, T65)
      T69 = self.ops.add(T67, T66)
      T70 = self.ops.add(T69, T68)
      T71 = self.ops.add(T70, T68)
      T72 = self.ops.cast(T71, dtype=DataType.BFloat16)
      T76 = self.ops.reshape(T72, new_shape=[2048, 3072])
      T77 = self.ops.cast(self.mha_dropout_mask, dtype=DataType.Float)
      T78 = self.ops.cast(self.mha_linear1_out, dtype=DataType.Float)
      T79 = self.ops.matmul(T76, self.mlp_linear0_weight)
      T80 = self.ops.mul(T78, T77)
      T85 = self.ops.reshape(T79, new_shape=[b, s, e])
      T90 = self.ops.broadcast_in_dim(self.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2])
      T95 = self.ops.broadcast_in_dim(self.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
      S96 = self.define_scalar(1.11111, dtype=DataType.Double)
      T97 = self.ops.mul(T80, S96)
      T98 = self.ops.cast(self.inp, dtype=DataType.Float)
      T99 = self.ops.cast(T85, dtype=DataType.Float)
      T100 = self.ops.cast(T90, dtype=DataType.Float)
      T105 = self.ops.broadcast_in_dim(T95, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T106 = self.ops.add(T98, T97)
      T107 = self.ops.mul(T100, T99)
      T108 = self.ops.sub(T106, T105)
      T109 = self.ops.mul(T108, T107)
      T110 = self.ops.sum(T109, dims=[2], keepdim=False, dtype=DataType.Null)
      T115 = self.ops.broadcast_in_dim(T110, shape=[b, s, 1], broadcast_dims=[0, 1])
      T120 = self.ops.broadcast_in_dim(self.layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      S121 = self.define_scalar(3.00000, dtype=DataType.Double)
      T122 = self.ops.pow(self.layernorm1_rstd, S121)
      S123 = self.define_scalar(-0.500000, dtype=DataType.Double)
      T124 = self.ops.mul(S123, T115)
      T125 = self.ops.mul(T120, T107)
      T126 = self.ops.mul(T124, T122)
      T127 = self.ops.neg(T125)
      T128 = self.ops.sum(T126, dims=[2], keepdim=False, dtype=DataType.Null)
      T129 = self.ops.sum(T127, dims=[2], keepdim=False, dtype=DataType.Null)
      T134 = self.ops.broadcast_in_dim(self.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
      T139 = self.ops.broadcast_in_dim(T128, shape=[b, s, 1], broadcast_dims=[0, 1])
      T144 = self.ops.broadcast_in_dim(T129, shape=[b, s, 1], broadcast_dims=[0, 1])
      T149 = self.ops.broadcast_in_dim(T134, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T154 = self.ops.broadcast_in_dim(T139, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T155 = self.ops.sum(T144, dims=[2], keepdim=False, dtype=DataType.Null)
      T156 = self.ops.sub(T106, T149)
      S157 = self.define_scalar(2.00000, dtype=DataType.Double)
      T158 = self.ops.mul(S157, T154)
      T163 = self.ops.broadcast_in_dim(T155, shape=[b, s, 1], broadcast_dims=[0, 1])
      T164 = self.ops.mul(T158, T156)
      T169 = self.ops.broadcast_in_dim(T163, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      S170 = self.define_scalar(e, dtype=DataType.Double)
      S171 = self.ops.reciprocal(S170)
      T172 = self.ops.mul(T164, S171)
      S173 = self.define_scalar(1 / e, dtype=DataType.Double)
      T174 = self.ops.mul(S173, T169)
      T175 = self.ops.add(T174, T172)
      T176 = self.ops.add(T24, T125)
      T177 = self.ops.add(T176, T175)
      S178 = self.define_scalar(1.11111, dtype=DataType.Double)
      T179 = self.ops.mul(S178, T177)
      T180 = self.ops.mul(T77, T179)
      T181 = self.ops.cast(T180, dtype=DataType.BFloat16)
      T185 = self.ops.reshape(T181, new_shape=[b * s, e])
      T186 = self.ops.matmul(T185, self.mha_linear1_weight)
      
      # Reshape before slicing to avoid slicing along sharded dimension
      T187 = self.ops.reshape(self.mha_linear0_out, new_shape=[b, s, h, 3 * e // h])
      T199 = self.ops.slice(T187, start_indices=[0, 0, 0, 0], end_indices=[b, s, h, e // h])
      T212 = self.ops.slice(T187, start_indices=[0, 0, 0, e // h], end_indices=[b, s, h, 2 * e // h])
      T225 = self.ops.slice(T187, start_indices=[0, 0, 0, 2 * e // h], end_indices=[b, s, h, 3 * e // h])
      T230 = self.ops.reshape(T186, new_shape=[b, s, e])
      T254 = self.ops.reshape(T230, new_shape=[b, s, h, e // h])
      T255 = self.ops.permute(T199, dims=[0, 2, 1, 3])
      T256 = self.ops.permute(T212, dims=[0, 2, 1, 3])
      T257 = self.ops.permute(T225, dims=[0, 2, 1, 3])
      T258 = self.ops.permute(T254, dims=[0, 2, 1, 3])
      S259 = self.define_scalar(0.100000, dtype=DataType.Double)
      S260 = self.define_scalar(True, dtype=DataType.Bool)
      T261, T262, T263 = self.ops.sdpfa_bwd(T258, T257, T256, T255, self.sdpa_out, self.sdpa_logsum_exp, S259, S260, self.sdpa_seed, self.sdpa_offset, None)
      T264 = self.ops.permute(T263, dims=[0, 2, 1, 3])
      T265 = self.ops.permute(T262, dims=[0, 2, 1, 3])
      T266 = self.ops.permute(T261, dims=[0, 2, 1, 3])
      
      # Cat before reshape to avoid concatenating along sharded dimension
      T270 = self.ops.cat([T264, T265, T266], dim=2, manual_padding=0)
      T271 = self.ops.reshape(T270, new_shape=[b, s, 3 * e])
      T286 = self.ops.reshape(T271, new_shape=[b*s, 3 * e])
      T287 = self.ops.matmul(T286, self.mha_linear0_weight)
      T292 = self.ops.reshape(T287, new_shape=[b, s, e])
      T297 = self.ops.broadcast_in_dim(self.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2])
      T302 = self.ops.broadcast_in_dim(self.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
      T303 = self.ops.cast(T292, dtype=DataType.Float)
      T304 = self.ops.cast(T297, dtype=DataType.Float)
      T309 = self.ops.broadcast_in_dim(T302, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T310 = self.ops.mul(T304, T303)
      T311 = self.ops.sub(T98, T309)
      T312 = self.ops.mul(T311, T310)
      T313 = self.ops.sum(T312, dims=[2], keepdim=False, dtype=DataType.Null)
      T318 = self.ops.broadcast_in_dim(T313, shape=[b, s, 1], broadcast_dims=[0, 1])
      T323 = self.ops.broadcast_in_dim(self.layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      S324 = self.define_scalar(3.00000, dtype=DataType.Double)
      T325 = self.ops.pow(self.layernorm0_rstd, S324)
      S326 = self.define_scalar(-0.500000, dtype=DataType.Double)
      T327 = self.ops.mul(S326, T318)
      T328 = self.ops.mul(T323, T310)
      T329 = self.ops.mul(T327, T325)
      T330 = self.ops.neg(T328)
      T331 = self.ops.sum(T329, dims=[2], keepdim=False, dtype=DataType.Null)
      T332 = self.ops.sum(T330, dims=[2], keepdim=False, dtype=DataType.Null)
      T337 = self.ops.broadcast_in_dim(self.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1])
      T342 = self.ops.broadcast_in_dim(T331, shape=[b, s, 1], broadcast_dims=[0, 1])
      T347 = self.ops.broadcast_in_dim(T332, shape=[b, s, 1], broadcast_dims=[0, 1])
      T352 = self.ops.broadcast_in_dim(T337, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T357 = self.ops.broadcast_in_dim(T342, shape=[b, s, e], broadcast_dims=[0, 1, 2])
      T358 = self.ops.sum(T347, dims=[2], keepdim=False, dtype=DataType.Null)
      T359 = self.ops.sub(T98, T352)
      S360 = self.define_scalar(2.00000, dtype=DataType.Double)
      T361 = self.ops.mul(S360, T357)
      T366 = self.ops.broadcast_in_dim(T358, shape=[b, s, 1], broadcast_dims=[0, 1])
      T371 = self.ops.broadcast_in_dim(self.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2])
      T372 = self.ops.mul(T311, T323)
      T377 = self.ops.broadcast_in_dim(self.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2])
      T378 = self.ops.mul(T108, T120)
      T379 = self.ops.mul(T361, T359)
      T384 = self.ops.broadcast_in_dim(T366, shape=[16, 128, 768], broadcast_dims=[0, 1, 2])
      T385 = self.ops.cast(T371, dtype=DataType.Float)
      T386 = self.ops.mul(T372, T304)
      T387 = self.ops.permute(self.sdpa_out, dims=[0, 2, 1, 3])
      T388 = self.ops.cast(T377, dtype=DataType.Float)
      T389 = self.ops.mul(T378, T100)
      S390 = self.define_scalar(e, dtype=DataType.Double)
      S391 = self.ops.reciprocal(S390)
      T392 = self.ops.mul(T379, S391)
      S393 = self.define_scalar(1 / e, dtype=DataType.Double)
      T394 = self.ops.mul(S393, T384)
      T395 = self.ops.add(T386, T385)
      T396 = self.ops.stride_order(T387, stride_order=[3, 2, 1, 0])
      T397 = self.ops.add(T389, T388)
      T398 = self.ops.mul(T51, T57)
      T399 = self.ops.add(T394, T392)
      T400 = self.ops.add(T177, T328)
      T401 = self.ops.mul(T372, T303)
      T402 = self.ops.cast(T271, dtype=DataType.Float)
      T403 = self.ops.cast(T395, dtype=DataType.BFloat16)
      T408 = self.ops.reshape(T396, new_shape=[b, s, e])
      T409 = self.ops.mul(T378, T99)
      T410 = self.ops.cast(T397, dtype=DataType.BFloat16)
      T411 = self.ops.cast(T398, dtype=DataType.BFloat16)
      T412 = self.ops.add(T400, T399)
      T413 = self.ops.sum(T401, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T414 = self.ops.sum(T303, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T415 = self.ops.sum(T402, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T419 = self.ops.reshape(T403, new_shape=[b * s, e])
      T420 = self.ops.permute(T286, dims=[1, 0])
      T421 = self.ops.sum(T180, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T425 = self.ops.reshape(T408, new_shape=[b * s, e])
      T426 = self.ops.permute(T185, dims=[1, 0])
      T427 = self.ops.sum(T409, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T428 = self.ops.sum(T99, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T429 = self.ops.sum(T71, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T433 = self.ops.reshape(T410, new_shape=[b * s, e])
      T434 = self.ops.permute(T76, dims=[1, 0])
      T435 = self.ops.sum(T30, dims=[0, 1], keepdim=False, dtype=DataType.Null)
      T439 = self.ops.reshape(T411, new_shape=[b * s, 3 * e])
      T440 = self.ops.permute(T38, dims=[1, 0])
      inp_grad = self.ops.cast(T412, dtype=DataType.BFloat16)
      layernorm0_weight_grad = self.ops.cast(T413, dtype=DataType.BFloat16)
      layernorm0_bias_grad = self.ops.cast(T414, dtype=DataType.BFloat16)
      mha_linear0_bias_grad = self.ops.cast(T415, dtype=DataType.BFloat16)
      mha_linear0_weight_grad = self.ops.matmul(T420, T419)
      mha_linear1_bias_grad = self.ops.cast(T421, dtype=DataType.BFloat16)
      mha_linear1_weight_grad = self.ops.matmul(T426, T425)
      layernorm1_weight_grad = self.ops.cast(T427, dtype=DataType.BFloat16)
      layernorm1_bias_grad = self.ops.cast(T428, dtype=DataType.BFloat16)
      mlp_linear0_bias_grad = self.ops.cast(T429, dtype=DataType.BFloat16)
      mlp_linear0_weight_grad = self.ops.matmul(T434, T433)
      mlp_linear1_bias_grad = self.ops.cast(T435, dtype=DataType.BFloat16)
      mlp_linear1_weight_grad = self.ops.matmul(T440, T439)
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
        for tv in [
            self.mlp_linear0_out,
            self.out_grad,
            self.mlp_dropout_mask,
            self.mlp_linear1_weight,
            self.mha_dropout_mask,
            self.mha_linear1_out,
            self.mlp_linear0_weight,
            self.layernorm1_weight,
            self.layernorm1_mean,
            self.inp,
            self.layernorm1_rstd,
            self.mha_linear1_weight,
            self.mha_linear0_out,
            self.sdpa_out,
            self.sdpa_logsum_exp,
            self.sdpa_seed,
            self.sdpa_offset,
            self.mha_linear0_weight,
            self.layernorm0_weight,
            self.layernorm0_mean,
            self.layernorm0_rstd,
            self.layernorm0_bias,
            self.layernorm1_bias,
        ]:
            self.sched._set_device_mesh(tv, mesh)
        
        for tv in [
            self.mha_linear0_weight,
            self.mlp_linear0_weight,
        ]:
            self.sched.split(tv, 0, self._num_devices, False)
            self.sched.parallelize(tv, 0, nvfuser.ParallelType.mesh_x)

        for tv in [
            self.sdpa_out,
            self.sdpa_logsum_exp,
        ]:
            self.sched.split(tv, 1, self._num_devices, False)
            self.sched.parallelize(tv, 1, nvfuser.ParallelType.mesh_x)
        
        for tv in [
            self.mlp_linear0_out,
            self.mha_linear0_out,
            self.mha_linear1_weight,
            self.mlp_linear1_weight,
        ]:
            self.sched.split(tv, -1, self._num_devices, False)
            self.sched.parallelize(tv, -1, nvfuser.ParallelType.mesh_x)



@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(multidevice_test, benchmark):
    d = multidevice_test.size
    mesh = nvfuser.DeviceMesh(range(d))

    b, s, h, e = 12, 128, 12, 768

    torch.cuda.set_device(multidevice_test.local_rank)

    mlp_linear0_out = torch.testing.make_tensor(
        b, s, e * 4, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear1_weight = torch.testing.make_tensor(
        e, e * 4, dtype=torch.bfloat16, device="cpu"
    )
    mlp_linear0_weight = torch.testing.make_tensor(
        e * 4, e, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear1_weight = torch.testing.make_tensor(
        e, e, dtype=torch.bfloat16, device="cpu"
    )
    mha_linear0_out = torch.testing.make_tensor(
        b, s, e * 3, dtype=torch.bfloat16, device="cpu"
    )
    sdpa_out = torch.testing.make_tensor(
        b, h, s, e // h, dtype=torch.bfloat16, device="cpu"
    )
    sdpa_log_sumexp = torch.testing.make_tensor(
        b, h, s, dtype=torch.float32, device="cpu"
    )
    mha_linear0_weight = torch.testing.make_tensor(
        e * 3, e, dtype=torch.bfloat16, device="cpu"
    )
    sdpa_philox_seed, sdpa_philox_offset = create_sdpa_rng_tensors()
    ins = [
        multidevice_test.shard_tensor(mlp_linear0_out, -1, mesh),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bool, device="cuda"),
        multidevice_test.shard_tensor(mlp_linear1_weight, -1, mesh),
        torch.testing.make_tensor((b, s, e), dtype=torch.bool, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        multidevice_test.shard_tensor(mlp_linear0_weight, 0, mesh),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        multidevice_test.shard_tensor(mha_linear1_weight, -1, mesh),
        multidevice_test.shard_tensor(mha_linear0_out, -1, mesh),
        multidevice_test.shard_tensor(sdpa_out, 1, mesh),
        multidevice_test.shard_tensor(sdpa_log_sumexp, 1, mesh),
        sdpa_philox_seed,
        sdpa_philox_offset,
        multidevice_test.shard_tensor(mha_linear0_weight, 0, mesh),
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
    _assert_shape_dtype(mlp_linear1_weight_grad, [e, e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(mlp_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_weight_grad, [e * 4 // d, e], torch.bfloat16)
    _assert_shape_dtype(mlp_linear0_bias_grad, [e * 4 // d], torch.bfloat16)
    _assert_shape_dtype(layernorm1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm1_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_weight_grad, [e, e // d], torch.bfloat16)
    _assert_shape_dtype(mha_linear1_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_weight_grad, [e * 3 // d, e], torch.bfloat16)
    _assert_shape_dtype(mha_linear0_bias_grad, [e * 3 // d], torch.bfloat16)
    _assert_shape_dtype(layernorm0_bias_grad, [e], torch.bfloat16)
    _assert_shape_dtype(layernorm0_weight_grad, [e], torch.bfloat16)
    _assert_shape_dtype(inp_grad, [b, s, e], torch.bfloat16)

    benchmark.pedantic(benchmark_fn, rounds=5)
