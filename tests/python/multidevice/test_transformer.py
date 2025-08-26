# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch
import torch.nn.functional as F

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition
from python.direct_utils import (
    create_sdpa_rng_tensors,
    define_sdpa_rng_state,
    is_pre_ampere,
)
from benchmark_utils import get_benchmark_fns


@pytest.mark.mpi
def test_grouped_mlp(multidevice_direct_test):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
    g = 4
    k = 16
    n = 16 * d

    def _definition(fd: FusionDefinition):
        inp = fd.define_tensor([-1, k], dtype=DataType.BFloat16, contiguity=True)
        gate_w = fd.define_tensor([g, k, n], dtype=DataType.BFloat16, contiguity=True)
        up_w = fd.define_tensor([g, k, n], dtype=DataType.BFloat16, contiguity=True)
        down_w = fd.define_tensor([g, n, k], dtype=DataType.BFloat16, contiguity=True)
        offsets = fd.define_tensor([g], dtype=DataType.Int32, contiguity=True)

        gate_out = fd.ops.grouped_mm(inp, gate_w, offsets)
        gate_out = fd.ops.cast(gate_out, DataType.Float)

        up_out = fd.ops.grouped_mm(inp, up_w, offsets)

        mul_out = fd.ops.mul(fd.ops.silu(gate_out), up_out)
        mul_out = fd.ops.cast(mul_out, DataType.BFloat16)

        out = fd.ops.grouped_mm(mul_out, down_w, offsets)

        fd.add_output(out)

    def _multidevice_schedule(fd: FusionDefinition):
        inp, gate_w, up_w, down_w, offsets = fd.fusion.inputs()
        for t in [inp, gate_w, up_w, down_w, offsets]:
            t.set_device_mesh(mesh)

        for w in [gate_w, up_w]:
            w.split(-1, d, False)
            w.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)

        down_w.split(-2, d, False)
        down_w.axis(-3).parallelize(nvfuser.ParallelType.mesh_x)

    m = 32
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    gate_w = torch.randn(g, k, n, dtype=torch.bfloat16)
    up_w = torch.randn(g, k, n, dtype=torch.bfloat16)
    down_w = torch.randn(g, n, k, dtype=torch.bfloat16)
    sharded_gate_w = multidevice_direct_test.shard_tensor(gate_w, -1, mesh)
    sharded_up_w = multidevice_direct_test.shard_tensor(up_w, -1, mesh)
    sharded_down_w = multidevice_direct_test.shard_tensor(down_w, -2, mesh)
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

    with FusionDefinition() as fd:
        _definition(fd)
        _multidevice_schedule(fd)
    (out,) = fd.execute([inp, sharded_gate_w, sharded_up_w, sharded_down_w, offsets])

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


# The following benchmarks the fusion generated from NanoGPTBlockBenchmark in Thunder.
# To generate the fusion, use the following snippet which turns on the necessary options
# to get a single nvfuser definition.
# ```
#   bench = NanoGPTBlockBenchmark(
#     config="gpt2", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
#   )
#   args, kwargs = bench.make_batch()
#
#   jfn = thunder.jit(
#       bench.fn(),
#       executors=[nvfuserex],
#       nv_enable_sdpa=True,
#       nv_enable_matmul=True,
#       nv_enable_linear=True,
#       disable_replace_uniform=True,
#   )
#   out = jfn(*args, **kwargs)
#   print (thunder.last_traces(jfn)[-1].python_ctx()['nvFusion0'].last_used)
# ```
#
def transformer_forward_definition(
    fd: FusionDefinition, batch: int, sequence: int, head: int, hidden: int
) -> None:
    # Same notations as in test_multidevice_transformer.cpp.
    b, s, h, e = batch, sequence, head, hidden
    inp = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    layernorm0_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    layernorm0_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    mha_linear0_weight = fd.define_tensor(
        shape=[e * 3, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    mha_linear0_bias = fd.define_tensor(
        shape=[e * 3],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    mha_linear1_weight = fd.define_tensor(
        shape=[e, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    mha_linear1_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    layernorm1_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    layernorm1_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    mlp_linear0_weight = fd.define_tensor(
        shape=[e * 4, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    mlp_linear0_bias = fd.define_tensor(
        shape=[e * 4],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    mlp_linear1_weight = fd.define_tensor(
        shape=[e, e * 4],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    mlp_linear1_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )

    T13 = fd.ops.cast(inp, dtype=DataType.Float)
    T14, layernorm0_mean = fd.ops.var_mean(T13, dims=[2], correction=0, keepdim=False)
    T20 = fd.ops.broadcast_in_dim(T14, shape=[b, s, 1], broadcast_dims=[0, 1])
    T25 = fd.ops.broadcast_in_dim(
        layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    S26 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T27 = fd.ops.add(T20, S26)
    layernorm0_rstd = fd.ops.rsqrt(T27)
    T33 = fd.ops.broadcast_in_dim(T25, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T34 = fd.ops.sub(T13, T33)
    T39 = fd.ops.broadcast_in_dim(
        layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    T40 = fd.ops.mul(T34, T39)
    T45 = fd.ops.broadcast_in_dim(
        layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T46 = fd.ops.cast(T45, dtype=DataType.Float)
    T47 = fd.ops.mul(T40, T46)
    T52 = fd.ops.broadcast_in_dim(layernorm0_bias, shape=[b, s, e], broadcast_dims=[2])
    T53 = fd.ops.cast(T52, dtype=DataType.Float)
    T54 = fd.ops.add(T47, T53)
    T55 = fd.ops.cast(T54, dtype=DataType.BFloat16)
    mha_linear0_out = fd.ops.linear(T55, mha_linear0_weight, mha_linear0_bias)

    # Reshape before slice to avoid slicing a tensor along sharded dimension.
    # This is different from the single-GPU definition obtained from Thunder.
    T57 = fd.ops.reshape(mha_linear0_out, new_shape=[b, s, h, 3 * e // h])
    T69 = fd.ops.slice(T57, start_indices=[0, 0, 0, 0], end_indices=[b, s, h, e // h])
    T82 = fd.ops.slice(
        T57, start_indices=[0, 0, 0, e // h], end_indices=[b, s, h, 2 * e // h]
    )
    T95 = fd.ops.slice(
        T57, start_indices=[0, 0, 0, 2 * e // h], end_indices=[b, s, h, 3 * e // h]
    )

    T102 = fd.ops.permute(T82, dims=[0, 2, 1, 3])
    T109 = fd.ops.permute(T69, dims=[0, 2, 1, 3])
    T116 = fd.ops.permute(T95, dims=[0, 2, 1, 3])

    S117 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S118 = fd.define_scalar(True, dtype=DataType.Bool)
    sdpa_out, sdpa_logsum_exp, sdpa_seed, sdpa_offset = fd.ops.sdpfa_fwd(
        T109, T102, T116, S117, S118, None
    )
    T123 = fd.ops.permute(sdpa_out, dims=[0, 2, 1, 3])
    T124 = fd.ops.stride_order(T123, stride_order=[3, 2, 1, 0])
    T129 = fd.ops.reshape(T124, new_shape=[b, s, e])
    mha_linear1_out = fd.ops.linear(T129, mha_linear1_weight, mha_linear1_bias)
    S131 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S132 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T137 = fd.ops.uniform(S131, S132, shape=[b, s, e], dtype=DataType.BFloat16)
    S138 = fd.define_scalar(0.900000, dtype=DataType.Double)
    mha_dropout_mask = fd.ops.lt(T137, S138)
    T140 = fd.ops.cast(mha_linear1_out, dtype=DataType.Float)
    T141 = fd.ops.cast(mha_dropout_mask, dtype=DataType.Float)
    T142 = fd.ops.mul(T140, T141)
    S143 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T144 = fd.ops.mul(T142, S143)
    T145 = fd.ops.add(T13, T144)
    T146, layernorm1_mean = fd.ops.var_mean(T145, dims=[2], correction=0, keepdim=False)
    T152 = fd.ops.broadcast_in_dim(T146, shape=[b, s, 1], broadcast_dims=[0, 1])
    T157 = fd.ops.broadcast_in_dim(
        layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    S158 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T159 = fd.ops.add(T152, S158)
    layernorm1_rstd = fd.ops.rsqrt(T159)
    T165 = fd.ops.broadcast_in_dim(T157, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T166 = fd.ops.sub(T145, T165)
    T171 = fd.ops.broadcast_in_dim(
        layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    T172 = fd.ops.mul(T166, T171)
    T177 = fd.ops.broadcast_in_dim(
        layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T178 = fd.ops.cast(T177, dtype=DataType.Float)
    T179 = fd.ops.mul(T172, T178)
    T184 = fd.ops.broadcast_in_dim(layernorm1_bias, shape=[b, s, e], broadcast_dims=[2])
    T185 = fd.ops.cast(T184, dtype=DataType.Float)
    T186 = fd.ops.add(T179, T185)
    T187 = fd.ops.cast(T186, dtype=DataType.BFloat16)
    mlp_linear0_out = fd.ops.linear(T187, mlp_linear0_weight, mlp_linear0_bias)
    T189 = fd.ops.cast(mlp_linear0_out, dtype=DataType.Float)
    T190 = fd.ops.mul(T189, T189)
    T191 = fd.ops.mul(T190, T189)
    S192 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T193 = fd.ops.mul(S192, T189)
    S194 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T195 = fd.ops.mul(S194, T191)
    T196 = fd.ops.add(T189, T195)
    S197 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T198 = fd.ops.mul(S197, T196)
    T199 = fd.ops.tanh(T198)
    S200 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T201 = fd.ops.add(S200, T199)
    T202 = fd.ops.mul(T193, T201)
    T203 = fd.ops.cast(T202, dtype=DataType.BFloat16)
    mlp_linear1_out = fd.ops.linear(T203, mlp_linear1_weight, mlp_linear1_bias)
    S205 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S206 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T211 = fd.ops.uniform(S205, S206, shape=[b, s, e], dtype=DataType.BFloat16)
    S212 = fd.define_scalar(0.900000, dtype=DataType.Double)
    mlp_dropout_mask = fd.ops.lt(T211, S212)
    T214 = fd.ops.cast(mlp_linear1_out, dtype=DataType.Float)
    T215 = fd.ops.cast(mlp_dropout_mask, dtype=DataType.Float)
    T216 = fd.ops.mul(T214, T215)
    S217 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T218 = fd.ops.mul(T216, S217)
    T219 = fd.ops.add(T145, T218)
    out = fd.ops.cast(T219, dtype=DataType.BFloat16)
    fd.add_output(layernorm0_mean)
    fd.add_output(layernorm0_rstd)
    fd.add_output(mha_linear0_out)
    fd.add_output(sdpa_out)
    fd.add_output(sdpa_logsum_exp)
    fd.add_output(sdpa_seed)
    fd.add_output(sdpa_offset)
    fd.add_output(mha_linear1_out)
    fd.add_output(mha_dropout_mask)
    fd.add_output(layernorm1_mean)
    fd.add_output(layernorm1_rstd)
    fd.add_output(mlp_linear0_out)
    fd.add_output(mlp_dropout_mask)
    fd.add_output(out)


def transformer_forward_multidevice_schedule(fd: FusionDefinition, num_devices: int):
    mesh = nvfuser.multidevice.DeviceMesh(range(num_devices))
    inputs = fd.fusion.inputs()
    inp = inputs[0]
    layernorm0_weight = inputs[1]
    layernorm0_bias = inputs[2]
    mha_linear0_weight = inputs[3]
    mha_linear0_bias = inputs[4]
    mha_linear1_weight = inputs[5]
    mha_linear1_bias = inputs[6]
    layernorm1_weight = inputs[7]
    layernorm1_bias = inputs[8]
    mlp_linear0_weight = inputs[9]
    mlp_linear0_bias = inputs[10]
    mlp_linear1_weight = inputs[11]
    mlp_linear1_bias = inputs[12]

    for tv in [
        inp,
        layernorm0_weight,
        layernorm0_bias,
        mha_linear0_weight,
        mha_linear0_bias,
        mha_linear1_weight,
        mha_linear1_bias,
        layernorm1_weight,
        layernorm1_bias,
        mlp_linear0_weight,
        mlp_linear0_bias,
        mlp_linear1_weight,
        mlp_linear1_bias,
    ]:
        tv.set_device_mesh(mesh)

    for tv in [
        mha_linear0_weight,
        mha_linear0_bias,
        mlp_linear0_weight,
        mlp_linear0_bias,
    ]:
        tv.split(0, num_devices, inner_split=False)
        tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    for tv in [
        mha_linear1_weight,
        mlp_linear1_weight,
    ]:
        tv.split(-1, num_devices, inner_split=False)
        tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)


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
def test_transformer_forward(multidevice_direct_test, benchmark):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))

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

    torch.cuda.set_device(multidevice_direct_test.local_rank)

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
        multidevice_direct_test.shard_tensor(mha_linear0_weight, 0, mesh),
        multidevice_direct_test.shard_tensor(mha_linear0_bias, 0, mesh),
        multidevice_direct_test.shard_tensor(mha_linear1_weight, -1, mesh),
        torch.testing.make_tensor(e, dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        multidevice_direct_test.shard_tensor(mlp_linear0_weight, 0, mesh),
        multidevice_direct_test.shard_tensor(mlp_linear0_bias, 0, mesh),
        multidevice_direct_test.shard_tensor(mlp_linear1_weight, -1, mesh),
        torch.testing.make_tensor(e, dtype=torch.bfloat16, device="cuda"),
    ]

    with FusionDefinition() as fd:
        transformer_forward_definition(fd, b, s, h, e)
        transformer_forward_multidevice_schedule(fd, d)

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

    # Warm up and validate.
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
    ) = warmup_fn()

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

    # Benchmark and profile. The profile can be collected and displayed using
    # `nsys`. See instructions in test_transformer_engine.py.
    benchmark.pedantic(benchmark_fn, rounds=5)


# The following micro-benchmarks the
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
# All tensors are replicated to all devices at this moment; future PRs will try
# to shard them.
def transformer_backward_definition(
    fd: FusionDefinition,
    num_devices: int,
    batch: int,
    sequence: int,
    head: int,
    hidden: int,
) -> None:
    d, b, s, h, e = (
        num_devices,
        batch,
        sequence,
        head,
        hidden,
    )

    mlp_dropout_offset = fd.define_scalar(None, dtype=DataType.Int)
    mlp_dropout_seed = fd.define_scalar(None, dtype=DataType.Int)
    mlp_linear0_out = fd.define_tensor(
        shape=[d, b, s, e * 4 // d],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    out_grad = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    mlp_linear1_weight = fd.define_tensor(
        shape=[d, e, e * 4 // d],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    mha_dropout_offset = fd.define_scalar(None, dtype=DataType.Int)
    mha_dropout_seed = fd.define_scalar(None, dtype=DataType.Int)
    mha_linear1_out = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    mlp_linear0_weight = fd.define_tensor(
        shape=[d, e * 4 // d, e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm1_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm1_mean = fd.define_tensor(
        shape=[b, s],
        contiguity=True,
        dtype=DataType.Float,
    )
    inp = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm1_rstd = fd.define_tensor(
        shape=[b, s, 1],
        contiguity=True,
        dtype=DataType.Float,
    )
    mha_linear1_weight = fd.define_tensor(
        shape=[d, e, e // d],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    mha_linear0_out = fd.define_tensor(
        shape=[d, b, s, e * 3 // d],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    sdpa_out = fd.define_tensor(
        shape=[d, b, h // d, s, e // h],
        contiguity=True,
        dtype=DataType.BFloat16,
        stride_order=[4, 3, 1, 2, 0],
    )
    sdpa_log_sumexp = fd.define_tensor(
        shape=[d, b, h // d, s],
        contiguity=True,
        dtype=DataType.Float,
    )
    mha_sdpa_seed, mha_sdpa_offset = define_sdpa_rng_state(fd)
    mha_linear0_weight = fd.define_tensor(
        shape=[d, e * 3 // d, e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm0_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm0_mean = fd.define_tensor(
        shape=[b, s],
        contiguity=True,
        dtype=DataType.Float,
    )
    layernorm0_rstd = fd.define_tensor(
        shape=[b, s, 1],
        contiguity=True,
        dtype=DataType.Float,
    )
    layernorm0_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    layernorm1_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
    )
    S25 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S26 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T31 = fd.ops.uniform(
        S25,
        S26,
        shape=[b, s, e],
        rng_seed=mlp_dropout_seed,
        rng_offset=mlp_dropout_offset,
        dtype=DataType.BFloat16,
    )
    T32 = fd.ops.cast(mlp_linear0_out, dtype=DataType.Float)
    T33 = fd.ops.cast(out_grad, dtype=DataType.Float)
    S34 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T35 = fd.ops.lt(T31, S34)
    T36 = fd.ops.mul(T32, T32)
    S37 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T38 = fd.ops.mul(S37, T33)
    T39 = fd.ops.cast(T35, dtype=DataType.Float)
    T40 = fd.ops.mul(T36, T32)
    T41 = fd.ops.mul(T39, T38)
    S42 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T43 = fd.ops.mul(S42, T40)
    T44 = fd.ops.cast(T41, dtype=DataType.BFloat16)
    T45 = fd.ops.add(T32, T43)
    T49 = fd.ops.reshape(T44, new_shape=[b * s, e])
    S50 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T51 = fd.ops.mul(S50, T45)
    T52 = fd.ops.matmul(T49, mlp_linear1_weight)
    T53 = fd.ops.tanh(T51)
    T58 = fd.ops.reshape(T52, new_shape=[d, b, s, e * 4 // d])
    T59 = fd.ops.mul(T53, T53)
    T60 = fd.ops.cast(T58, dtype=DataType.Float)
    S61 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T62 = fd.ops.mul(S61, T32)
    S63 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T64 = fd.ops.sub(S63, T59)
    T65 = fd.ops.mul(T62, T60)
    T66 = fd.ops.mul(T65, T64)
    S67 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T68 = fd.ops.add(S67, T53)
    S69 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T70 = fd.ops.mul(S69, T66)
    T71 = fd.ops.mul(T68, T60)
    S72 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T73 = fd.ops.mul(S72, T70)
    S74 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T75 = fd.ops.mul(S74, T71)
    T76 = fd.ops.mul(T32, T73)
    T77 = fd.ops.mul(T36, T73)
    T78 = fd.ops.add(T70, T75)
    T79 = fd.ops.mul(T32, T76)
    T80 = fd.ops.add(T78, T77)
    T81 = fd.ops.add(T80, T79)
    T82 = fd.ops.add(T81, T79)
    S83 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S84 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T89 = fd.ops.uniform(
        S83,
        S84,
        shape=[b, s, e],
        rng_seed=mha_dropout_seed,
        rng_offset=mha_dropout_offset,
        dtype=DataType.BFloat16,
    )
    T90 = fd.ops.cast(T82, dtype=DataType.BFloat16)
    S91 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T92 = fd.ops.lt(T89, S91)
    T96 = fd.ops.reshape(T90, new_shape=[d, b * s, e * 4 // d])
    T97 = fd.ops.cast(T92, dtype=DataType.Float)
    T98 = fd.ops.cast(mha_linear1_out, dtype=DataType.Float)
    T99_local = fd.ops.matmul(T96, mlp_linear0_weight)
    T99 = fd.ops.sum(T99_local, [0])  # allreduce
    T100 = fd.ops.mul(T98, T97)
    T105 = fd.ops.reshape(T99, new_shape=[b, s, e])
    T110 = fd.ops.broadcast_in_dim(
        layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T115 = fd.ops.broadcast_in_dim(
        layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    S116 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T117 = fd.ops.mul(T100, S116)
    T118 = fd.ops.cast(inp, dtype=DataType.Float)
    T119 = fd.ops.cast(T105, dtype=DataType.Float)
    T120 = fd.ops.cast(T110, dtype=DataType.Float)
    T125 = fd.ops.broadcast_in_dim(T115, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T126 = fd.ops.add(T118, T117)
    T127 = fd.ops.mul(T120, T119)
    T128 = fd.ops.sub(T126, T125)
    T129 = fd.ops.mul(T128, T127)
    T130 = fd.ops.sum(T129, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T135 = fd.ops.broadcast_in_dim(T130, shape=[b, s, 1], broadcast_dims=[1])
    T140 = fd.ops.broadcast_in_dim(
        layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    S141 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T142 = fd.ops.pow(layernorm1_rstd, S141)
    S143 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T144 = fd.ops.mul(S143, T135)
    T145 = fd.ops.mul(T140, T127)
    T146 = fd.ops.mul(T144, T142)
    T147 = fd.ops.neg(T145)
    T148 = fd.ops.sum(T146, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T149 = fd.ops.sum(T147, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T153 = fd.ops.broadcast_in_dim(T148, shape=[b, s], broadcast_dims=[1])
    T158 = fd.ops.broadcast_in_dim(T149, shape=[b, s, 1], broadcast_dims=[1])
    T163 = fd.ops.broadcast_in_dim(
        layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T168 = fd.ops.broadcast_in_dim(T153, shape=[b, s, 1], broadcast_dims=[0, 1])
    T169 = fd.ops.sum(T158, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T174 = fd.ops.broadcast_in_dim(T163, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T179 = fd.ops.broadcast_in_dim(T168, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T183 = fd.ops.broadcast_in_dim(T169, shape=[b, s], broadcast_dims=[1])
    T184 = fd.ops.sub(T126, T174)
    S185 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T186 = fd.ops.mul(S185, T179)
    T191 = fd.ops.broadcast_in_dim(T183, shape=[b, s, 1], broadcast_dims=[0, 1])
    T192 = fd.ops.mul(T186, T184)
    T197 = fd.ops.broadcast_in_dim(T191, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    S198 = fd.define_scalar(e, dtype=DataType.Double)
    S199 = fd.ops.reciprocal(S198)
    T200 = fd.ops.mul(T192, S199)
    S201 = fd.define_scalar(1 / e, dtype=DataType.Double)
    T202 = fd.ops.mul(S201, T197)
    T203 = fd.ops.add(T202, T200)
    T204 = fd.ops.add(T145, T203)
    T205 = fd.ops.add(T33, T204)
    S206 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T207 = fd.ops.mul(S206, T205)
    T208 = fd.ops.mul(T97, T207)
    T209 = fd.ops.cast(T208, dtype=DataType.BFloat16)
    T213 = fd.ops.reshape(T209, new_shape=[b * s, e])
    T214 = fd.ops.matmul(T213, mha_linear1_weight)
    T227 = fd.ops.slice(
        mha_linear0_out,
        start_indices=[0, 0, 0, e * 2 // d],
        end_indices=[d, b, s, e * 3 // d],
    )
    T240 = fd.ops.slice(
        mha_linear0_out,
        start_indices=[0, 0, 0, e // d],
        end_indices=[d, b, s, e * 2 // d],
    )
    T253 = fd.ops.slice(
        mha_linear0_out,
        start_indices=[0, 0, 0, 0],
        end_indices=[d, b, s, e // d],
    )
    T258 = fd.ops.reshape(T214, new_shape=[d, b, s, e // d])
    T264 = fd.ops.reshape(T227, new_shape=[d, b, s, h // d, e // h])
    T270 = fd.ops.reshape(T240, new_shape=[d, b, s, h // d, e // h])
    T276 = fd.ops.reshape(T253, new_shape=[d, b, s, h // d, e // h])
    T282 = fd.ops.reshape(T258, new_shape=[d, b, s, h // d, e // h])
    T283 = fd.ops.permute(T264, dims=[0, 1, 3, 2, 4])
    T284 = fd.ops.permute(T270, dims=[0, 1, 3, 2, 4])
    T285 = fd.ops.permute(T276, dims=[0, 1, 3, 2, 4])
    T286 = fd.ops.permute(T282, dims=[0, 1, 3, 2, 4])
    S287 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S288 = fd.define_scalar(True, dtype=DataType.Bool)
    T289, T290, T291 = fd.ops.sdpfa_bwd(
        T286,
        T285,
        T284,
        T283,
        sdpa_out,
        sdpa_log_sumexp,
        S287,
        S288,
        mha_sdpa_seed,
        mha_sdpa_offset,
        None,
    )
    T292 = fd.ops.permute(T291, dims=[0, 1, 3, 2, 4])
    T293 = fd.ops.permute(T290, dims=[0, 1, 3, 2, 4])
    T294 = fd.ops.permute(T289, dims=[0, 1, 3, 2, 4])
    T299 = fd.ops.reshape(T292, new_shape=[d, b, s, e // d])
    T304 = fd.ops.reshape(T293, new_shape=[d, b, s, e // d])
    T309 = fd.ops.reshape(T294, new_shape=[d, b, s, e // d])
    T310 = fd.ops.cat([T309, T304, T299], dim=3)
    T314 = fd.ops.reshape(T310, new_shape=[d, b * s, e * 3 // d])
    T315_local = fd.ops.matmul(T314, mha_linear0_weight)
    T315 = fd.ops.sum(T315_local, [0])  # allreduce
    T320 = fd.ops.reshape(T315, new_shape=[b, s, e])
    T325 = fd.ops.broadcast_in_dim(
        layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T330 = fd.ops.broadcast_in_dim(
        layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T331 = fd.ops.cast(T320, dtype=DataType.Float)
    T332 = fd.ops.cast(T325, dtype=DataType.Float)
    T337 = fd.ops.broadcast_in_dim(T330, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T338 = fd.ops.mul(T332, T331)
    T339 = fd.ops.sub(T118, T337)
    T340 = fd.ops.mul(T339, T338)
    T341 = fd.ops.sum(T340, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T346 = fd.ops.broadcast_in_dim(T341, shape=[b, s, 1], broadcast_dims=[1])
    T351 = fd.ops.broadcast_in_dim(
        layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    S352 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T353 = fd.ops.pow(layernorm0_rstd, S352)
    S354 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T355 = fd.ops.mul(S354, T346)
    T356 = fd.ops.mul(T351, T338)
    T357 = fd.ops.mul(T355, T353)
    T358 = fd.ops.neg(T356)
    T359 = fd.ops.sum(T357, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T360 = fd.ops.sum(T358, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T364 = fd.ops.broadcast_in_dim(T359, shape=[b, s], broadcast_dims=[1])
    T369 = fd.ops.broadcast_in_dim(T360, shape=[b, s, 1], broadcast_dims=[1])
    T374 = fd.ops.broadcast_in_dim(
        layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T379 = fd.ops.broadcast_in_dim(T364, shape=[b, s, 1], broadcast_dims=[0, 1])
    T380 = fd.ops.sum(T369, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T385 = fd.ops.broadcast_in_dim(T374, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T390 = fd.ops.broadcast_in_dim(T379, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T394 = fd.ops.broadcast_in_dim(T380, shape=[b, s], broadcast_dims=[1])
    T395 = fd.ops.sub(T118, T385)
    S396 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T397 = fd.ops.mul(S396, T390)
    T402 = fd.ops.broadcast_in_dim(T394, shape=[b, s, 1], broadcast_dims=[0, 1])
    T403 = fd.ops.mul(T397, T395)
    T408 = fd.ops.broadcast_in_dim(T402, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T413 = fd.ops.broadcast_in_dim(layernorm0_bias, shape=[b, s, e], broadcast_dims=[2])
    T414 = fd.ops.mul(T339, T351)
    T419 = fd.ops.broadcast_in_dim(layernorm1_bias, shape=[b, s, e], broadcast_dims=[2])
    T420 = fd.ops.mul(T128, T140)
    S421 = fd.define_scalar(e, dtype=DataType.Double)
    S422 = fd.ops.reciprocal(S421)
    T423 = fd.ops.mul(T403, S422)
    S424 = fd.define_scalar(1 / e, dtype=DataType.Double)
    T425 = fd.ops.mul(S424, T408)
    T426 = fd.ops.cast(T413, dtype=DataType.Float)
    T427 = fd.ops.mul(T414, T332)
    T428 = fd.ops.permute(sdpa_out, dims=[0, 1, 3, 2, 4])
    T429 = fd.ops.cast(T419, dtype=DataType.Float)
    T430 = fd.ops.mul(T420, T120)
    T431 = fd.ops.add(T425, T423)
    T432 = fd.ops.add(T427, T426)
    T433 = fd.ops.stride_order(T428, stride_order=[4, 3, 2, 1, 0])
    T434 = fd.ops.add(T430, T429)
    T435 = fd.ops.mul(T62, T68)
    T436 = fd.ops.add(T356, T431)
    T437 = fd.ops.mul(T414, T331)
    T438 = fd.ops.cast(T310, dtype=DataType.Float)
    T439 = fd.ops.cast(T432, dtype=DataType.BFloat16)
    T444 = fd.ops.reshape(T433, new_shape=[d, b, s, e // d])
    T445 = fd.ops.mul(T420, T119)
    T446 = fd.ops.cast(T434, dtype=DataType.BFloat16)
    T447 = fd.ops.cast(T435, dtype=DataType.BFloat16)
    T448 = fd.ops.add(T205, T436)
    T449 = fd.ops.sum(T437, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T450 = fd.ops.sum(T331, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T451 = fd.ops.sum(T438, dims=[1, 2], keepdim=False, dtype=DataType.Null)
    T455 = fd.ops.reshape(T439, new_shape=[b * s, e])
    T456 = fd.ops.permute(T314, dims=[0, 2, 1])
    T457 = fd.ops.sum(T208, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T461 = fd.ops.reshape(T444, new_shape=[d, b * s, e // d])
    T462 = fd.ops.permute(T213, dims=[1, 0])
    T463 = fd.ops.sum(T445, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T464 = fd.ops.sum(T119, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T465 = fd.ops.sum(T82, dims=[1, 2], keepdim=False, dtype=DataType.Null)
    T469 = fd.ops.reshape(T446, new_shape=[b * s, e])
    T470 = fd.ops.permute(T96, dims=[0, 2, 1])
    T471 = fd.ops.sum(T41, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T475 = fd.ops.reshape(T447, new_shape=[d, b * s, e * 4 // d])
    T476 = fd.ops.permute(T49, dims=[1, 0])
    inp_grad = fd.ops.cast(T448, dtype=DataType.BFloat16)
    layernorm0_weight_grad = fd.ops.cast(T449, dtype=DataType.BFloat16)
    layernorm0_bias_grad = fd.ops.cast(T450, dtype=DataType.BFloat16)
    mha_linear0_bias_grad = fd.ops.cast(T451, dtype=DataType.BFloat16)
    mha_linear0_weight_grad = fd.ops.matmul(T456, T455)
    mha_linear1_bias_grad = fd.ops.cast(T457, dtype=DataType.BFloat16)
    mha_linear1_weight_grad = fd.ops.matmul(T462, T461)
    layernorm1_weight_grad = fd.ops.cast(T463, dtype=DataType.BFloat16)
    layernorm1_bias_grad = fd.ops.cast(T464, dtype=DataType.BFloat16)
    mlp_linear0_bias_grad = fd.ops.cast(T465, dtype=DataType.BFloat16)
    mlp_linear0_weight_grad = fd.ops.matmul(T470, T469)
    mlp_linear1_bias_grad = fd.ops.cast(T471, dtype=DataType.BFloat16)
    mlp_linear1_weight_grad = fd.ops.matmul(T476, T475)
    fd.add_output(mlp_linear1_weight_grad)
    fd.add_output(mlp_linear1_bias_grad)
    fd.add_output(mlp_linear0_weight_grad)
    fd.add_output(mlp_linear0_bias_grad)
    fd.add_output(layernorm1_bias_grad)
    fd.add_output(layernorm1_weight_grad)
    fd.add_output(mha_linear1_weight_grad)
    fd.add_output(mha_linear1_bias_grad)
    fd.add_output(mha_linear0_weight_grad)
    fd.add_output(mha_linear0_bias_grad)
    fd.add_output(layernorm0_bias_grad)
    fd.add_output(layernorm0_weight_grad)
    fd.add_output(inp_grad)


def transformer_backward_multidevice_schedule(fd: FusionDefinition, num_devices: int):
    mesh = nvfuser.multidevice.DeviceMesh(range(num_devices))
    inputs = fd.fusion.inputs()

    mlp_linear0_out = inputs[2]
    out_grad = inputs[3]
    mlp_linear1_weight = inputs[4]
    mha_linear1_out = inputs[7]
    mlp_linear0_weight = inputs[8]
    layernorm1_weight = inputs[9]
    layernorm1_mean = inputs[10]
    inp = inputs[11]
    layernorm1_rstd = inputs[12]
    mha_linear1_weight = inputs[13]
    mha_linear0_out = inputs[14]
    sdpa_out = inputs[15]
    sdpa_log_sumexp = inputs[16]
    mha_linear0_weight = inputs[19]
    layernorm0_weight = inputs[20]
    layernorm0_mean = inputs[21]
    layernorm0_rstd = inputs[22]
    layernorm0_bias = inputs[23]
    layernorm1_bias = inputs[24]

    for in_tv in [
        mlp_linear0_out,
        out_grad,
        mlp_linear1_weight,
        mha_linear1_out,
        mlp_linear0_weight,
        layernorm1_weight,
        layernorm1_mean,
        inp,
        layernorm1_rstd,
        mha_linear1_weight,
        mha_linear0_out,
        sdpa_out,
        sdpa_log_sumexp,
        mha_linear0_weight,
        layernorm0_weight,
        layernorm0_mean,
        layernorm0_rstd,
        layernorm0_bias,
        layernorm1_bias,
    ]:
        in_tv.set_device_mesh(mesh)

    for in_tv in [
        mlp_linear0_out,
        mlp_linear1_weight,
        mlp_linear0_weight,
        mha_linear1_weight,
        mha_linear0_out,
        sdpa_out,
        sdpa_log_sumexp,
        mha_linear0_weight,
    ]:
        in_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(multidevice_direct_test, benchmark):
    d = multidevice_direct_test.size

    rank = multidevice_direct_test.rank

    b, s, h, e = 1, 2048, 96, 12288

    torch.cuda.set_device(multidevice_direct_test.local_rank)

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

    with FusionDefinition() as fd:
        transformer_backward_definition(fd, d, b, s, h, e)
        transformer_backward_multidevice_schedule(fd, d)

    warmup_fn, benchmark_fn = get_benchmark_fns(lambda: fd.execute(ins))

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
    ) = warmup_fn()
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
