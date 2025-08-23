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
#   from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig
#   from thunder.executors.nvfuserex import nvfuserex

#   config = NanoGPTConfig(seq_len=2048, n_head=96, n_embd=12288)
#   bench = NanoGPTBlockBenchmark(
#     batchdims=(1,), config=config, device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
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
#   grads = torch.randn_like(out, device="cuda", dtype=torch.bfloat16)
#   out.backward(grads)
#   print(thunder.last_traces(jfn)[-1].python_ctx()['nvFusion0'].last_used)
#   print(thunder.last_backward_traces(jfn)[-1].python_ctx()['nvFusion0'].last_used)
# ```
# Fusions generated from Thunder commit: b0dc72ef1a9825a70923ae1a270d919f5948c4ed


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


def transformer_backward_definition(
    fd: FusionDefinition, batch: int, sequence: int, head: int, hidden: int
) -> None:
    b, s, h, e = batch, sequence, head, hidden

    fd.mlp_linear0_out = fd.define_tensor(
        shape=[b, s, 4 * e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.out_grad = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.mlp_dropout_mask = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.Bool,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.mlp_linear1_weight = fd.define_tensor(
        shape=[e, e * 4],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.mha_dropout_mask = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.Bool,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.mha_linear1_out = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.mlp_linear0_weight = fd.define_tensor(
        shape=[4 * e, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.layernorm1_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    fd.layernorm1_mean = fd.define_tensor(
        shape=[b, s],
        contiguity=True,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.inp = fd.define_tensor(
        shape=[b, s, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.layernorm1_rstd = fd.define_tensor(
        shape=[b, s, 1],
        contiguity=True,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.mha_linear1_weight = fd.define_tensor(
        shape=[e, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.mha_linear0_out = fd.define_tensor(
        shape=[b, s, 3 * e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.sdpa_out = fd.define_tensor(
        shape=[b, h, s, e // h],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    fd.sdpa_logsum_exp = fd.define_tensor(
        shape=[b, h, s],
        contiguity=True,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.sdpa_seed = fd.define_tensor(
        shape=[2],
        contiguity=True,
        dtype=DataType.UInt64,
        is_cpu=False,
        stride_order=[0],
    )
    fd.sdpa_offset = fd.define_tensor(
        shape=[], contiguity=[], dtype=DataType.UInt64, is_cpu=False
    )
    fd.mha_linear0_weight = fd.define_tensor(
        shape=[3 * e, e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.layernorm0_weight = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    fd.layernorm0_mean = fd.define_tensor(
        shape=[b, s],
        contiguity=True,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    fd.layernorm0_rstd = fd.define_tensor(
        shape=[b, s, 1],
        contiguity=True,
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    fd.layernorm0_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    fd.layernorm1_bias = fd.define_tensor(
        shape=[e],
        contiguity=True,
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T23 = fd.ops.cast(fd.mlp_linear0_out, dtype=DataType.Float)
    T24 = fd.ops.cast(fd.out_grad, dtype=DataType.Float)
    T25 = fd.ops.mul(T23, T23)
    S26 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T27 = fd.ops.mul(S26, T24)
    T28 = fd.ops.cast(fd.mlp_dropout_mask, dtype=DataType.Float)
    T29 = fd.ops.mul(T25, T23)
    T30 = fd.ops.mul(T28, T27)
    S31 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T32 = fd.ops.mul(S31, T29)
    T33 = fd.ops.cast(T30, dtype=DataType.BFloat16)
    T34 = fd.ops.add(T23, T32)
    T38 = fd.ops.reshape(T33, new_shape=[b * s, e])
    S39 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T40 = fd.ops.mul(S39, T34)
    T41 = fd.ops.matmul(T38, fd.mlp_linear1_weight)
    T42 = fd.ops.tanh(T40)
    T47 = fd.ops.reshape(T41, new_shape=[b, s, 4 * e])
    T48 = fd.ops.mul(T42, T42)
    T49 = fd.ops.cast(T47, dtype=DataType.Float)
    S50 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T51 = fd.ops.mul(S50, T23)
    S52 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T53 = fd.ops.sub(S52, T48)
    T54 = fd.ops.mul(T51, T49)
    T55 = fd.ops.mul(T54, T53)
    S56 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T57 = fd.ops.add(S56, T42)
    S58 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T59 = fd.ops.mul(S58, T55)
    T60 = fd.ops.mul(T57, T49)
    S61 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T62 = fd.ops.mul(S61, T59)
    S63 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T64 = fd.ops.mul(S63, T60)
    T65 = fd.ops.mul(T23, T62)
    T66 = fd.ops.mul(T25, T62)
    T67 = fd.ops.add(T59, T64)
    T68 = fd.ops.mul(T23, T65)
    T69 = fd.ops.add(T67, T66)
    T70 = fd.ops.add(T69, T68)
    T71 = fd.ops.add(T70, T68)
    T72 = fd.ops.cast(T71, dtype=DataType.BFloat16)
    T76 = fd.ops.reshape(T72, new_shape=[b * s, e * 4])
    T77 = fd.ops.cast(fd.mha_dropout_mask, dtype=DataType.Float)
    T78 = fd.ops.cast(fd.mha_linear1_out, dtype=DataType.Float)
    T79 = fd.ops.matmul(T76, fd.mlp_linear0_weight)
    T80 = fd.ops.mul(T78, T77)
    T85 = fd.ops.reshape(T79, new_shape=[b, s, e])
    T90 = fd.ops.broadcast_in_dim(
        fd.layernorm1_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T95 = fd.ops.broadcast_in_dim(
        fd.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    S96 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T97 = fd.ops.mul(T80, S96)
    T98 = fd.ops.cast(fd.inp, dtype=DataType.Float)
    T99 = fd.ops.cast(T85, dtype=DataType.Float)
    T100 = fd.ops.cast(T90, dtype=DataType.Float)
    T105 = fd.ops.broadcast_in_dim(T95, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T106 = fd.ops.add(T98, T97)
    T107 = fd.ops.mul(T100, T99)
    T108 = fd.ops.sub(T106, T105)
    T109 = fd.ops.mul(T108, T107)
    T110 = fd.ops.sum(T109, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T115 = fd.ops.broadcast_in_dim(T110, shape=[b, s, 1], broadcast_dims=[1])
    T120 = fd.ops.broadcast_in_dim(
        fd.layernorm1_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    S121 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T122 = fd.ops.pow(fd.layernorm1_rstd, S121)
    S123 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T124 = fd.ops.mul(S123, T115)
    T125 = fd.ops.mul(T120, T107)
    T126 = fd.ops.mul(T124, T122)
    T127 = fd.ops.neg(T125)
    T128 = fd.ops.sum(T126, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T129 = fd.ops.sum(T127, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T133 = fd.ops.broadcast_in_dim(T128, shape=[b, s], broadcast_dims=[1])
    T138 = fd.ops.broadcast_in_dim(T129, shape=[b, s, 1], broadcast_dims=[1])
    T143 = fd.ops.broadcast_in_dim(
        fd.layernorm1_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T148 = fd.ops.broadcast_in_dim(T133, shape=[b, s, 1], broadcast_dims=[0, 1])
    T149 = fd.ops.sum(T138, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T154 = fd.ops.broadcast_in_dim(T143, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T159 = fd.ops.broadcast_in_dim(T148, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T163 = fd.ops.broadcast_in_dim(T149, shape=[b, s], broadcast_dims=[1])
    T164 = fd.ops.sub(T106, T154)
    S165 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T166 = fd.ops.mul(S165, T159)
    T171 = fd.ops.broadcast_in_dim(T163, shape=[b, s, 1], broadcast_dims=[0, 1])
    T172 = fd.ops.mul(T166, T164)
    T177 = fd.ops.broadcast_in_dim(T171, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    S178 = fd.define_scalar(float(e), dtype=DataType.Double)
    S179 = fd.ops.reciprocal(S178)
    T180 = fd.ops.mul(T172, S179)
    S181 = fd.define_scalar(1 / e, dtype=DataType.Double)
    T182 = fd.ops.mul(S181, T177)
    T183 = fd.ops.add(T182, T180)
    T184 = fd.ops.add(T24, T125)
    T185 = fd.ops.add(T184, T183)
    S186 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T187 = fd.ops.mul(S186, T185)
    T188 = fd.ops.mul(T77, T187)
    T189 = fd.ops.cast(T188, dtype=DataType.BFloat16)
    T193 = fd.ops.reshape(T189, new_shape=[b * s, e])
    T194 = fd.ops.matmul(T193, fd.mha_linear1_weight)

    # Reshape before slicing to avoid slicing along sharded dimensions.
    T195 = fd.ops.reshape(fd.mha_linear0_out, new_shape=[b, s, h, 3 * e // h])
    T244 = fd.ops.slice(
        T195,
        start_indices=[0, 0, 0, 2 * e // h],
        end_indices=[b, s, h, 3 * e // h],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T250 = fd.ops.slice(
        T195,
        start_indices=[0, 0, 0, e // h],
        end_indices=[b, s, h, 2 * e // h],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T256 = fd.ops.slice(
        T195,
        start_indices=[0, 0, 0, 0],
        end_indices=[b, s, h, e // h],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T238 = fd.ops.reshape(T194, new_shape=[b, s, e])
    T262 = fd.ops.reshape(T238, new_shape=[b, s, h, e // h])
    T263 = fd.ops.permute(T244, dims=[0, 2, 1, 3])
    T264 = fd.ops.permute(T250, dims=[0, 2, 1, 3])
    T265 = fd.ops.permute(T256, dims=[0, 2, 1, 3])
    T266 = fd.ops.permute(T262, dims=[0, 2, 1, 3])
    S267 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S268 = fd.define_scalar(True, dtype=DataType.Bool)
    T269, T270, T271 = fd.ops.sdpfa_bwd(
        T266,
        T265,
        T264,
        T263,
        fd.sdpa_out,
        fd.sdpa_logsum_exp,
        S267,
        S268,
        fd.sdpa_seed,
        fd.sdpa_offset,
        None,
    )
    T272 = fd.ops.permute(T271, dims=[0, 2, 1, 3])
    T273 = fd.ops.permute(T270, dims=[0, 2, 1, 3])
    T274 = fd.ops.permute(T269, dims=[0, 2, 1, 3])
    # Cat before reshape to avoid concatenating along sharded dimensions.

    T290 = fd.ops.cat([T274, T273, T272], dim=3, manual_padding=0)
    T291 = fd.ops.reshape(T290, new_shape=[b, s, 3 * e])
    T294 = fd.ops.reshape(T291, new_shape=[b * s, 3 * e])
    T295 = fd.ops.matmul(T294, fd.mha_linear0_weight)
    T300 = fd.ops.reshape(T295, new_shape=[b, s, e])
    T305 = fd.ops.broadcast_in_dim(
        fd.layernorm0_weight, shape=[b, s, e], broadcast_dims=[2]
    )
    T310 = fd.ops.broadcast_in_dim(
        fd.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T311 = fd.ops.cast(T300, dtype=DataType.Float)
    T312 = fd.ops.cast(T305, dtype=DataType.Float)
    T317 = fd.ops.broadcast_in_dim(T310, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T318 = fd.ops.mul(T312, T311)
    T319 = fd.ops.sub(T98, T317)
    T320 = fd.ops.mul(T319, T318)
    T321 = fd.ops.sum(T320, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T326 = fd.ops.broadcast_in_dim(T321, shape=[b, s, 1], broadcast_dims=[1])
    T331 = fd.ops.broadcast_in_dim(
        fd.layernorm0_rstd, shape=[b, s, e], broadcast_dims=[0, 1, 2]
    )
    S332 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T333 = fd.ops.pow(fd.layernorm0_rstd, S332)
    S334 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T335 = fd.ops.mul(S334, T326)
    T336 = fd.ops.mul(T331, T318)
    T337 = fd.ops.mul(T335, T333)
    T338 = fd.ops.neg(T336)
    T339 = fd.ops.sum(T337, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T340 = fd.ops.sum(T338, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T344 = fd.ops.broadcast_in_dim(T339, shape=[b, s], broadcast_dims=[1])
    T349 = fd.ops.broadcast_in_dim(T340, shape=[b, s, 1], broadcast_dims=[1])
    T354 = fd.ops.broadcast_in_dim(
        fd.layernorm0_mean, shape=[b, s, 1], broadcast_dims=[0, 1]
    )
    T359 = fd.ops.broadcast_in_dim(T344, shape=[b, s, 1], broadcast_dims=[0, 1])
    T360 = fd.ops.sum(T349, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T365 = fd.ops.broadcast_in_dim(T354, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T370 = fd.ops.broadcast_in_dim(T359, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T374 = fd.ops.broadcast_in_dim(T360, shape=[b, s], broadcast_dims=[1])
    T375 = fd.ops.sub(T98, T365)
    S376 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T377 = fd.ops.mul(S376, T370)
    T382 = fd.ops.broadcast_in_dim(T374, shape=[b, s, 1], broadcast_dims=[0, 1])
    T387 = fd.ops.broadcast_in_dim(
        fd.layernorm0_bias, shape=[b, s, e], broadcast_dims=[2]
    )
    T388 = fd.ops.mul(T319, T331)
    T393 = fd.ops.broadcast_in_dim(
        fd.layernorm1_bias, shape=[b, s, e], broadcast_dims=[2]
    )
    T394 = fd.ops.mul(T108, T120)
    T395 = fd.ops.mul(T377, T375)
    T400 = fd.ops.broadcast_in_dim(T382, shape=[b, s, e], broadcast_dims=[0, 1, 2])
    T401 = fd.ops.cast(T387, dtype=DataType.Float)
    T402 = fd.ops.mul(T388, T312)
    T403 = fd.ops.permute(fd.sdpa_out, dims=[0, 2, 1, 3])
    T404 = fd.ops.cast(T393, dtype=DataType.Float)
    T405 = fd.ops.mul(T394, T100)
    S406 = fd.define_scalar(e, dtype=DataType.Double)
    S407 = fd.ops.reciprocal(S406)
    T408 = fd.ops.mul(T395, S407)
    S409 = fd.define_scalar(1 / e, dtype=DataType.Double)
    T410 = fd.ops.mul(S409, T400)
    T411 = fd.ops.add(T402, T401)
    T412 = fd.ops.stride_order(T403, stride_order=[3, 2, 1, 0])
    T413 = fd.ops.add(T405, T404)
    T414 = fd.ops.mul(T51, T57)
    T415 = fd.ops.add(T410, T408)
    T416 = fd.ops.add(T185, T336)
    T417 = fd.ops.mul(T388, T311)
    T418 = fd.ops.cast(T291, dtype=DataType.Float)
    T419 = fd.ops.cast(T411, dtype=DataType.BFloat16)
    T424 = fd.ops.reshape(T412, new_shape=[b, s, e])
    T425 = fd.ops.mul(T394, T99)
    T426 = fd.ops.cast(T413, dtype=DataType.BFloat16)
    T427 = fd.ops.cast(T414, dtype=DataType.BFloat16)
    T428 = fd.ops.add(T416, T415)
    T429 = fd.ops.sum(T417, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T430 = fd.ops.sum(T311, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T431 = fd.ops.sum(T418, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T435 = fd.ops.reshape(T419, new_shape=[b * s, e])
    T436 = fd.ops.permute(T294, dims=[1, 0])
    T437 = fd.ops.sum(T188, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T441 = fd.ops.reshape(T424, new_shape=[b * s, e])
    T442 = fd.ops.permute(T193, dims=[1, 0])
    T443 = fd.ops.sum(T425, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T444 = fd.ops.sum(T99, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T445 = fd.ops.sum(T71, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T449 = fd.ops.reshape(T426, new_shape=[b * s, e])
    T450 = fd.ops.permute(T76, dims=[1, 0])
    T451 = fd.ops.sum(T30, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T455 = fd.ops.reshape(T427, new_shape=[b * s, e * 4])
    T456 = fd.ops.permute(T38, dims=[1, 0])
    inp_grad = fd.ops.cast(T428, dtype=DataType.BFloat16)
    layernorm0_weight_grad = fd.ops.cast(T429, dtype=DataType.BFloat16)
    layernorm0_bias_grad = fd.ops.cast(T430, dtype=DataType.BFloat16)
    mha_linear0_bias_grad = fd.ops.cast(T431, dtype=DataType.BFloat16)
    mha_linear0_weight_grad = fd.ops.matmul(T436, T435)
    mha_linear1_bias_grad = fd.ops.cast(T437, dtype=DataType.BFloat16)
    mha_linear1_weight_grad = fd.ops.matmul(T442, T441)
    layernorm1_weight_grad = fd.ops.cast(T443, dtype=DataType.BFloat16)
    layernorm1_bias_grad = fd.ops.cast(T444, dtype=DataType.BFloat16)
    mlp_linear0_bias_grad = fd.ops.cast(T445, dtype=DataType.BFloat16)
    mlp_linear0_weight_grad = fd.ops.matmul(T450, T449)
    mlp_linear1_bias_grad = fd.ops.cast(T451, dtype=DataType.BFloat16)
    mlp_linear1_weight_grad = fd.ops.matmul(T456, T455)
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
    (
        mlp_linear0_out,
        out_grad,
        mlp_dropout_mask,
        mlp_linear1_weight,
        mha_dropout_mask,
        mha_linear1_out,
        mlp_linear0_weight,
        layernorm1_weight,
        layernorm1_mean,
        inp,
        layernorm1_rstd,
        mha_linear1_weight,
        mha_linear0_out,
        sdpa_out,
        sdpa_logsum_exp,
        sdpa_seed,
        sdpa_offset,
        mha_linear0_weight,
        layernorm0_weight,
        layernorm0_mean,
        layernorm0_rstd,
        layernorm0_bias,
        layernorm1_bias,
    ) = inputs
    for tv in inputs:
        tv.set_device_mesh(mesh)

    for tv in [
        mha_linear0_weight,
        mlp_linear0_weight,
    ]:
        tv.split(0, num_devices, inner_split=False)
        tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)

    for tv in [
        sdpa_out,
        sdpa_logsum_exp,
    ]:
        tv.split(1, num_devices, inner_split=False)
        tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    for tv in [
        mlp_linear0_out,
        mha_linear0_out,
        mha_linear1_weight,
        mlp_linear1_weight,
    ]:
        tv.split(-1, num_devices, inner_split=False)
        tv.axis(-2).parallelize(nvfuser.ParallelType.mesh_x)


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.mpi
def test_transformer_backward(multidevice_direct_test, benchmark):
    d = multidevice_direct_test.size
    mesh = nvfuser.multidevice.DeviceMesh(range(d))

    b, s, h, e = 1, 2048, 96, 12288

    torch.cuda.set_device(multidevice_direct_test.local_rank)

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
        multidevice_direct_test.shard_tensor(mlp_linear0_out, -1, mesh),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bool, device="cuda"),
        multidevice_direct_test.shard_tensor(mlp_linear1_weight, -1, mesh),
        torch.testing.make_tensor((b, s, e), dtype=torch.bool, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        multidevice_direct_test.shard_tensor(mlp_linear0_weight, 0, mesh),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        multidevice_direct_test.shard_tensor(mha_linear1_weight, -1, mesh),
        multidevice_direct_test.shard_tensor(mha_linear0_out, -1, mesh),
        multidevice_direct_test.shard_tensor(sdpa_out, 1, mesh)
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2),
        multidevice_direct_test.shard_tensor(sdpa_log_sumexp, 1, mesh),
        sdpa_philox_seed,
        sdpa_philox_offset,
        multidevice_direct_test.shard_tensor(mha_linear0_weight, 0, mesh),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((b, s), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((b, s, 1), dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
        torch.testing.make_tensor((e,), dtype=torch.bfloat16, device="cuda"),
    ]

    with FusionDefinition() as fd:
        transformer_backward_definition(fd, b, s, h, e)
        transformer_backward_multidevice_schedule(fd, d)

    # Resize scheduler disabled due toissue: #4890
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
