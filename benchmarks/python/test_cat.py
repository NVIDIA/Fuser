# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from nvfuser import FusionDefinition, DataType
import core


def phi3_cat_fusion_1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[1, 48, 2048], contiguity=[None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.ops.permute(T0, dims=[0, 2, 1])
    T2 = fd.ops.cat([T1, T1], dim=-1, manual_padding=0)
    T3 = fd.ops.cos(T2)
    T4 = fd.ops.sin(T2)
    S5 = fd.define_scalar(1.19024, dtype=DataType.Double)
    T6 = fd.ops.mul(T3, S5)
    S7 = fd.define_scalar(1.19024, dtype=DataType.Double)
    T8 = fd.ops.mul(T4, S7)
    T9 = fd.ops.cast(T6, dtype=DataType.BFloat16)
    T10 = fd.ops.cast(T8, dtype=DataType.BFloat16)
    fd.add_output(T9)
    fd.add_output(T10)


def phi3_cat_fusion_2(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[2048, 9216], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    T3 = fd.define_tensor(shape=[1, 2048, 9216], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T4 = fd.define_tensor(shape=[1, 2048, 96], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
    T5 = fd.define_tensor(shape=[1, 2048, 96], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
    T10 = fd.ops.reshape(T0, new_shape=[1, 2048, 9216])
    T11 = fd.ops.cast(T10, dtype=DataType.Float)
    S12 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S13 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S14 = fd.define_scalar(1, dtype=DataType.Int)
    S15 = fd.define_scalar(2048, dtype=DataType.Int)
    S16 = fd.define_scalar(9216, dtype=DataType.Int)
    T18 = fd.ops.uniform(S12, S13, shape=[S14, S15, S16], rng_seed=S2, rng_offset=S1, dtype=DataType.BFloat16)
    S19 = fd.define_scalar(4.00000, dtype=DataType.Double)
    T20 = fd.ops.mul(T11, S19)
    S21 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T22 = fd.ops.lt(T18, S21)
    T23 = fd.ops.cast(T22, dtype=DataType.Float)
    T24 = fd.ops.mul(T20, T23)
    S25 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T26 = fd.ops.mul(T24, S25)
    T27 = fd.ops.cast(T3, dtype=DataType.Float)
    T28 = fd.ops.add(T27, T26)
    T29 = fd.ops.cast(T28, dtype=DataType.BFloat16)
    T42 = fd.ops.slice(T29, start_indices=[0, 0, 3072], end_indices=[1, 2048, 6144], strides=[1, 1, 1], manual_normalization=0)
    T55 = fd.ops.slice(T29, start_indices=[0, 0, 0], end_indices=[1, 2048, 3072], strides=[1, 1, 1], manual_normalization=0)
    T61 = fd.ops.reshape(T42, new_shape=[1, 2048, 32, 96])
    T67 = fd.ops.reshape(T55, new_shape=[1, 2048, 32, 96])
    T68 = fd.ops.permute(T61, dims=[0, 2, 1, 3])
    T69 = fd.ops.permute(T67, dims=[0, 2, 1, 3])
    T85 = fd.ops.slice(T68, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 2048, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T101 = fd.ops.slice(T69, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 2048, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T102 = fd.ops.cast(T85, dtype=DataType.Float)
    T103 = fd.ops.cast(T101, dtype=DataType.Float)
    T104 = fd.ops.neg(T102)
    T105 = fd.ops.neg(T103)
    T111 = fd.ops.broadcast_in_dim(T4, shape=[1, 1, 2048, 96], broadcast_dims=[0, 2, 3])
    T127 = fd.ops.slice(T68, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 2048, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T128 = fd.ops.cast(T104, dtype=DataType.BFloat16)
    T134 = fd.ops.broadcast_in_dim(T5, shape=[1, 1, 2048, 96], broadcast_dims=[0, 2, 3])
    T150 = fd.ops.slice(T69, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 2048, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T151 = fd.ops.cast(T105, dtype=DataType.BFloat16)
    T157 = fd.ops.broadcast_in_dim(T111, shape=[1, 32, 2048, 96], broadcast_dims=[0, 1, 2, 3])
    T158 = fd.ops.cat([T128, T127], dim=-1, manual_padding=0)
    T164 = fd.ops.broadcast_in_dim(T134, shape=[1, 32, 2048, 96], broadcast_dims=[0, 1, 2, 3])
    T165 = fd.ops.cat([T151, T150], dim=-1, manual_padding=0)
    T166 = fd.ops.cast(T157, dtype=DataType.Float)
    T167 = fd.ops.cast(T158, dtype=DataType.Float)
    T168 = fd.ops.cast(T164, dtype=DataType.Float)
    T169 = fd.ops.cast(T68, dtype=DataType.Float)
    T170 = fd.ops.cast(T165, dtype=DataType.Float)
    T171 = fd.ops.cast(T69, dtype=DataType.Float)
    T172 = fd.ops.mul(T167, T166)
    T173 = fd.ops.mul(T169, T168)
    T174 = fd.ops.mul(T170, T166)
    T175 = fd.ops.mul(T171, T168)
    T188 = fd.ops.slice(T29, start_indices=[0, 0, 6144], end_indices=[1, 2048, 9216], strides=[1, 1, 1], manual_normalization=0)
    T194 = fd.ops.reshape(T188, new_shape=[1, 2048, 32, 96])
    T195 = fd.ops.add(T173, T172)
    T196 = fd.ops.add(T175, T174)
    T197 = fd.ops.permute(T194, dims=[0, 2, 1, 3])
    T198 = fd.ops.cast(T195, dtype=DataType.BFloat16)
    T199 = fd.ops.cast(T196, dtype=DataType.BFloat16)
    T200 = fd.ops.stride_order(T197, stride_order=[3, 2, 1, 0])
    T201 = fd.ops.stride_order(T198, stride_order=[3, 2, 1, 0])
    T202 = fd.ops.stride_order(T199, stride_order=[3, 2, 1, 0])
    fd.add_output(T197)
    fd.add_output(T198)
    fd.add_output(T200)
    fd.add_output(T201)
    fd.add_output(T202)


def qwen2_cat_fusion_1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[1, 64, 2048], contiguity=[None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[1, 2048, 3584], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T2 = fd.define_tensor(shape=[3584], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T3 = fd.ops.permute(T0, dims=[0, 2, 1])
    T4 = fd.ops.cat([T3, T3], dim=-1, manual_padding=0)
    T5 = fd.ops.cos(T4)
    T6 = fd.ops.sin(T4)
    S7 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T8 = fd.ops.mul(T5, S7)
    S9 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T10 = fd.ops.mul(T6, S9)
    T11 = fd.ops.cast(T8, dtype=DataType.BFloat16)
    T12 = fd.ops.cast(T10, dtype=DataType.BFloat16)
    T13 = fd.ops.cast(T1, dtype=DataType.Float)
    S14 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T15 = fd.ops.pow(T13, S14)
    T16 = fd.ops.sum(T15, dims=[2], keepdim=False, dtype=DataType.Null)
    T21 = fd.ops.broadcast_in_dim(T16, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    S22 = fd.define_scalar(3584.00, dtype=DataType.Double)
    S23 = fd.ops.reciprocal(S22)
    T24 = fd.ops.mul(T21, S23)
    S25 = fd.define_scalar(1.00000e-06, dtype=DataType.Double)
    T26 = fd.ops.add(T24, S25)
    T27 = fd.ops.rsqrt(T26)
    T32 = fd.ops.broadcast_in_dim(T27, shape=[1, 2048, 3584], broadcast_dims=[0, 1, 2])
    T33 = fd.ops.mul(T13, T32)
    T38 = fd.ops.broadcast_in_dim(T2, shape=[1, 2048, 3584], broadcast_dims=[2])
    T39 = fd.ops.cast(T38, dtype=DataType.Float)
    T40 = fd.ops.mul(T39, T33)
    T41 = fd.ops.cast(T40, dtype=DataType.BFloat16)
    fd.add_output(T11)
    fd.add_output(T12)
    fd.add_output(T41)


def qwen2_cat_fusion_2(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[2048, 512], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    T3 = fd.define_tensor(shape=[1, 4, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
    T4 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
    T5 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
    T6 = fd.define_tensor(shape=[1, 28, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
    T7 = fd.define_tensor(shape=[1, 2048, 512], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T12 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
    T13 = fd.ops.cast(T12, dtype=DataType.Float)
    S14 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S16 = fd.define_scalar(1, dtype=DataType.Int)
    S17 = fd.define_scalar(2048, dtype=DataType.Int)
    S18 = fd.define_scalar(512, dtype=DataType.Int)
    T20 = fd.ops.uniform(S14, S15, shape=[S16, S17, S18], rng_seed=S2, rng_offset=S1, dtype=DataType.BFloat16)
    S21 = fd.define_scalar(4.00000, dtype=DataType.Double)
    T22 = fd.ops.mul(T13, S21)
    S23 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T24 = fd.ops.lt(T20, S23)
    T25 = fd.ops.cast(T24, dtype=DataType.Float)
    T41 = fd.ops.slice(T3, start_indices=[0, 0, 0, 64], end_indices=[1, 4, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T42 = fd.ops.mul(T22, T25)
    T43 = fd.ops.cast(T41, dtype=DataType.Float)
    T44 = fd.ops.neg(T43)
    T50 = fd.ops.broadcast_in_dim(T4, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
    T66 = fd.ops.slice(T3, start_indices=[0, 0, 0, 0], end_indices=[1, 4, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T67 = fd.ops.cast(T44, dtype=DataType.BFloat16)
    T73 = fd.ops.broadcast_in_dim(T5, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
    T89 = fd.ops.slice(T6, start_indices=[0, 0, 0, 64], end_indices=[1, 28, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S90 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T91 = fd.ops.mul(T42, S90)
    T97 = fd.ops.broadcast_in_dim(T50, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
    T98 = fd.ops.cat([T67, T66], dim=-1, manual_padding=0)
    T104 = fd.ops.broadcast_in_dim(T73, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
    T105 = fd.ops.cast(T89, dtype=DataType.Float)
    T106 = fd.ops.cast(T97, dtype=DataType.Float)
    T107 = fd.ops.cast(T98, dtype=DataType.Float)
    T108 = fd.ops.cast(T104, dtype=DataType.Float)
    T109 = fd.ops.cast(T3, dtype=DataType.Float)
    T110 = fd.ops.neg(T105)
    T111 = fd.ops.cast(T7, dtype=DataType.Float)
    T112 = fd.ops.mul(T107, T106)
    T113 = fd.ops.mul(T109, T108)
    T129 = fd.ops.slice(T6, start_indices=[0, 0, 0, 0], end_indices=[1, 28, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T130 = fd.ops.cast(T110, dtype=DataType.BFloat16)
    T131 = fd.ops.add(T111, T91)
    T137 = fd.ops.broadcast_in_dim(T50, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
    T138 = fd.ops.cat([T130, T129], dim=-1, manual_padding=0)
    T144 = fd.ops.broadcast_in_dim(T73, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
    T145 = fd.ops.cast(T131, dtype=DataType.BFloat16)
    T146 = fd.ops.cast(T137, dtype=DataType.Float)
    T147 = fd.ops.cast(T138, dtype=DataType.Float)
    T148 = fd.ops.cast(T144, dtype=DataType.Float)
    T149 = fd.ops.cast(T6, dtype=DataType.Float)
    T155 = fd.ops.reshape(T145, new_shape=[1, 2048, 4, 128])
    T156 = fd.ops.add(T113, T112)
    T157 = fd.ops.mul(T147, T146)
    T158 = fd.ops.mul(T149, T148)
    T159 = fd.ops.permute(T155, dims=[0, 2, 1, 3])
    T160 = fd.ops.cast(T156, dtype=DataType.BFloat16)
    T167 = fd.ops.broadcast_in_dim(T159, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
    T174 = fd.ops.broadcast_in_dim(T160, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
    T181 = fd.ops.broadcast_in_dim(T167, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
    T188 = fd.ops.broadcast_in_dim(T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
    T189 = fd.ops.add(T158, T157)
    T195 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
    T201 = fd.ops.reshape(T188, new_shape=[1, 28, 2048, 128])
    T202 = fd.ops.cast(T189, dtype=DataType.BFloat16)
    T203 = fd.ops.stride_order(T195, stride_order=[3, 2, 1, 0])
    T204 = fd.ops.stride_order(T201, stride_order=[3, 2, 1, 0])
    T205 = fd.ops.stride_order(T202, stride_order=[3, 2, 1, 0])
    fd.add_output(T159)
    fd.add_output(T160)
    fd.add_output(T203)
    fd.add_output(T204)
    fd.add_output(T205)


def test_cat_phi3_v1(benchmark):
    inputs = [
      torch.randn((1, 48, 2048), dtype=torch.float32, device='cuda:0')
    ]

    with FusionDefinition() as fd:
        phi3_cat_fusion_1(fd)

    def benchmark_fn(inputs):
        fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)


def test_cat_phi3_v2(benchmark):
    inputs = [
        torch.testing.make_tensor((2048, 9216), dtype=torch.bfloat16, device='cuda:0'),
        13284,
        7808972590298790,
        torch.testing.make_tensor((1, 2048, 9216), dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(196608, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 96), (196608, 1, 2048)),
        torch.randn(196608, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 96), (196608, 1, 2048)),
    ]
    with FusionDefinition() as fd:
        phi3_cat_fusion_2(fd)

    def benchmark_fn(inputs):
        fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)


# Note that this is also basically the same fusion the Mistral-Nemo network.
def test_cat_qwen2_v1(benchmark):
    inputs = [
        torch.testing.make_tensor((1, 64, 2048), dtype=torch.float32, device='cuda:0'),
        torch.testing.make_tensor((1, 2048, 3584), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((3584,), dtype=torch.bfloat16, device='cuda:0'),
    ]
    with FusionDefinition() as fd:
        qwen2_cat_fusion_1(fd)

    def benchmark_fn(inputs):
        fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)


@pytest.mark.skip("issue 3740")
def test_cat_qwen2_v2(benchmark):
    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device='cuda:0'),
        25546,
        1400552702872758,
        torch.randn(1048576, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 4, 2048, 128), (1048576, 128, 512, 1)),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(7340032, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 28, 2048, 128), (7340032, 128, 3584, 1)),
        torch.testing.make_tensor((1, 2048, 512), dtype=torch.bfloat16, device='cuda:0'),
    ]

    with FusionDefinition() as fd:
        qwen2_cat_fusion_2(fd)

    def benchmark_fn(inputs):
        fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)
