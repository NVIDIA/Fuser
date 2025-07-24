# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import torch._refs as refs
import torch._prims as prims

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

import pytest
from python.direct_utils import is_pre_hopper, is_pre_blackwell


def test_basic(nvfuser_direct_test):
    inputs = [
        torch.ones(2, 4, 8, device="cuda"),
        torch.ones(2, 4, 8, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

        fd.add_output(t4)

    # t0 and t1 are ones(2, 4, 8) tensors.
    # t2 = t0 + t1 = twos(2, 4, 8)
    # t3 = t2 * 3.0 = sixes(2,4,8)
    # t4 = sum(t3, dim=-1) = forty-eights(2, 4)
    # The expected output is a tensor of 48's.
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_basic_fp16(nvfuser_direct_test):
    inputs = [
        torch.ones(2, 4, 8, device="cuda", dtype=torch.float16),
        torch.ones(2, 4, 8, device="cuda", dtype=torch.float16),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

        t5 = fd.ops.cast(t4, DataType.Half)
        fd.add_output(t5)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_cast_scalar(nvfuser_direct_test):
    inputs = [
        torch.ones(2, 4, 8, device="cuda", dtype=torch.int32),
        torch.ones(2, 4, 8, device="cuda", dtype=torch.int32),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        c1 = fd.ops.cast(c0, DataType.Int32)
        t3 = fd.ops.mul(t2, c1)
        t4 = fd.ops.sum(t3, [-1], False, DataType.Int32)

        fd.add_output(t4)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.sum((inputs[0] + inputs[1]) * 3, dim=-1, dtype=torch.int32)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_cast_double_to_half(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 4, device="cuda", dtype=torch.float64),
        torch.randn(2, 4, device="cuda", dtype=torch.float64),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0h = fd.ops.cast(t0, DataType.Half)
        t1h = fd.ops.cast(t1, DataType.Half)
        t2 = fd.ops.add(t0h, t1h)
        t3 = fd.ops.relu(t2)
        t4 = fd.ops.cast(t3, DataType.Half)

        fd.add_output(t4)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.relu(inputs[0].to(torch.half) + inputs[1].to(torch.half))
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_cast_fp8(nvfuser_direct_test):
    def fn(in_type, out_type):
        inputs = [
            torch.randn([5, 5], device="cuda").to(in_type),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.ops.cast(T0, dtype=torch_dtype_to_nvfuser_dtype(out_type))
            fd.add_output(T1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
        eager_out = inputs[0].to(out_type)
        if in_type == torch.float8_e8m0fnu or out_type == torch.float8_e8m0fnu:
            # Eager mode uses manual bit manipulation, and nvFuser uses
            # hardware instructions. Unfortunately, these implementations
            # do not match exactly. e8m0 can only represent 2^x, so we are
            # asserting that the x of the two results are off by at most 1.
            nvf_out_fp32 = nvf_out[0].to(torch.float32)
            eager_out_fp32 = eager_out.to(torch.float32)
            rel_err = eager_out_fp32.div(nvf_out_fp32).max().item()
            nvfuser_direct_test.assertTrue(rel_err <= 2 and rel_err >= 0.5)
        else:
            nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    for type0 in [torch.double, torch.float32, torch.float16, torch.bfloat16]:
        type1_list = [torch.float8_e4m3fn, torch.float8_e5m2]
        if not is_pre_blackwell():
            type1_list.append(torch.float8_e8m0fnu)
        for type1 in type1_list:
            fn(type0, type1)
            fn(type1, type0)


def test_promote_to_double(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 4, device="cuda", dtype=torch.float16),
        torch.randn(2, 4, device="cuda", dtype=torch.float64),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t2 = fd.ops.add(t0, t1)
        t5 = fd.ops.relu(t2)

        fd.add_output(t5)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.relu(inputs[0] + inputs[1])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_broadcast(nvfuser_direct_test):
    inputs = [
        torch.randn(3, device="cuda"),
        torch.randn(2, 3, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0_b = fd.ops.broadcast(t0, [True, False, True])
        t2 = fd.ops.add(t0_b, t1)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(
        prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1]
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_matmul(nvfuser_direct_test):
    m = 24
    n = 16
    k = 8
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.matmul(t0, t1)
        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.matmul(inputs[0], inputs[1])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_linear_with_bias(nvfuser_direct_test):
    m = 24
    n = 16
    k = 8
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.linear(t0, t1)
        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.nn.functional.linear(inputs[0], inputs[1])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_linear_without_bias(nvfuser_direct_test):
    m = 24
    n = 16
    k = 8
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, device="cuda", dtype=torch.bfloat16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.from_pytorch(inputs[2])
        t3 = fd.ops.linear(t0, t1, t2)
        fd.add_output(t3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.nn.functional.linear(inputs[0], inputs[1], inputs[2])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_tensor_ndim(nvfuser_direct_test):
    shape = [2 for i in range(12)]
    new_shape = shape[:9]
    new_shape.append(8)

    inputs = [torch.randn(shape, device="cuda"), new_shape]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        n_shape = fd.define_vector(10)

        t1 = fd.ops.reshape(t0, n_shape)
        t2 = fd.ops.sum(t1, dims=[3])

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.sum(inputs[0].reshape(new_shape), dim=3)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_execute_with_tuple_and_list(nvfuser_direct_test):
    shape = [2, 3, 4]
    new_shape = [6, 4]

    tensor = torch.randn(shape, device="cuda")
    inputs_with_list = [tensor, new_shape]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs_with_list[0])
        n_shape = fd.define_vector(2)

        t1 = fd.ops.reshape(t0, n_shape)
        t2 = fd.ops.sum(t1, dims=[0])

        fd.add_output(t2)

    eager_out = torch.sum(inputs_with_list[0].reshape(new_shape), dim=0)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs_with_list)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    inputs_with_tuple = [tensor, tuple(new_shape)]
    # expect to reuse fusion
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs_with_tuple)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_dynamic_reshape(nvfuser_direct_test):
    def dynamic_reshape(fd: FusionDefinition) -> None:
        x = fd.define_tensor([-1, -1], [True, True])
        d0 = fd.ops.size(x, 0)
        d1 = fd.define_scalar(dtype=DataType.Int32)
        d2 = fd.define_scalar(dtype=DataType.Int32)
        y = fd.ops.reshape(x, [d0, d1, d2])
        fd.add_output(y)

    x = torch.rand(3, 4, device="cuda")
    ys, _ = nvfuser_direct_test.exec_nvfuser(dynamic_reshape, [x, 2, 2])
    nvfuser_direct_test.assertEqual(len(ys), 1)
    y = ys[0]

    nvfuser_direct_test.assertEqual(y.shape, torch.Size([3, 2, 2]))
    nvfuser_direct_test.assertEqual(x.flatten(), y.flatten())


def test_reshape_dynamic(nvfuser_direct_test):
    inputs = [
        32,
        torch.randn((192,), dtype=torch.float32, device="cuda:0").as_strided(
            (4, 8, 6), (48, 6, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(None, dtype=DataType.Int)
        T1 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        S2 = fd.define_scalar(1, dtype=DataType.Int)
        S3 = fd.ops.mul(S2, S0)
        S4 = fd.ops.signbit(S3)
        S5 = fd.define_scalar(False, dtype=DataType.Bool)
        S6 = fd.ops.ne(S4, S5)
        S7 = fd.define_scalar(192, dtype=DataType.Int)
        S8 = fd.ops.fmod(S7, S3)
        S9 = fd.ops.cast(S8, dtype=DataType.Int)
        S10 = fd.define_scalar(0, dtype=DataType.Int)
        S11 = fd.ops.ne(S9, S10)
        S12 = fd.ops.bitwise_and(S6, S11)
        S13 = fd.define_scalar(192, dtype=DataType.Int)
        S14 = fd.ops.reciprocal(S3)
        S15 = fd.ops.mul(S13, S14)
        S16 = fd.ops.cast(S12, dtype=DataType.Int)
        S17 = fd.ops.sub(S15, S16)
        T19 = fd.ops.reshape(T1, new_shape=[S0, S17])
        T20 = fd.ops.sum(T19, dims=[1], keep_dim=False, dtype=DataType.Null)
        fd.add_output(T20)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


# Test empty symbolic tensors can be reshaped
# See https://github.com/NVIDIA/Fuser/issues/2362
def test_empty_reshape(nvfuser_direct_test):
    inputs = [torch.randint(0, 10, (0, 1, 2, 3, 4), dtype=torch.int64, device="cuda:0")]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, 1, -1, -1, -1],
            contiguity=[False, None, True, True, True],
            dtype=DataType.Int,
            is_cpu=False,
            stride_order=[4, 3, 2, 1, 0],
        )
        S2 = fd.define_scalar(5, dtype=DataType.Int)
        S3 = fd.define_scalar(0, dtype=DataType.Int)
        T5 = fd.ops.reshape(T0, new_shape=[S2, S3])
        fd.add_output(T5)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_squeeze(nvfuser_direct_test):
    t0_sizes = [4]
    t1_sizes = [1, 4, 1]
    t2_sizes = [2, 1, 4]
    inputs = [
        torch.randn(*t0_sizes, device="cuda"),
        torch.randn(*t1_sizes, device="cuda"),
        torch.randn(*t2_sizes, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1], contiguity=[True])
        t1 = fd.define_tensor(sizes=t1_sizes, strides=[4, 1, 1])
        t2 = fd.define_tensor(sizes=t2_sizes, strides=[4, 4, 1])
        t3 = fd.ops.squeeze(t1, [0, -1])
        t4 = fd.ops.squeeze(t2, [-2])
        t5 = fd.ops.sum(t4, [0])
        t6 = fd.ops.mul(t0, t3)
        t7 = fd.ops.mul(t6, t5)
        fd.add_output(t7)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    v1 = torch.sum(inputs[1], [0, -1])
    v2 = torch.sum(inputs[2], [0, 1])
    eager_out = inputs[0] * v1 * v2
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])
