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
import itertools
from python.direct_utils import is_pre_ampere, is_pre_hopper, is_pre_blackwell


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

    # Check that keep_dim argument is not in fd_str
    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False)
    tv2 = fd.ops.add(tv0, tv1)
    c7 = fd.define_scalar(3.00000, dtype=DataType.Double)
    tv3 = fd.ops.mul(tv2, c7)
    tv4 = fd.ops.sum(tv3, dims=[2], dtype=DataType.Float)
    fd.add_output(tv4)"""

    # t0 and t1 are ones(2, 4, 8) tensors.
    # t2 = t0 + t1 = twos(2, 4, 8)
    # t3 = t2 * 3.0 = sixes(2,4,8)
    # t4 = sum(t3, dim=-1) = forty-eights(2, 4)
    # The expected output is a tensor of 48's.
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )
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


def test_implicit_broadcast_input(nvfuser_direct_test):
    inputs = [
        torch.randn(3, device="cuda"),
        torch.randn(2, 3, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
        t2 = fd.ops.add(t0_b, t1)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(
        prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1]
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_explicit_broadcast_input(nvfuser_direct_test):
    inputs = [
        torch.randn(1, 1, 4, device="cuda"),
        torch.randn(2, 3, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0_b = fd.ops.broadcast_in_dim(t0, inputs[1].size(), [0, 1, 2])
        t2 = fd.ops.add(t0_b, t1)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(
        prims.broadcast_in_dim(inputs[0], inputs[1].size(), [0, 1, 2]), inputs[1]
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_broadcast_mixing(nvfuser_direct_test):
    inputs = [
        torch.randn(3, 1, device="cuda"),
        torch.randn(3, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t1_b = fd.ops.broadcast_in_dim(t1, [3, 3], [0])
        t2 = fd.ops.add(t0, t1_b)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], [3, 3], [0]))
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


def test_linear_without_bias(nvfuser_direct_test):
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

    # Check that bias is not included with linear
    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv2 = fd.ops.linear(tv0, tv1)
    fd.add_output(tv2)"""

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )
    eager_out = torch.nn.functional.linear(inputs[0], inputs[1])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_linear_with_bias(nvfuser_direct_test):
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

    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    tv3 = fd.ops.linear(tv0, tv1, bias=tv2)
    fd.add_output(tv3)"""

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )
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

    # Check that squeeze does not print default argument: squeeze_expanded
    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    tv1 = fd.define_tensor(shape=[1, -1, 1], contiguity=[None, True, None], dtype=DataType.Float, is_cpu=False)
    tv2 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True], dtype=DataType.Float, is_cpu=False)
    tv3 = fd.ops.squeeze(tv1, dims=[0, 2])
    tv6 = fd.ops.mul(tv0, tv3)
    tv4 = fd.ops.squeeze(tv2, dims=[1])
    tv5 = fd.ops.sum(tv4, dims=[0], dtype=DataType.Float)
    tv7 = fd.ops.mul(tv6, tv5)
    fd.add_output(tv7)"""

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )

    v1 = torch.sum(inputs[1], [0, -1])
    v2 = torch.sum(inputs[2], [0, 1])
    eager_out = inputs[0] * v1 * v2
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


# Test that expanded dimensions can be reduced properly
# See https://github.com/NVIDIA/Fuser/issues/1678
def test_expanded_reduction(nvfuser_direct_test):
    inputs = [torch.tensor(1.0, device="cuda").as_strided((2, 3), (0, 0))]

    for keepdim in [False, True]:

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[None, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.sum(T0, dims=[0], keep_dim=keepdim, dtype=DataType.Null)
            fd.add_output(T1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        nvfuser_direct_test.assertEqual(
            nvf_out[0], inputs[0].sum(dim=0, keepdim=keepdim)
        )


def test_expand(nvfuser_direct_test):
    inputs = [
        torch.randn(1, 1, 4, device="cuda"),
        torch.randn(2, 3, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0_b = fd.ops.expand(t0, inputs[1].size())
        t2 = fd.ops.add(t0_b, t1)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = inputs[0].expand(inputs[1].size()) + inputs[1]
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_index_select(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
        torch.randn(8, 16, device="cuda"),
        torch.randint(0, 8, (6,), device="cuda").to(dtype=torch.long),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.index_select(t3, t2, dim)
            fd.add_output(t4)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.index_select(inputs[0] + inputs[1], dim, inputs[2])
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_index_select_scalar_indices(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
        torch.tensor(2, device="cuda").to(dtype=torch.long),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.index_select(t0, t1, dim)
            fd.add_output(t2)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.index_select(inputs[0], dim, inputs[1])
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_select(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
        index := 2,
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            s1 = fd.define_scalar(dtype=DataType.Int)
            t1 = fd.ops.select(t0, s1, dim)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.select(inputs[0], dim, inputs[1])
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_cumsum(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.cumsum(t0, dim)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.cumsum(inputs[0], dim)
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_cumprod(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.cumprod(t0, dim)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.cumprod(inputs[0], dim)
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_cummin(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.cummin(t0, dim)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.cummin(inputs[0], dim)
        nvfuser_direct_test.assertEqual(eager_out[0], nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_cummax(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.cummax(t0, dim)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.cummax(inputs[0], dim)
        nvfuser_direct_test.assertEqual(eager_out[0], nvf_out[0])

    test_fn(0)
    test_fn(1)


def test_where(nvfuser_direct_test):
    # nvfuser_where is a decorator function. It takes the input arguments
    # and creates a function that builds a FusionDefinition.
    def nvfuser_where(pred, a, b):
        def fusion_func(fd: FusionDefinition):
            nv_pred = fd.define_tensor(
                sizes=pred.shape, strides=pred.stride(), dtype=DataType.Bool
            )
            nv_a = fd.define_tensor(
                sizes=a.shape,
                strides=a.stride(),
                dtype=torch_dtype_to_nvfuser_dtype(a.dtype),
            )
            nv_b = fd.define_tensor(
                sizes=b.shape,
                strides=b.stride(),
                dtype=torch_dtype_to_nvfuser_dtype(b.dtype),
            )
            result = fd.ops.where(nv_pred, nv_a, nv_b)
            fd.add_output(result)

        return fusion_func

    # get list of dtypes to test with
    list_of_dtype = [torch.float16, torch.float32]
    if not is_pre_ampere():
        list_of_dtype.append(torch.bfloat16)

    pred = torch.testing.make_tensor((5,), device="cuda", dtype=torch.bool)
    for atype, btype in itertools.product(list_of_dtype, list_of_dtype):
        a = torch.randn((5,), device="cuda", dtype=atype)
        b = torch.randn((5,), device="cuda", dtype=btype)
        fusion_func = nvfuser_where(pred, a, b)
        nv_result, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, [pred, a, b])
        torch_result = torch.where(pred, a, b)
        nvfuser_direct_test.assertEqual(nv_result[0], torch_result)


def test_where_dtypes(nvfuser_direct_test):
    inputs = [
        torch.arange(2, device="cuda").type(torch.bool),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])

        c0 = fd.define_scalar(3.0)
        c1 = fd.define_scalar(5.0)
        t1 = fd.ops.where(t0, c0, c1)  # DataType.Double
        fd.add_output(t1)

        c0f = fd.define_scalar(3.0, DataType.Float)
        c1f = fd.define_scalar(5.0, DataType.Float)
        t1f = fd.ops.where(t0, c0f, c1f)  # DataType.Float
        fd.add_output(t1f)

        c0d = fd.define_scalar(3.0, DataType.Double)
        c1d = fd.define_scalar(5.0, DataType.Double)
        t1d = fd.ops.where(t0, c0d, c1d)  # DataType.Double
        fd.add_output(t1d)

        c0i = fd.define_scalar(3, DataType.Int32)
        c1i = fd.define_scalar(5, DataType.Int32)
        t1i = fd.ops.where(t0, c0i, c1i)  # DataType.Int32
        fd.add_output(t1i)

        c0l = fd.define_scalar(3)
        c1l = fd.define_scalar(5)
        t1l = fd.ops.where(t0, c0l, c1l)  # DataType.Int
        fd.add_output(t1l)

        c0c = fd.define_scalar(complex(3.0))
        c1c = fd.define_scalar(complex(5.0))
        t1c = fd.ops.where(t0, c0c, c1c)  # DataType.ComplexDouble
        fd.add_output(t1c)

        c0cf = fd.define_scalar(3.0 + 0j, DataType.ComplexFloat)
        c1cf = fd.define_scalar(5.0 + 0j, DataType.ComplexFloat)
        t1cf = fd.ops.where(t0, c0cf, c1cf)  # DataType.ComplexFloat
        fd.add_output(t1cf)

        c0cd = fd.define_scalar(3.0 + 0j, DataType.ComplexDouble)
        c1cd = fd.define_scalar(5.0 + 0j, DataType.ComplexDouble)
        t1cd = fd.ops.where(t0, c0cd, c1cd)  # DataType.ComplexDouble
        fd.add_output(t1cd)

        c0b = fd.define_scalar(True, DataType.Bool)
        c1b = fd.define_scalar(False, DataType.Bool)
        t1b = fd.ops.where(t0, c0b, c1b)  # DataType.Bool
        fd.add_output(t1b)

    (
        n,
        nf,
        nd,
        ni,
        nl,
        nc,
        ncf,
        ncd,
        nb,
    ), _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    eager_out = torch.where(inputs[0], 3.0, 5.0)

    # explicit Float dtype matches torch.where behavior
    nvfuser_direct_test.assertEqual(eager_out, nf)

    assert n.dtype == torch.float64
    assert nf.dtype == torch.float32
    assert nd.dtype == torch.float64
    assert ni.dtype == torch.int32
    assert nl.dtype == torch.int64
    assert nc.dtype == torch.complex128
    assert ncf.dtype == torch.complex64
    assert ncd.dtype == torch.complex128
    assert nb.dtype == torch.bool


def test_addcmul(nvfuser_direct_test):
    inputs = [
        torch.randn(4, device="cuda", dtype=torch.float32),
        torch.randn(4, device="cuda", dtype=torch.float32),
        torch.randn(4, device="cuda", dtype=torch.float32),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.from_pytorch(inputs[2])
        c0 = fd.define_scalar(0.1)

        t3 = fd.ops.addcmul(t0, t1, t2, c0)

        fd.add_output(t3)

    nvfout, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    torch_out = torch.addcmul(*inputs, value=0.1)

    nvfuser_direct_test.assertEqual(nvfout[0], torch_out)


def test_slice(nvfuser_direct_test):
    x = torch.randn((2, 5, 10), dtype=torch.float32, device="cuda:0")

    offset = (0, 1, 2)

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.ops.slice(
            T0, start_indices=offset, end_indices=(2, 5, 10), strides=(1, 1, 1)
        )
        fd.add_output(T1)
        V_start = list(offset)
        V_end = T0.shape()
        T2 = fd.ops.slice(T0, V_start, V_end)
        fd.add_output(T2)
        dynamic_start = fd.define_vector(3)
        dynamic_end = fd.define_vector(3)
        T3 = fd.ops.slice(T0, dynamic_start, dynamic_end)
        fd.add_output(T3)

    inputs = [x, *offset, *x.shape]

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    for out in nvf_out:
        nvfuser_direct_test.assertTrue(out.allclose(x[:, 1:, 2:]))


def test_iota(nvfuser_direct_test):
    inputs = [
        (2, 0, 2, DataType.Int),
        (3, 100, 1, DataType.Int32),
    ]

    def fusion_func(fd: FusionDefinition):
        for input in inputs:
            c0 = fd.define_scalar(input[0])
            c1 = None if input[1] is None else fd.define_scalar(input[1])
            c2 = None if input[2] is None else fd.define_scalar(input[2])
            dt = input[3]
            t3 = fd.ops.iota(c0, c1, c2, dt)
            fd.add_output(t3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, [])

    eager_out1 = torch.tensor([0, 2], dtype=torch.long, device="cuda")
    eager_out2 = torch.tensor([100, 101, 102], dtype=torch.int, device="cuda")
    nvfuser_direct_test.assertEqual(eager_out1, nvf_out[0])
    nvfuser_direct_test.assertEqual(eager_out2, nvf_out[1])


def test_scalar_only_inputs(nvfuser_direct_test):
    # We don't allow scalar outputs, currently,
    # so a tensor has to be returned
    def fusion_func(fd: FusionDefinition):
        s0 = fd.define_scalar()
        s1 = fd.define_scalar()
        s2 = fd.ops.add(s0, s1)
        c0 = fd.define_scalar(1.0, DataType.Float)
        t3 = fd.ops.full(shape=[2, 2], fill_value=c0, dtype=DataType.Float)
        t4 = fd.ops.mul(t3, s2)
        fd.add_output(t4)

    with FusionDefinition() as fd:
        fusion_func(fd)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, [2.0, 3.0])
    eager_out = torch.full([2, 2], 1.0) * 5.0
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])
