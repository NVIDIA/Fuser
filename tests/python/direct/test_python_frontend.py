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
    compute_tensor_descriptor,
)
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

import pytest
import itertools
from python.direct_utils import (
    is_pre_ampere,
    is_pre_hopper,
    is_pre_blackwell,
    verify_stride_order,
)


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

    # Check that keepdim argument is not in fd_str
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
        T20 = fd.ops.sum(T19, dims=[1], keepdim=False, dtype=DataType.Null)
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
            T1 = fd.ops.sum(T0, dims=[0], keepdim=keepdim, dtype=DataType.Null)
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


def test_take_along_axis(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
        torch.randn(8, 16, device="cuda"),
        torch.randint(0, 8, (8, 16), device="cuda").to(dtype=torch.long),
    ]

    def test_fn(dim):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.take_along_axis(t3, t2, dim)
            fd.add_output(t4)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
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


def test_alias_output_to_input(nvfuser_direct_test):
    in_tensors = [
        torch.ones(4, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(in_tensors[0])  # = 1.0
        one = fd.define_scalar(1.0)
        two = fd.define_scalar(2.0)
        t1 = fd.ops.add(t0, one)  # = t0 + 1.0 = 2.0
        t2 = fd.ops.add(t1, two)  # = t1 + 2.0 = 4.0
        fd.add_output(t1, alias_input=t0)
        fd.add_output(t2)

    out_tensors, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, in_tensors)

    # t1 is an alias and therefore is hidden.
    nvfuser_direct_test.assertEqual(len(out_tensors), 1)
    nvfuser_direct_test.assertEqual(
        out_tensors[0], torch.full((4, 4), 4.0, device="cuda")
    )
    nvfuser_direct_test.assertEqual(
        in_tensors[0], torch.full((4, 4), 2.0, device="cuda")
    )


def test_returning_aliased_outputs(nvfuser_direct_test):
    inputs = [torch.randn((1, 2, 3, 4), dtype=torch.float32, device="cuda:0")]

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        S1 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T2 = fd.ops.gt(T0, S1)
        S3 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T4 = fd.ops.where(T2, T0, S3)
        fd.add_output(T4)
        fd.add_output(T4, T0)
        fd.add_output(T4)
        fd.add_output(T0)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    num_out = len(nvf_out)
    nvfuser_direct_test.assertEqual(num_out, 3)
    for i in range(num_out):
        nvfuser_direct_test.assertEqual(nvf_out[i].data_ptr(), inputs[0].data_ptr())


def test_welford(nvfuser_direct_test):
    inputs = [torch.randn(2, 2, device="cuda")]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        mean, var_sum, n = fd.ops.welford(t0, [-1])
        var = fd.ops.div(var_sum, n)
        fd.add_output(var)
        fd.add_output(mean)

    fuser_result, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_result = torch.var_mean(inputs[0], [-1], correction=0)
    nvfuser_direct_test.assertEqual(fuser_result, torch_result)


def test_gather(nvfuser_direct_test):
    inputs = [
        torch.randn(8, 16, device="cuda"),
        torch.randn(8, 16, device="cuda"),
        torch.randint(0, 8, (4, 4), device="cuda").to(dtype=torch.long),
    ]

    for dim in [0, 1]:

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.gather(t3, t2, dim)
            fd.add_output(t4)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_pad(nvfuser_direct_test):
    inputs = [
        torch.testing.make_tensor((1, 2, 3), dtype=torch.float32, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])

        t1 = fd.ops.pad(t0, [1, 1, 1, 1])
        fd.add_output(t1)

        # zero padding in some dims
        t2 = fd.ops.pad(t0, [0, 0, 2, 3])
        fd.add_output(t2)

        # zero padding in all dims
        t3 = fd.ops.pad(t0, [0, 0, 0, 0])
        fd.add_output(t3)

        # no padding provided in first dim
        t4 = fd.ops.pad(t0, [2, 3])
        fd.add_output(t4)

        # test padding with a value other than 0
        fill_val = fd.define_scalar(2.0)
        t5 = fd.ops.pad(t0, [2, 3], fill_val)
        fd.add_output(t5)

        # pad a broadcast dimension with a value other than 0
        t6 = fd.ops.pad(t0, [2, 3, 0, 0, 0, 0])
        fd.add_output(t6)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [1, 1, 1, 1]), nvf_out[0]
    )
    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [0, 0, 2, 3]), nvf_out[1]
    )
    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [0, 0, 0, 0]), nvf_out[2]
    )
    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [2, 3]), nvf_out[3]
    )
    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [2, 3], "constant", 2.0), nvf_out[4]
    )
    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [2, 3, 0, 0, 0, 0]), nvf_out[5]
    )


def test_pad_dynamic(nvfuser_direct_test):
    inputs = [
        torch.testing.make_tensor((1, 2, 3), dtype=torch.float32, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])

        S10 = fd.define_scalar(2.5, dtype=DataType.Float)
        S13 = fd.define_scalar(7, dtype=DataType.Int)
        S15 = fd.ops.mul(S10, S13)
        S16 = fd.ops.cast(S15, dtype=DataType.Int)
        t1 = fd.ops.pad(t0, [S16, S16, S16, S16])
        fd.add_output(t1)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    nvfuser_direct_test.assertEqual(
        torch.nn.functional.pad(inputs[0], [17, 17, 17, 17]), nvf_out[0]
    )


def test_cat(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 4, device="cuda"),
        torch.randn(2, 3, device="cuda"),
        torch.randn(4, 4, device="cuda"),
        torch.randn(0, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.from_pytorch(inputs[2])
        t3 = fd.from_pytorch(inputs[3])

        t3 = fd.ops.cat([t0, t1], 1)
        fd.add_output(t3)

        t4 = fd.ops.cat([t0, t2], 0)
        fd.add_output(t4)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    nvfuser_direct_test.assertEqual(
        torch.cat([inputs[0], inputs[1]], dim=1), nvf_out[0]
    )
    nvfuser_direct_test.assertEqual(
        torch.cat([inputs[0], inputs[2]], dim=0), nvf_out[1]
    )


def test_normal(nvfuser_direct_test):
    input_size = [64, 128, 1024]
    dtype = torch.float32
    device = "cuda"
    inputs = [
        torch.randn(*input_size, device=device, dtype=dtype),
    ]
    mean = 3.7
    std = 2.5

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        s_mean = fd.define_scalar(mean)
        s_std = fd.define_scalar(std)
        t1 = fd.ops.normal(s_mean, s_std, t0.shape(), dtype=DataType.Double)
        fd.add_output(t1)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    # Is there a better way to test distribution?!
    nvfuser_direct_test.assertTrue(
        nvf_out[0]
        .mean()
        .cpu()
        .float()
        .isclose(torch.tensor(mean), rtol=1e-2, atol=1e-2)
        .item()
    )
    nvfuser_direct_test.assertTrue(
        nvf_out[0]
        .std()
        .cpu()
        .float()
        .isclose(torch.tensor(std), rtol=1e-2, atol=1e-2)
        .item()
    )


def test_uniform(nvfuser_direct_test):
    input_size = [64, 128, 1024]
    dtype = torch.float32
    device = "cuda"
    inputs = [
        torch.randn(*input_size, device=device, dtype=dtype),
    ]
    lo = 1.8
    hi = 1223.5

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        s_lo = fd.define_scalar(lo)
        s_hi = fd.define_scalar(hi)
        t1 = fd.ops.uniform(s_lo, s_hi, t0.shape(), dtype=DataType.Double)
        fd.add_output(t1)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    # Is there a better way to test distribution?!
    nvfuser_direct_test.assertTrue(
        nvf_out[0]
        .mean()
        .cpu()
        .float()
        .isclose(torch.tensor((hi - lo) / 2.0), rtol=1e-2, atol=1e-2)
        .item()
    )
    nvfuser_direct_test.assertTrue(
        nvf_out[0]
        .min()
        .cpu()
        .float()
        .isclose(torch.tensor(lo), rtol=1e-2, atol=1e-2)
        .item()
    )
    nvfuser_direct_test.assertTrue(
        nvf_out[0]
        .max()
        .cpu()
        .float()
        .isclose(torch.tensor(hi), rtol=1e-2, atol=1e-2)
        .item()
    )


@pytest.mark.parametrize("padding_idx", [None, -2])
@pytest.mark.parametrize("max_norm", [None, 1e-5])
@pytest.mark.parametrize("norm_type", [None, 1.0])
@pytest.mark.parametrize("scale_grad_by_freq", [None, True])
@pytest.mark.parametrize("sparse", [None, True])
def test_embedding(
    padding_idx: None | int,
    max_norm: None | float,
    norm_type: None | float,
    scale_grad_by_freq: None | bool,
    sparse: None | bool,
):
    def fusion_func(
        fd: FusionDefinition,
        has_optional_inputs: list[bool],
        optional_inputs_dtypes: list[DataType],
    ):
        input = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Int,
            is_cpu=False,
        )
        weight = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        # padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
        optional_inputs = [None] * 5
        for idx in range(len(optional_inputs)):
            if has_optional_inputs[idx]:
                optional_inputs[idx] = fd.define_scalar(
                    value=None, dtype=optional_inputs_dtypes[idx]
                )
        out = fd.ops.embedding_fwd(input, weight, *optional_inputs)
        fd.add_output(out)

    N, S = 10, 3
    input = torch.randint(
        N, (S,), dtype=torch.int64, device="cuda", requires_grad=False
    )
    weight = torch.randn(N, S, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    optional_inputs_dtypes = [
        DataType.Int,
        DataType.Float,
        DataType.Float,
        DataType.Bool,
        DataType.Bool,
    ]

    # This is not in pytest_ops.py since the torch API does not accept None values for some arguments.
    # Different inputs for nvfuser and torch API cannot be handled within OpInfo
    optional_inputs = [padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
    has_optional_inputs = [None] * 5
    inputs = [input, weight]
    for idx, param in enumerate(optional_inputs):
        if param is not None:
            has_optional_inputs[idx] = True
            inputs.append(param)

    with FusionDefinition() as fd:
        fusion_func(
            fd,
            has_optional_inputs=has_optional_inputs,
            optional_inputs_dtypes=optional_inputs_dtypes,
        )
    nvf_out = fd.execute(inputs)

    norm_type = 2.0 if norm_type is None else norm_type
    scale_grad_by_freq = False if scale_grad_by_freq is None else scale_grad_by_freq
    sparse = False if sparse is None else sparse
    ref_out = torch.nn.functional.embedding(
        input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )
    torch.testing.assert_close(nvf_out[0], ref_out)


def test_output_stride_order(nvfuser_direct_test):
    inputs = [
        torch.arange(0, 24).reshape(2, 3, 4).cuda().float(),
    ]
    eager_out = inputs[0] + 3.0

    for perm in itertools.permutations(range(3), 3):
        # testing stride_order in set
        def fusion_set_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_scalar(3.0)
            t1 = fd.ops.add(t0, c0)
            t2 = fd.ops.stride_order(t1, perm)
            fd.add_output(t2)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_set_func, inputs)
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

        nvf_stride = nvf_out[0].stride()
        verify_stride_order(nvf_stride, perm)


def test_output_stride_order_with_reduction(nvfuser_direct_test):
    inputs = [torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float)]

    for stride_order in itertools.permutations(range(3), 3):

        def fusion_stride_order_op(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.ops.sum(T0, dims=[2])
            T2 = fd.ops.stride_order(T1, stride_order)
            fd.add_output(T2)

        with FusionDefinition() as fd:
            fusion_stride_order_op(fd)

        out = fd.execute(inputs)[0]
        verify_stride_order(out.stride(), stride_order)


def test_triu(nvfuser_direct_test):
    inputs = [
        torch.randn(4, 16, device="cuda", dtype=torch.float16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.ops.triu(t0, -1)
        fd.add_output(t1)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out0 = torch.triu(inputs[0], -1)
    nvfuser_direct_test.assertEqual(eager_out0, nvf_out[0])


def test_scatter_output_intermediate(nvfuser_direct_test):
    bsz = 128
    hidden = 1024
    scatter_size = 64
    scatter_dim = 0

    x = torch.randn([bsz, hidden], device="cuda")
    _, ind = torch.topk(x, k=scatter_size, dim=scatter_dim)
    src = torch.randn(scatter_size, hidden, device="cuda")
    inputs = [x, ind, src]

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Int,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.ops.scatter(T0, T1, T2, scatter_dim)
        T4 = fd.ops.sigmoid(T3)
        fd.add_output(T4)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.sigmoid(torch.scatter(x, scatter_dim, ind, src))
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_scatter_scalar_src(nvfuser_direct_test):
    bsz = 128
    hidden = 1024
    scatter_size = 64
    scatter_dim = 0

    x = torch.randn([bsz, hidden], device="cuda")
    _, ind = torch.topk(x, k=scatter_size, dim=scatter_dim)
    src = 1.5
    inputs = [x, ind, src]

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Int,
            is_cpu=False,
            stride_order=[1, 0],
        )
        S2 = fd.define_scalar(None, dtype=DataType.Double)
        T3 = fd.ops.scatter(T0, T1, S2, scatter_dim)
        fd.add_output(T3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.scatter(x, scatter_dim, ind, src)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_compute_tensor_descriptor(nvfuser_direct_test):
    configs = (
        (
            # size
            [2, 1, 3, 1, 4, 3],
            # stride
            [12, 4, 4, 4, 1, 0],
            # expected contiguity
            [True, None, True, None, True, None],
            # expected stride_order
            [5, 4, 3, 2, 1, 0],
        ),
        (
            [2, 3, 1, 5, 4],
            [28, 4, 14, 0, 1],
            [False, None, True, None, True],
            [4, 2, 3, 1, 0],
        ),
        (
            [2, 2, 1, 1, 2, 2, 2],
            [8, 4, 3, 9, 2, 0, 1],
            [None, True, True, None, True, None, True],
            [5, 4, 3, 6, 2, 1, 0],
        ),
        (
            [2, 2, 1, 2, 4, 2],
            [2, 32, 1, 8, 0, 4],
            [False, True, True, False, None, None],
            [2, 5, 0, 4, 1, 3],
        ),
        (
            [2, 2, 2, 2],
            [8, 4, 2, 1],
            [True, True, True, True],
            [3, 2, 1, 0],
        ),
        (
            [2, 1, 3, 1, 4],
            [24, 4, 8, 4, 2],
            [True, True, None, None, False],
            [4, 2, 3, 1, 0],
        ),
        (
            [2, 2, 2, 2],
            [8, 4, 0, 2],
            [True, True, None, False],
            [3, 2, 1, 0],
        ),
    )

    for sizes, strides, contiguity, stride_order in configs:
        computed_contiguity, computed_stride_order = compute_tensor_descriptor(
            sizes, strides
        )
        nvfuser_direct_test.assertEqual(computed_contiguity, contiguity)
        nvfuser_direct_test.assertEqual(computed_stride_order, stride_order)


def test_complex_constants(nvfuser_direct_test):
    inputs = [
        torch.arange(2, device="cuda").type(torch.complex64),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        c0 = fd.define_scalar(complex(3.0, 0.5))
        t1 = fd.ops.mul(t0, c0)
        fd.add_output(t1)

    (n,), _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    eager_out = inputs[0] * (3.0 + 0.5j)

    nvfuser_direct_test.assertEqual(eager_out, n)
    assert n.dtype == torch.complex64


def test_complex_rsqrt(nvfuser_direct_test):
    inputs = [
        torch.randn(4, device="cuda", dtype=torch.complex64),
        torch.randn(4, device="cuda", dtype=torch.complex128),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.rsqrt(t0)
        fd.add_output(t2)
        t3 = fd.ops.rsqrt(t1)
        fd.add_output(t3)

    (rfloat, rdouble), _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    at_rfloat = inputs[0].rsqrt()
    at_rdouble = inputs[1].rsqrt()

    nvfuser_direct_test.assertEqual(at_rfloat, rfloat)
    nvfuser_direct_test.assertEqual(at_rdouble, rdouble)


def test_constant_nans(nvfuser_direct_test):
    inputs = [
        torch.randn(4, 4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        c0 = fd.define_scalar(float("nan"))
        t1 = fd.ops.add(t0, c0)
        fd.add_output(t1)

    eager_out = inputs[0] + float("nan")

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_gcd(nvfuser_direct_test):
    inputs = [
        torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
        torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.gcd(t0, t1)
        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    nvfuser_direct_test.assertEqual(nvf_out[0], torch.gcd(inputs[0], inputs[1]))


def test_input_scalar(nvfuser_direct_test):
    inputs = [
        torch.randn((3,), dtype=torch.float32, device="cuda:0"),
        0.1,
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        S1 = fd.define_scalar()
        T1 = fd.ops.mul(T0, S1)
        fd.add_output(T1)

    # Just test that this executes, not that it's correct
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_integer_division(nvfuser_direct_test):
    inputs = [
        torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
        torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.div(t0, t1)
        t3 = fd.ops.truediv(t0, t1)
        fd.add_output(t2)
        fd.add_output(t3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    nvfuser_direct_test.assertEqual(
        nvf_out[0], torch.div(inputs[0], inputs[1], rounding_mode="trunc")
    )
    nvfuser_direct_test.assertEqual(nvf_out[1], torch.true_divide(inputs[0], inputs[1]))


def test_mark_alias_pass(nvfuser_direct_test):
    def reshape(fd: FusionDefinition) -> None:
        x = fd.define_tensor(
            [2, 3, 4], contiguity=[True, True, True], dtype=DataType.Float
        )
        y = fd.ops.reshape(x, [2, 12])
        fd.add_output(y)

    x = torch.rand(2, 3, 4, device="cuda")
    ys, _ = nvfuser_direct_test.exec_nvfuser(reshape, [x])
    nvfuser_direct_test.assertEqual(len(ys), 1)
    y = ys[0]

    nvfuser_direct_test.assertEqual(y.data_ptr(), x.data_ptr())


def test_misaligned_add(nvfuser_direct_test):
    inputs = [
        torch.ones(2**20 + 1, device="cuda")[1:],  # cannot vectorize
        torch.ones(2**20, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)

        fd.add_output(t2)

    # Fails because vectorization 4 is set but only 1 supported
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_nextafter(nvfuser_direct_test):
    inputs = [
        # torch.nextafter is only defined for float{32,64} tensor inputs
        torch.testing.make_tensor(4, device="cuda", dtype=torch.float32),
        torch.testing.make_tensor(4, device="cuda", dtype=torch.float64),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        s0 = fd.define_scalar(1.0, dtype=DataType.Float)
        s1 = fd.define_scalar(-1.0, dtype=DataType.Double)

        for a, b in itertools.product(
            [t0, t1, s0, s1],
            [t0, t1, s0, s1],
        ):
            # always enter the fusion...
            t = fd.ops.nextafter(a, b)
            if t.is_tensor():
                # ...but skip outputting scalars, which we don't support
                fd.add_output(t)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    ab = [inputs[0], inputs[1], 1.0, -1.0]
    i = 0
    for a, b in itertools.product(ab, ab):
        if not (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
            continue
        n = nvf_out[i]
        i += 1
        torch_out = torch.nextafter(
            torch.as_tensor(a, device="cuda"), torch.as_tensor(b, device="cuda")
        )
        nvfuser_direct_test.assertEqual(n, torch_out)


def test_prod(nvfuser_direct_test):
    inputs = [
        torch.ones(2, 4, 8, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])

        t1 = fd.ops.prod(t0, DataType.Float)
        t2 = fd.ops.prod(t0, 1, False, DataType.Float)
        t3 = fd.ops.prod(t0, 1, True, DataType.Float)
        t4 = fd.ops.prod(t0, [-1], False, DataType.Float)

        fd.add_output(t1)
        fd.add_output(t2)
        fd.add_output(t3)
        fd.add_output(t4)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    eager_outs = [
        torch.prod(inputs[0], dtype=torch.float32),
        torch.prod(inputs[0], 1, False, dtype=torch.float32),
        torch.prod(inputs[0], 1, True, dtype=torch.float32),
        torch.prod(inputs[0], -1, False, dtype=torch.float32),
    ]
    assert len(nvf_out) == len(eager_outs)

    for n, e in zip(nvf_out, eager_outs):
        nvfuser_direct_test.assertEqual(n, e)


def test_real_imag(nvfuser_direct_test):
    for dtype in [torch.complex128, torch.complex64]:
        inputs = [
            torch.randn(5, dtype=dtype, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            fd.add_output(fd.ops.real(t0))
            fd.add_output(fd.ops.imag(t0))

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

        nvfuser_direct_test.assertEqual(torch.real(inputs[0]), nvf_out[0])
        nvfuser_direct_test.assertEqual(torch.imag(inputs[0]), nvf_out[1])


def test_reduction_complex_number(nvfuser_direct_test):
    def test_dtype(torch_dtype):
        inputs = [torch.randn(2, 32, device="cuda", dtype=torch_dtype)]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.sum(t0, [-1], False, torch_dtype_to_nvfuser_dtype(torch_dtype))
            fd.add_output(t1)

        nvf_out1, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum(inputs[0], dim=-1)
        nvfuser_direct_test.assertEqual(eager_out, nvf_out1[0])

    list_of_dtype = [torch.complex64, torch.complex128]
    for torch_dtype in list_of_dtype:
        test_dtype(torch_dtype)


def test_right_shift_arithmetic(nvfuser_direct_test):
    inputs = [torch.tensor([-2147483648, 1073741824], dtype=torch.int32, device="cuda")]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        c0 = fd.define_scalar(3)
        t1 = fd.ops.bitwise_right_shift(t0, c0)
        fd.add_output(t1)

    nvf_out1, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out1 = torch.bitwise_right_shift(inputs[0], 3)
    nvfuser_direct_test.assertEqual(eager_out1, nvf_out1[0])


def test_segment_set(nvfuser_direct_test):
    inputs = [
        torch.randn(5, 5, 5, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.ops.neg(T0)
        T2 = fd.ops.segment_set(T1)
        T3 = fd.ops.relu(T2)
        fd.add_output(T3)

    eager_out = inputs[0].neg().relu()

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_signbit(nvfuser_direct_test):
    inputs = [
        torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
        torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.where(
            fd.ops.signbit(t0), fd.ops.neg(fd.ops.abs(t1)), fd.ops.abs(t1)
        )
        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    at_out = torch.where(
        torch.signbit(inputs[0]), -torch.abs(inputs[1]), torch.abs(inputs[1])
    )
    nvfuser_direct_test.assertEqual(at_out, nvf_out[0])


def test_tensor_shape(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 3, 4, device="cuda"),
        torch.randn(4, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [2])
        t2 = fd.ops.sub(t0, t1_b)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.sub(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_tensor_shape_expand_bcast(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        t1 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True])
        t2 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True])

        t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [0, 1, 2])
        t2_b = fd.ops.broadcast_in_dim(t2, t1_b.shape(), [0, 1, 2])

        fd.add_output(t2_b)

    inputs = [
        torch.randn(2, 3, 4, device="cuda"),
        torch.randn(2, 1, 4, device="cuda"),
        torch.randn(2, 1, 4, device="cuda"),
    ]

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out1 = prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1, 2])
    eager_out2 = prims.broadcast_in_dim(inputs[2], eager_out1.size(), [0, 1, 2])
    nvfuser_direct_test.assertEqual(eager_out2, nvf_out[0])


def test_tensor_shape_nobcast(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 3, device="cuda"),
        torch.randn(2, 3, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [0, 1])
        t2 = fd.ops.add(t0, t1_b)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_tensor_shape_with_output_bcast(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])

        t1 = fd.ops.sum(t0, dims=[2])
        t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [0, 1])

        fd.add_output(t1_b)

    inputs_1 = [
        torch.randn(2, 3, 4, device="cuda"),
    ]

    inputs_2 = [
        torch.randn(4, 5, 32, device="cuda"),
    ]

    inputs = inputs_1
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = prims.broadcast_in_dim(
        torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1]
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Testing Dynamic usage of same Fusion
    inputs = inputs_2
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = prims.broadcast_in_dim(
        torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1]
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_tensor_size_both_args_bcast(nvfuser_direct_test):
    inputs = [
        torch.randn(1, 3, device="cuda"),
        torch.randn(2, 1, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        t0_b = fd.ops.broadcast_in_dim(t0, [t1.size(0), t0.size(1)], [0, 1])
        t1_b = fd.ops.broadcast_in_dim(t1, [t1.size(0), t0.size(1)], [0, 1])
        t2 = fd.ops.add(t0_b, t1_b)

        fd.add_output(t2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = refs.add(
        prims.broadcast_in_dim(
            inputs[0], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]
        ),
        prims.broadcast_in_dim(
            inputs[1], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]
        ),
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_var_correction(nvfuser_direct_test):
    num_elem = 2
    inputs = [torch.randn(2, num_elem, device="cuda")]

    # use decorator to create fusion_func
    def fusion_decorator(correction):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.var(t0, [-1], correction)
            fd.add_output(t1)

        return fusion_func

    # correction must be less than the reduction factor, which is the input
    # numel divided by output numel.
    for correction in range(num_elem):
        fuser_result, _ = nvfuser_direct_test.exec_nvfuser(
            fusion_decorator(correction), inputs
        )
        torch_result = torch.var(inputs[0], [-1], correction=correction)
        nvfuser_direct_test.assertEqual(fuser_result[0], torch_result)


def test_var_mean_correction(nvfuser_direct_test):
    num_elem = 2
    inputs = [torch.randn(2, num_elem, device="cuda")]

    # use decorator to create fusion_func
    def fusion_decorator(correction):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1, t2 = fd.ops.var_mean(t0, [-1], correction)
            fd.add_output(t1)
            fd.add_output(t2)

        return fusion_func

    # correction must be less than the reduction factor, which is the input
    # numel divided by output numel.
    for correction in range(num_elem):
        fuser_result, _ = nvfuser_direct_test.exec_nvfuser(
            fusion_decorator(correction), inputs
        )
        torch_result = torch.var_mean(inputs[0], [-1], correction=correction)
        nvfuser_direct_test.assertEqual(fuser_result, torch_result)


def test_zero_size_dim(nvfuser_direct_test):
    inputs = [
        torch.ones(0, 0, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.define_tensor(
            shape=[0, 0], contiguity=[True, True], dtype=DataType.Float
        )
        t1 = fd.ops.relu(t0)
        fd.add_output(t1)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    eager_out = torch.relu(inputs[0])
    nvfuser_direct_test.assertEqual(eager_out.numel(), nvf_out[0].numel())


def test_allocation_domain_concretization(nvfuser_direct_test):
    inputs = [
        # we need an empty tensor here so we'll trigger `concretizeEmptyExtents`
        torch.randn((0,), dtype=torch.float64, device="cuda:0").as_strided(
            (1, 0, 1, 1), (0, 1, 1, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(
            shape=[1, -1, 1, 1],
            contiguity=[True, None, None, None],
            dtype=DataType.Double,
            is_cpu=False,
            stride_order=[0, 3, 2, 1],
        )
        S1 = fd.define_scalar(2.0, dtype=DataType.Double)
        T2 = fd.ops.mul(T1, S1)
        fd.add_output(T2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_ref = inputs[0] * 2.0
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_allocation_domain_index_select(nvfuser_direct_test):
    inputs = [
        torch.randn((252,), dtype=torch.float32, device="cuda:0").as_strided(
            (9, 28), (1, 9)
        ),
        torch.randint(0, 28, (4,), dtype=torch.int64, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0, 1],
        )
        T2 = fd.define_tensor(
            shape=[-1], contiguity=[True], dtype=DataType.Int, is_cpu=False
        )
        T3 = fd.ops.index_select(T1, T2, dim=1)
        fd.add_output(T3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_ref = torch.index_select(inputs[0], 1, inputs[1])
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_expand_to_zero(nvfuser_direct_test):
    inputs = [
        # This is an actually empty tensor
        torch.zeros((1, 0), dtype=torch.float32, device="cuda:0"),
        # This one is not actually empty, but should appear to be empty due to expand
        torch.zeros((1, 1), dtype=torch.float32, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.from_pytorch(inputs[1])
        T2 = fd.ops.broadcast_in_dim(T0, shape=[0, 0], broadcast_dims=[0, 1])
        T3 = fd.ops.broadcast_in_dim(T1, shape=[0, 0], broadcast_dims=[0, 1])
        fd.add_output(T2)
        fd.add_output(T3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    nvfuser_direct_test.assertEqual(nvf_out[0].shape, (0, 0))
    nvfuser_direct_test.assertEqual(nvf_out[1].shape, (0, 0))


def test_expanded_bcast_tensor(nvfuser_direct_test):
    inputs = [
        torch.tensor(1.5, device="cuda"),
        torch.randn(5, 5, 5, device="cuda"),
        torch.randint(0, 1, (5, 5), device="cuda").bool().unsqueeze(-1).expand(5, 5, 5),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.from_pytorch(inputs[1])
        T2 = fd.from_pytorch(inputs[2])
        T3 = fd.ops.add(T0, T1)
        T4 = fd.ops.add(T2, T3)
        fd.add_output(T4)

    eager_out = inputs[0] + inputs[1] + inputs[2]

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_inplace_update_on_non_contiguous_inputs(nvfuser_direct_test):
    inputs = [
        torch.randn(5, dtype=torch.float32, device="cuda:0").as_strided((2, 2), (1, 3)),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2, 2],
            contiguity=[False, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0, 1],
        )
        S1 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T2 = fd.ops.gt(T0, S1)
        S3 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T4 = fd.ops.where(T2, T0, S3)
        T5 = fd.ops.cast(T4, dtype=DataType.Float)
        T6 = fd.ops.set(T5)
        fd.add_output(T6, T0)
        fd.add_output(T6)

    ref_inp = inputs[0].clone()
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func,
        inputs,
    )

    assert len(nvf_out) == 1
    nvfuser_direct_test.assertEqual(nvf_out[0], inputs[0])
    nvfuser_direct_test.assertEqual(nvf_out[0], ref_inp.relu())


def test_pad_expanded_empty(nvfuser_direct_test):
    inputs = [
        torch.randn((0,), dtype=torch.float64, device="cuda:0").as_strided(
            (2, 0, 3), (0, 0, 0)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        S1 = fd.define_scalar(-3.70753, dtype=DataType.Double)
        T2 = fd.ops.pad(T0, [0, 0, 1, 1, 1, 0], S1)
        fd.add_output(T2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    torch_ref = torch.nn.functional.pad(
        inputs[0], (0, 0, 1, 1, 1, 0), "constant", -3.70753
    )

    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_pad_prior_cat(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 4, device="cuda"),
        torch.randn(3, 3, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])

        # pad tensors t0 and t1, so their first dimension are size 10.
        t0_pad = fd.ops.pad(t0, [0, 0, 0, 8])
        t1_pad = fd.ops.pad(t1, [0, 0, 0, 7])

        t3 = fd.ops.cat([t0_pad, t1_pad], 1)
        fd.add_output(t3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    # pad tensors t0 and t1, so their first dimension are size 10.
    pad_input0 = torch.nn.functional.pad(inputs[0], [0, 0, 0, 8])
    pad_input1 = torch.nn.functional.pad(inputs[1], [0, 0, 0, 7])
    nvfuser_direct_test.assertEqual(
        torch.cat([pad_input0, pad_input1], dim=1), nvf_out[0]
    )


def test_replaced_sizes_pr2714(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.ops.exp(T0)
        T3 = fd.ops.tanh(T1)
        S4 = fd.define_scalar(4, dtype=DataType.Int)
        T6 = fd.ops.reshape(T2, new_shape=[S4])
        S7 = fd.define_scalar(4, dtype=DataType.Int)
        T9 = fd.ops.reshape(T3, new_shape=[S7])
        T10 = fd.ops.add(T6, T9)
        T11 = fd.ops.reciprocal(T0)
        T12 = fd.ops.mul(T3, T11)
        S13 = fd.define_scalar(2.00000, dtype=DataType.Double)
        S14 = fd.ops.reciprocal(S13)
        T15 = fd.ops.mul(T10, S14)
        fd.add_output(T10)
        fd.add_output(T12)
        fd.add_output(T15)

    inputs = [
        torch.randn((4,), dtype=torch.float32, device="cuda:0").as_strided(
            (2, 2), (2, 1)
        ),
        torch.randn((4,), dtype=torch.float32, device="cuda:0").as_strided(
            (2, 2), (2, 1)
        ),
    ]

    nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_reshape_squeeze_concretization(nvfuser_direct_test):
    inputs = [
        torch.randn((100,), dtype=torch.float32, device="cuda:0").as_strided(
            (2, 5, 10), (50, 10, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.ops.slice(
            T0, start_indices=[0, 0, 0], end_indices=[1, 2, 4], strides=[1, 1, 1]
        )
        S2 = fd.define_scalar(1, dtype=DataType.Int)
        S3 = fd.define_scalar(8, dtype=DataType.Int)
        T6 = fd.ops.reshape(T1, new_shape=[S2, S3])
        T7 = fd.ops.reshape(T6, new_shape=[S3])
        fd.add_output(T7)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_sum_sliced_reshape_to_broadcast(nvfuser_direct_test):
    inputs = [torch.randn((24, 128, 25, 32), dtype=torch.float32, device="cuda:0")]

    def fusion_func(fd: FusionDefinition) -> None:
        T18 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        S91 = fd.define_scalar(12, dtype=DataType.Int)
        S92 = fd.define_scalar(128, dtype=DataType.Int)
        S93 = fd.define_scalar(25, dtype=DataType.Int)
        S94 = fd.define_scalar(32, dtype=DataType.Int)
        S95 = fd.define_scalar(2, dtype=DataType.Int)
        T97 = fd.ops.reshape(T18, new_shape=[S91, S92, S93, S94, S95])
        T98 = fd.ops.slice(
            T97,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[12, 128, 25, 32, 1],
            strides=[1, 1, 1, 1, 1],
        )
        T89 = fd.ops.sum(T98, dims=[4], keepdim=False, dtype=DataType.Null)
        fd.add_output(T89)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


# See https://github.com/NVIDIA/Fuser/issues/3833
def test_bcast_squeeze_replace_aliased_output(nvfuser_direct_test):
    inputs = [
        torch.testing.make_tensor((1, 1, 576), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((1, 576), dtype=torch.bfloat16, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 1, 576],
            contiguity=[None, None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 576],
            contiguity=[None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T5 = fd.ops.reshape(T0, new_shape=[1, 576])
        T6 = fd.ops.set(T5)
        fd.add_output(T6, T1)
        fd.add_output(T5)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    assert len(nvf_out) == 1
    nvfuser_direct_test.assertEqual(nvf_out[0], inputs[0].squeeze(1))


def test_broadcast_and_stride_order(nvfuser_direct_test):
    inputs = [
        torch.randn(2, 3, 4, dtype=torch.float32, device="cuda:0"),
    ]

    # Direct bindings does not support `add_output` with stride_order argument.
    # Instead, we use `stride_order` operation to set the stride order before
    # adding the output.
    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.ops.broadcast(T0, is_broadcast_dim=[False, True, False, False])
        T2 = fd.ops.stride_order(T1, stride_order=[0, 1, 2, 3])
        fd.add_output(T2)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    nvfuser_direct_test.assertEqual(nvf_out[0], inputs[0].unsqueeze(1))
    nvfuser_direct_test.assertEqual(nvf_out[0].stride(), (1, 2, 2, 6))


def test_right_shift_logical(nvfuser_direct_test):
    dtypes = [torch.int32, torch.int64]
    input = torch.tensor(
        [
            -1,
            -2147483648,
            1073741824,
            -64463884,
            -65968277,
            4042311,
            -98914167,
            5526216,
        ],
        device="cuda",
    )

    # expected_outputs given by jax.lax.shift_right_logical(inputs, 3)
    expected_outputs = [
        torch.tensor(
            [
                536870911,
                268435456,
                134217728,
                528812926,
                528624877,
                505288,
                524506641,
                690777,
            ],
            dtype=torch.int32,
            device="cuda",
        ),
        torch.tensor(
            [
                2305843009213693951,
                2305843008945258496,
                134217728,
                2305843009205635966,
                2305843009205447917,
                505288,
                2305843009201329681,
                690777,
            ],
            dtype=torch.int64,
            device="cuda",
        ),
    ]

    for idx, dtype in enumerate(dtypes):
        current_input = input.to(dtype)

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(current_input)
            c0 = fd.define_scalar(3)
            t1 = fd.ops.logical_right_shift(t0, c0)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, [current_input])
        nvfuser_direct_test.assertEqual(nvf_out[0], expected_outputs[idx])


def test_right_shift_logical_sizeof_dtype(nvfuser_direct_test):
    dtypes = [torch.int32, torch.int64]
    input = torch.tensor(
        [
            -1,
            -2147483648,
            1073741824,
            -64463884,
            -65968277,
            4042311,
            -98914167,
            5526216,
        ],
        device="cuda",
    )

    for idx, dtype in enumerate(dtypes):
        current_input = input.to(dtype)
        num_bits = 32 if (dtype == torch.int32) else 64

        # expected_outputs given by jax.lax.shift_right_logical(inputs, sizeof(dtype))
        # >>> jax.lax.shift_right_logical(input.to('cpu').numpy(), 32)
        # Array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
        # >>> jax.lax.shift_right_logical(input.to('cpu').numpy(), 64)
        # Array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
        expected_output = torch.zeros_like(current_input)

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(current_input)
            c0 = fd.define_scalar(None, dtype=DataType.Int)
            t1 = fd.ops.logical_right_shift(t0, c0)
            fd.add_output(t1)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
            fusion_func, [current_input, num_bits]
        )
        nvfuser_direct_test.assertEqual(nvf_out[0], expected_output)
