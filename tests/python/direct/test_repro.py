# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser_direct import FusionDefinition, DataType


def test_issue1129(nvfuser_direct_test):
    """
    Test for issue 1129 - tests reshape and index_select operations with strided tensors.

    This test verifies that reshape and index_select operations work correctly
    with tensors that have non-standard strides, particularly when reshaping
    a tensor and then using it for index selection.
    """
    inputs = [
        torch.randint(0, 10, (25,), dtype=torch.int64, device="cuda:0").as_strided(
            (5, 5), (5, 1)
        ),
        torch.randn((129024,), dtype=torch.float32, device="cuda:0").as_strided(
            (2016, 64), (64, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Int,
            is_cpu=False,
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        S2 = fd.define_scalar(25, dtype=DataType.Int)
        T4 = fd.ops.reshape(T0, new_shape=[S2])
        T5 = fd.ops.index_select(T1, T4, dim=0)
        S6 = fd.define_scalar(5, dtype=DataType.Int)
        S7 = fd.define_scalar(5, dtype=DataType.Int)
        S8 = fd.define_scalar(64, dtype=DataType.Int)
        T10 = fd.ops.reshape(T5, new_shape=[S6, S7, S8])
        fd.add_output(T10)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_ref = torch.reshape(
        torch.index_select(inputs[1], 0, torch.reshape(inputs[0], [25])), [5, 5, 64]
    )
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_issue1246(nvfuser_direct_test):
    """
    Test for issue 1246 - tests concatenation with empty tensors and strided tensors.

    This test verifies that concatenation operations work correctly with:
    - Strided tensors with non-standard memory layouts
    - Empty tensors (zero-sized dimensions)
    - Both with and without additional operations after concatenation
    """
    inputs = [
        torch.randn((8388608,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 32, 2048, 128), (8388608, 262144, 128, 1)
        ),
        torch.randn((0,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 32, 2048, 0), (8388608, 262144, 128, 1)
        ),
    ]

    for final_mul in [False, True]:

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[1, -1, -1, -1],
                contiguity=[None, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            T1 = fd.define_tensor(
                shape=[1, -1, -1, -1],
                contiguity=[None, True, False, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            S2 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T3 = fd.ops.mul(T0, S2)
            T4 = fd.ops.cat([T3, T1], dim=-1)
            if final_mul:
                # NOTE: original repro does not have this final op
                S3 = fd.define_scalar(1.00000, dtype=DataType.Double)
                T5 = fd.ops.mul(T4, S3)
                fd.add_output(T5)
            else:
                fd.add_output(T4)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
        torch_ref = torch.cat([2.0 * inputs[0], inputs[1]], dim=-1)
        nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_issue1270(nvfuser_direct_test):
    """
    Test for issue 1270 - tests operations with empty tensors and dead code removal.

    This test verifies that operations work correctly with:
    - Empty tensors (zero-sized dimensions)
    - Dead code removal during fusion optimization
    - Complex operations involving casting, multiplication, and reduction
    - Proper handling of empty tensor operations that should not cause problems
    """
    inputs = [
        torch.randn(0, device="cuda", dtype=torch.bfloat16).as_strided((5, 0), (1, 0)),
        torch.randn(0, device="cuda", dtype=torch.bfloat16).as_strided((5, 0), (0, 1)),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, None],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T2 = fd.ops.cast(T1, dtype=DataType.Float)
        S3 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T4 = fd.ops.full(fill_value=S3, shape=[5, 0], dtype=DataType.BFloat16)
        T5 = fd.ops.cast(T4, dtype=DataType.Float)
        T6 = fd.ops.mul(T2, T5)
        T7 = fd.ops.cast(T0, dtype=DataType.Float)
        T8 = fd.ops.mul(T7, T5)
        T24 = fd.ops.sum(T6, dims=[1], keepdim=False, dtype=DataType.Null)
        T11 = fd.ops.sum(T8, dims=[0], keepdim=False, dtype=DataType.Null)
        fd.add_output(T24)
        fd.add_output(T11)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    t2 = inputs[1].type(torch.float32)
    t4 = torch.full([5, 0], 1.0, dtype=torch.bfloat16, device="cuda")
    t5 = t4.type(torch.float32)
    t6 = t2 * t5
    t7 = inputs[0].type(torch.float32)
    t8 = t7 * t5
    t24 = t6.sum([1])
    t11 = t8.sum([0])
    nvfuser_direct_test.assertEqual(nvf_out[0], t24)
    nvfuser_direct_test.assertEqual(nvf_out[1], t11)


def test_issue2545(nvfuser_direct_test):
    """
    Test for issue 2545 - tests complex operations with empty tensors and concatenation.

    This test verifies that complex operations work correctly with:
    - Empty tensors (zero-sized dimensions)
    - Multiple concatenation operations
    - Conditional operations (where, lt)
    - Arithmetic operations with scalars
    - Proper handling of empty tensor removal during optimization
    """
    inputs = [
        torch.randint(0, 10, (2,), dtype=torch.int64, device="cuda:0").as_strided(
            (2,), (1,)
        ),
        torch.randint(0, 10, (0,), dtype=torch.int64, device="cuda:0").as_strided(
            (0,), (1,)
        ),
    ]

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Int,
            is_cpu=False,
            stride_order=[0],
        )
        T1 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Int,
            is_cpu=False,
            stride_order=[0],
        )
        S2 = fd.define_scalar(0, dtype=DataType.Int)
        T3 = fd.ops.lt(T0, S2)
        S4 = fd.define_scalar(5, dtype=DataType.Int)
        S5 = fd.define_scalar(0, dtype=DataType.Int)
        T6 = fd.ops.where(T3, S4, S5)
        T7 = fd.ops.add(T0, T6)
        S8 = fd.define_scalar(0, dtype=DataType.Int)
        T9 = fd.ops.add(T7, S8)
        T10 = fd.ops.cat([T1, T9], dim=0)
        S11 = fd.define_scalar(0, dtype=DataType.Int)
        T12 = fd.ops.add(T10, S11)
        T13 = fd.ops.cat([T1, T12], dim=0)
        S14 = fd.define_scalar(5, dtype=DataType.Int)
        T15 = fd.ops.add(T10, S14)
        T16 = fd.ops.cat([T13, T15], dim=0)
        S17 = fd.define_scalar(10, dtype=DataType.Int)
        T18 = fd.ops.add(T10, S17)
        fd.add_output(T18)
        fd.add_output(T16)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue2549(nvfuser_direct_test):
    """
    Test for issue 2549 - tests broadcast_in_dim and division operations.

    This test verifies that broadcast_in_dim and division operations work correctly:
    - Broadcasting a tensor to a specific shape with explicit broadcast dimensions
    - Division operations with broadcasted tensors
    - Proper handling of tensor shapes and strides
    - Correct computation of division with broadcasted operands
    """
    a = torch.ones(4, 1, dtype=torch.double, device="cuda")
    b = torch.ones(4, 4, dtype=torch.double, device="cuda")

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            sizes=a.shape, strides=a.stride(), dtype=DataType.Double, is_cpu=False
        )
        T1 = fd.define_tensor(
            sizes=b.shape, strides=b.stride(), dtype=DataType.Double, is_cpu=False
        )
        T2 = fd.ops.broadcast_in_dim(T0, shape=[4, 4], broadcast_dims=[0, 1])
        T3 = fd.ops.div(T1, T2)
        fd.add_output(T3)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, [a, b])
    nvfuser_direct_test.assertEqual(nvf_out[0], b / a)
