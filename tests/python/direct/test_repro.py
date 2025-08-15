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


def test_issue1273(nvfuser_direct_test):
    """
    Test for issue 1273 - tests squeeze of dynamic input handling.

    This test verifies that squeeze operations work correctly with:
    - Dynamic input tensors with strided layouts
    - Complex operations involving reshape, var_mean, broadcast_in_dim
    - Layer normalization-like operations with variance and mean
    - Proper handling of tensor reshaping and broadcasting
    - Correct computation of normalization operations
    """
    inputs = [
        torch.randn((4,), dtype=torch.float32, device="cuda:0").as_strided(
            (2, 2), (2, 1)
        ),
        1e-05,
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        S1 = fd.define_scalar(None, dtype=DataType.Double)
        T7 = fd.ops.reshape(T0, new_shape=[2, 1, 2])
        T8, T9 = fd.ops.var_mean(T7, dims=[2], correction=0, keepdim=False)
        T14 = fd.ops.broadcast_in_dim(T8, shape=[2, 1, 1], broadcast_dims=[0, 1])
        T19 = fd.ops.broadcast_in_dim(T9, shape=[2, 1, 1], broadcast_dims=[0, 1])
        T20 = fd.ops.add(T14, S1)
        T21 = fd.ops.rsqrt(T20)
        T26 = fd.ops.broadcast_in_dim(T19, shape=[2, 1, 2], broadcast_dims=[0, 1, 2])
        T27 = fd.ops.sub(T7, T26)
        T32 = fd.ops.broadcast_in_dim(T21, shape=[2, 1, 2], broadcast_dims=[0, 1, 2])
        T33 = fd.ops.mul(T27, T32)
        T37 = fd.ops.reshape(T33, new_shape=[2, 2])
        fd.add_output(T37)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    t7 = inputs[0].reshape((2, 1, 2))
    t8 = t7.var(dim=2, unbiased=False)
    t9 = t7.mean(dim=2)
    t27 = t7 - t9.unsqueeze(-1).expand((2, 1, 2))
    t32 = torch.rsqrt(inputs[1] + t8.unsqueeze(-1)).expand((2, 1, 2))
    torch_ref = (t27 * t32).reshape((2, 2))
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_issue1277(nvfuser_direct_test):
    """
    Test for issue 1277 - tests complex operations with strided tensors and slicing.

    This test verifies that complex operations work correctly with:
    - Multiple strided tensors with complex memory layouts
    - Extensive slicing operations with different indices
    - Padding operations with various configurations
    - Permutation and arithmetic operations
    - Complex tensor manipulation sequences
    - Proper handling of resized extents and expression simplification
    """
    inputs = [
        0.5,
        0.5,
        torch.randn((20,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 5, 4), (0, 0, 4, 1)
        ),
        torch.randn((20,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 5, 4), (0, 0, 4, 1)
        ),
        torch.randn((20,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 5, 4), (0, 0, 4, 1)
        ),
        torch.randn((20,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 5, 4), (0, 0, 4, 1)
        ),
        torch.randn((1600,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 5, 16), (320, 80, 16, 1)
        ),
        torch.randn((1600,), dtype=torch.float32, device="cuda:0").as_strided(
            (5, 4, 16, 5), (320, 80, 5, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(None, dtype=DataType.Double)
        S1 = fd.define_scalar(None, dtype=DataType.Double)
        T2 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T3 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T4 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T5 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T6 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T7 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T8 = fd.ops.mul(T6, S0)
        T9 = fd.ops.slice(
            T8,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
        )
        T10 = fd.ops.slice(
            T8,
            start_indices=[0, 0, 0, 4],
            end_indices=[5, 4, 5, 16],
            strides=[1, 1, 1, 1],
        )
        S11 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T12 = fd.ops.pad(T10, [4, 0, 0, 0, 0, 0, 0, 0], S11)
        S13 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T14 = fd.ops.mul(S13, T9)
        S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T16 = fd.ops.mul(S15, T9)
        T17 = fd.ops.mul(T16, T3)
        T18 = fd.ops.mul(T14, T2)
        T19 = fd.ops.slice(
            T17,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 2],
            strides=[1, 1, 1, 1],
        )
        T20 = fd.ops.slice(
            T17,
            start_indices=[0, 0, 0, 2],
            end_indices=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
        )
        T21 = fd.ops.neg(T19)
        S22 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T23 = fd.ops.pad(T21, [2, 0, 0, 0, 0, 0, 0, 0], S22)
        T24 = fd.ops.add(T18, T23)
        S25 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T26 = fd.ops.pad(T20, [0, 2, 0, 0, 0, 0, 0, 0], S25)
        T27 = fd.ops.add(T24, T26)
        S28 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T29 = fd.ops.pad(T27, [0, 12, 0, 0, 0, 0, 0, 0], S28)
        T30 = fd.ops.add(T12, T29)
        T31 = fd.ops.mul(T7, S1)
        T32 = fd.ops.permute(T31, dims=[0, 1, 3, 2])
        T33 = fd.ops.slice(
            T32,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
        )
        T34 = fd.ops.slice(
            T32,
            start_indices=[0, 0, 0, 4],
            end_indices=[5, 4, 5, 16],
            strides=[1, 1, 1, 1],
        )
        S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T36 = fd.ops.pad(T34, [4, 0, 0, 0, 0, 0, 0, 0], S35)
        S37 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T38 = fd.ops.mul(S37, T33)
        S39 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T40 = fd.ops.mul(S39, T33)
        T41 = fd.ops.mul(T40, T5)
        T42 = fd.ops.mul(T38, T4)
        T43 = fd.ops.slice(
            T41,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 2],
            strides=[1, 1, 1, 1],
        )
        T44 = fd.ops.slice(
            T41,
            start_indices=[0, 0, 0, 2],
            end_indices=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
        )
        T45 = fd.ops.neg(T43)
        S46 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T47 = fd.ops.pad(T45, [2, 0, 0, 0, 0, 0, 0, 0], S46)
        T48 = fd.ops.add(T42, T47)
        S49 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T50 = fd.ops.pad(T44, [0, 2, 0, 0, 0, 0, 0, 0], S49)
        T51 = fd.ops.add(T48, T50)
        S52 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T53 = fd.ops.pad(T51, [0, 12, 0, 0, 0, 0, 0, 0], S52)
        T54 = fd.ops.add(T36, T53)
        fd.add_output(T54)
        fd.add_output(T30)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue1279(nvfuser_direct_test):
    """
    Test for issue 1279 - tests var_mean operations with casting.

    This test verifies that var_mean operations work correctly with:
    - Half-precision (float16) input tensors
    - Casting operations between different data types
    - Variance and mean computation with correction
    - correction is 0 because reduction dimension is 1, which can cause
      division by zero.
    - Proper handling of dimension reduction
    - Multiple output tensors from var_mean operation
    """
    inputs = [
        torch.randn(2, 1, 2, dtype=torch.float16, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, 1, -1],
            contiguity=[True, None, True],
            dtype=DataType.Half,
            is_cpu=False,
        )
        T4 = fd.ops.cast(T0, dtype=DataType.Float)
        T5, T6 = fd.ops.var_mean(T4, dims=[1], correction=0, keepdim=False)
        T7 = fd.ops.cast(T5, dtype=DataType.Half)
        T8 = fd.ops.cast(T6, dtype=DataType.Half)
        fd.add_output(T7)
        fd.add_output(T8)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    a = inputs[0].type(torch.float32)
    b, c = torch.var_mean(a, dim=1, correction=0)
    d = b.type(torch.float16)
    e = c.type(torch.float16)

    nvfuser_direct_test.assertEqual(nvf_out[0], d)
    nvfuser_direct_test.assertEqual(nvf_out[1], e)


def test_issue1310(nvfuser_direct_test):
    """
    Test for issue 1310 - tests input forwarding with multiple UnaryOps.

    This test verifies that inputs are properly forwarded when an input is used in multiple
    UnaryOps, some having one and others having multiple further uses:
    - Multiple cast operations on the same input tensor
    - Different reduction operations on cast results
    - Proper handling of tensor aliasing and forwarding
    - Correct computation of multiple reduction operations
    """
    inputs = [torch.randn((16, 128, 768), dtype=torch.bfloat16, device="cuda:0")]

    def fusion_func(fd: FusionDefinition) -> None:
        T3 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T14 = fd.ops.cast(
            T3, dtype=DataType.Float
        )  # NOTE that RHS is same, but the result is assigned to different variables
        T15 = fd.ops.cast(
            T3, dtype=DataType.Float
        )  # NOTE that RHS is same, but the result is assigned to different variables
        T16 = fd.ops.sum(T15, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T20 = fd.ops.sum(T14, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T31 = fd.ops.sum(T14, dims=[2], keepdim=False, dtype=DataType.Null)
        fd.add_output(T16)
        fd.add_output(T20)
        fd.add_output(T31)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    t14 = inputs[0].type(torch.float32)
    t16 = t14.sum([0, 1])
    t31 = t14.sum([2])
    nvfuser_direct_test.assertEqual(nvf_out[0], t16)
    nvfuser_direct_test.assertEqual(nvf_out[1], t16)  # T16 == T20
    nvfuser_direct_test.assertEqual(nvf_out[2], t31)


def test_issue1393(nvfuser_direct_test):
    """
    Test for issue 1393 - tests complex operations with strided tensors and broadcasting.

    This test verifies that complex operations work correctly with:
    - Strided tensors with non-standard memory layouts
    - Casting operations between different data types
    - Multiplication and reshape operations
    - Broadcast_in_dim operations with explicit dimensions
    - Complex tensor manipulation sequences
    - Proper handling of tensor contiguity and strides
    """
    inputs = [
        torch.randn((5,), dtype=torch.float16, device="cuda:0").as_strided(
            (3, 4, 5), (0, 0, 1)
        ),
        torch.randn((3,), dtype=torch.float16, device="cuda:0").as_strided(
            (3, 4), (1, 0)
        ),
        torch.randn((4,), dtype=torch.float16, device="cuda:0").as_strided(
            (3, 4), (0, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[None, None, True],
            dtype=DataType.Half,
            is_cpu=False,
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, None],
            dtype=DataType.Half,
            is_cpu=False,
        )
        T2 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[None, True],
            dtype=DataType.Half,
            is_cpu=False,
        )
        T3 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T2, dtype=DataType.Float)
        T5 = fd.ops.mul(T3, T4)
        T6 = fd.ops.cast(T5, dtype=DataType.Half)
        S7 = fd.define_scalar(3, dtype=DataType.Int)
        S8 = fd.define_scalar(4, dtype=DataType.Int)
        S9 = fd.define_scalar(1, dtype=DataType.Int)
        T11 = fd.ops.reshape(T6, new_shape=[S7, S8, S9])
        S12 = fd.define_scalar(3, dtype=DataType.Int)
        S13 = fd.define_scalar(4, dtype=DataType.Int)
        S14 = fd.define_scalar(5, dtype=DataType.Int)
        T16 = fd.ops.broadcast_in_dim(
            T11, shape=[S12, S13, S14], broadcast_dims=[0, 1, 2]
        )
        T17 = fd.ops.cast(T16, dtype=DataType.Float)
        T18 = fd.ops.cast(T0, dtype=DataType.Float)
        T19 = fd.ops.mul(T17, T18)
        T20 = fd.ops.cast(T19, dtype=DataType.Half)
        fd.add_output(T20)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_ref = inputs[0] * (inputs[1] * inputs[2]).unsqueeze(-1)
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


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
