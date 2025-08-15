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


def test_issue1691(nvfuser_direct_test):
    """
    Test for issue 1691 - tests complex reduction operations with reshape and multiplication.

    This test verifies that complex reduction operations work correctly with:
    - Strided tensors with non-standard memory layouts
    - Multiple reduction operations along different dimensions
    - Reshape operations with scalar-defined shapes
    - Multiplication operations between reshaped tensors
    - Final reduction operations on multiplied results
    - Proper handling of tensor contiguity and stride order
    """
    inputs = [
        torch.randn((12,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 3, 4), (12, 4, 1)
        ),
        torch.randn((12,), dtype=torch.float32, device="cuda:0").as_strided(
            (4, 3), (3, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, -1, -1],
            contiguity=True,
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.ops.sum(T1, dims=[1], keepdim=False, dtype=DataType.Null)  # 1D
        T3 = fd.ops.sum(T0, dims=[1, 0], keepdim=False, dtype=DataType.Null)  # 1D
        S4 = fd.define_scalar(4, dtype=DataType.Int)
        T6 = fd.ops.reshape(T2, new_shape=[S4])
        S7 = fd.define_scalar(4, dtype=DataType.Int)
        T9 = fd.ops.reshape(T3, new_shape=[S7])
        T10 = fd.ops.mul(T6, T9)
        T11 = fd.ops.sum(T10, dims=[0], keepdim=False, dtype=DataType.Null)
        fd.add_output(T11)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    torch_ref = (inputs[0].sum(dim=[0, 1]) * inputs[1].sum(dim=1)).sum(dim=0)
    nvfuser_direct_test.assertEqual(nvf_out[0], torch_ref)


def test_issue1706(nvfuser_direct_test):
    """
    Test for issue 1706 - tests complex operations derived from Llama2 network.

    This test verifies that complex operations work correctly with:
    - Large tensors with bfloat16 precision
    - Extensive casting operations between different data types
    - Complex mathematical operations (rsqrt, pow, reciprocal)
    - Multiple broadcast_in_dim operations with different shapes
    - Reduction operations along different dimensions
    - Complex tensor manipulation sequences
    - Proper handling of tensor contiguity and memory layouts
    """
    inputs = [
        1e-6,
        10,
        4096,
        4096,
        torch.randn(
            (
                1,
                4096,
                4096,
            ),
            dtype=torch.bfloat16,
            device="cuda:0",
        ),
        torch.randn((10, 32), dtype=torch.bfloat16, device="cuda:0"),
        torch.randn(
            (
                1,
                4096,
                4096,
            ),
            dtype=torch.bfloat16,
            device="cuda:0",
        ),
        torch.randn(
            (
                1,
                4096,
                1,
            ),
            dtype=torch.bfloat16,
            device="cuda:0",
        ),
        torch.randn(
            (
                1,
                1,
                4096,
            ),
            dtype=torch.bfloat16,
            device="cuda:0",
        ).expand(1, 4096, 4096),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(None, dtype=DataType.Double)
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        S3 = fd.define_scalar(None, dtype=DataType.Int)
        T4 = fd.define_tensor(
            shape=[1, -1, -1],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T5 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T6 = fd.define_tensor(
            shape=[1, -1, -1],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T7 = fd.define_tensor(
            shape=[1, -1, 1],
            contiguity=[None, True, None],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T8 = fd.define_tensor(
            shape=[1, -1, -1],
            contiguity=[None, None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        T9 = fd.ops.cast(T6, dtype=DataType.Float)
        T10 = fd.ops.cast(T6, dtype=DataType.Float)
        T11 = fd.ops.cast(T7, dtype=DataType.Float)
        T12 = fd.ops.rsqrt(T11)
        T13 = fd.ops.cast(T12, dtype=DataType.BFloat16)
        S14 = fd.define_scalar(1, dtype=DataType.Int)
        S15 = fd.define_scalar(4096, dtype=DataType.Int)
        S16 = fd.define_scalar(4096, dtype=DataType.Int)
        T18 = fd.ops.broadcast_in_dim(
            T13, shape=[S14, S15, S16], broadcast_dims=[0, 1, 2]
        )
        T19 = fd.ops.cast(T6, dtype=DataType.Float)
        T20 = fd.ops.cast(T18, dtype=DataType.Float)
        T21 = fd.ops.mul(T19, T20)
        T22 = fd.ops.cast(T21, dtype=DataType.BFloat16)
        T23 = fd.ops.cast(T8, dtype=DataType.Float)
        T24 = fd.ops.cast(T22, dtype=DataType.Float)
        T25 = fd.ops.cast(T4, dtype=DataType.Float)
        T26 = fd.ops.mul(T25, T24)
        T27 = fd.ops.mul(T25, T23)
        T28 = fd.ops.cast(T27, dtype=DataType.BFloat16)
        T29 = fd.ops.cast(T26, dtype=DataType.BFloat16)
        T30 = fd.ops.cast(T29, dtype=DataType.Float)
        T31 = fd.ops.sum(T30, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T32 = fd.ops.cast(T31, dtype=DataType.BFloat16)
        T33 = fd.ops.cast(T32, dtype=DataType.Float)
        S34 = fd.define_scalar(2.00000, dtype=DataType.Double)
        S35 = fd.ops.reciprocal(S34)
        T36 = fd.ops.mul(T33, S35)
        T37 = fd.ops.cast(T36, dtype=DataType.BFloat16)
        T38 = fd.ops.cast(T28, dtype=DataType.Float)
        T39 = fd.ops.mul(T38, T20)
        T40 = fd.ops.mul(T38, T19)
        T41 = fd.ops.cast(T40, dtype=DataType.BFloat16)
        T42 = fd.ops.cast(T39, dtype=DataType.BFloat16)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.sum(T43, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T45 = fd.ops.cast(T44, dtype=DataType.BFloat16)
        S46 = fd.define_scalar(1, dtype=DataType.Int)
        S47 = fd.define_scalar(4096, dtype=DataType.Int)
        S48 = fd.define_scalar(1, dtype=DataType.Int)
        T50 = fd.ops.broadcast_in_dim(T45, shape=[S46, S47, S48], broadcast_dims=[1])
        T51 = fd.ops.cast(T50, dtype=DataType.Float)
        S52 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T53 = fd.ops.mul(S52, T51)
        S54 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T55 = fd.ops.pow(T12, S54)
        T56 = fd.ops.mul(T53, T55)
        T57 = fd.ops.cast(T56, dtype=DataType.BFloat16)
        T58 = fd.ops.cast(T57, dtype=DataType.Float)
        T59 = fd.ops.cast(T58, dtype=DataType.BFloat16)
        T60 = fd.ops.cast(T59, dtype=DataType.Float)
        S61 = fd.ops.reciprocal(S0)
        T62 = fd.ops.mul(T60, S61)
        T63 = fd.ops.sum(T62, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        S64 = fd.define_scalar(1, dtype=DataType.Int)
        S65 = fd.define_scalar(4096, dtype=DataType.Int)
        T67 = fd.ops.broadcast_in_dim(T63, shape=[S64, S65], broadcast_dims=[1])
        S68 = fd.define_scalar(1, dtype=DataType.Int)
        S69 = fd.define_scalar(4096, dtype=DataType.Int)
        S70 = fd.define_scalar(1, dtype=DataType.Int)
        T72 = fd.ops.broadcast_in_dim(T67, shape=[S68, S69, S70], broadcast_dims=[0, 1])
        S73 = fd.define_scalar(1, dtype=DataType.Int)
        S74 = fd.define_scalar(4096, dtype=DataType.Int)
        S75 = fd.define_scalar(4096, dtype=DataType.Int)
        T77 = fd.ops.broadcast_in_dim(
            T72, shape=[S73, S74, S75], broadcast_dims=[0, 1, 2]
        )
        T78 = fd.ops.cast(T77, dtype=DataType.BFloat16)
        T79 = fd.ops.cast(T78, dtype=DataType.Float)
        T80 = fd.ops.mul(T79, T10)
        T81 = fd.ops.mul(T79, T9)
        T82 = fd.ops.cast(T81, dtype=DataType.BFloat16)
        T83 = fd.ops.cast(T80, dtype=DataType.BFloat16)
        T84 = fd.ops.cast(T42, dtype=DataType.Float)
        T85 = fd.ops.cast(T83, dtype=DataType.Float)
        T86 = fd.ops.add(T84, T85)
        T87 = fd.ops.cast(T86, dtype=DataType.BFloat16)
        T88 = fd.ops.cast(T87, dtype=DataType.Float)
        T89 = fd.ops.cast(T82, dtype=DataType.Float)
        T90 = fd.ops.add(T88, T89)
        T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
        T92 = fd.ops.cast(T91, dtype=DataType.Float)
        T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
        T94 = fd.ops.cast(T92, dtype=DataType.BFloat16)
        T95 = fd.ops.cast(T93, dtype=DataType.Float)
        T96 = fd.ops.cast(T5, dtype=DataType.Float)
        S97 = fd.define_scalar(2.00000, dtype=DataType.Double)
        S98 = fd.ops.reciprocal(S97)
        T99 = fd.ops.mul(T96, S98)
        T100 = fd.ops.cast(T99, dtype=DataType.BFloat16)
        fd.add_output(T100)
        fd.add_output(T37)
        fd.add_output(T94)
        fd.add_output(T95)

    # skip pytorch check because fusion is derived from llama2 network.
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue1872(nvfuser_direct_test):
    """
    Test for issue 1872 - tests full tensor creation with slice operations and casting.

    This test verifies that tensor operations work correctly with:
    - Full tensor creation with scalar fill values
    - Slice operations with different start and end indices
    - Casting operations between different data types
    - Multiple output tensors from a single fusion
    - Proper handling of tensor shapes and data types
    """

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(1.00000, dtype=DataType.Double)
        S1 = fd.define_scalar(5, dtype=DataType.Int)
        T3 = fd.ops.full(shape=[S1], fill_value=S0, dtype=DataType.Float)
        T4 = fd.ops.slice(T3, start_indices=[0], end_indices=[2], strides=[1])
        T5 = fd.ops.cast(T4, dtype=DataType.Half)
        T6 = fd.ops.slice(T3, start_indices=[2], end_indices=[5], strides=[1])
        T7 = fd.ops.cast(T6, dtype=DataType.Half)
        fd.add_output(T5)
        fd.add_output(T7)

    nvfuser_direct_test.exec_nvfuser(fusion_func, [])


def test_issue1953(nvfuser_direct_test):
    """
    Test for issue 1953 - tests complex operations with strided tensors and multiple data types.

    This test verifies that complex operations work correctly with:
    - Large strided tensors with complex memory layouts
    - Multiple data types (Float32 and BFloat16)
    - Complex tensor operations (permute, reshape, slice, sum, mul, neg, add)
    - Broadcasting operations with different shapes
    - Padding operations with scalar values
    - Multiple output tensors from a single fusion
    - Proper handling of tensor contiguity and stride order
    """
    inputs = [
        128,
        256,
        6,
        24,
        2,
        128,
        256,
        6,
        24,
        2,
        torch.randn((6144,), dtype=torch.float32, device="cuda:0").as_strided(
            (128, 256, 6, 24), (0, 24, 0, 1)
        ),
        torch.randn((6144,), dtype=torch.float32, device="cuda:0").as_strided(
            (128, 256, 6, 24), (0, 24, 0, 1)
        ),
        torch.randn((9437184,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (128, 6, 256, 48), (73728, 48, 288, 1)
        ),
        torch.randn((9437184,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (128, 6, 256, 48), (73728, 48, 288, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(None, dtype=DataType.Int)
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        S3 = fd.define_scalar(None, dtype=DataType.Int)
        S4 = fd.define_scalar(None, dtype=DataType.Int)
        S5 = fd.define_scalar(None, dtype=DataType.Int)
        S6 = fd.define_scalar(None, dtype=DataType.Int)
        S7 = fd.define_scalar(None, dtype=DataType.Int)
        S8 = fd.define_scalar(None, dtype=DataType.Int)
        S9 = fd.define_scalar(None, dtype=DataType.Int)
        T10 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, True, None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T11 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, True, None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T12 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T13 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T14 = fd.ops.cast(T13, dtype=DataType.Float)
        T15 = fd.ops.permute(T14, dims=[0, 2, 1, 3])
        S16 = fd.define_scalar(128, dtype=DataType.Int)
        S17 = fd.define_scalar(256, dtype=DataType.Int)
        S18 = fd.define_scalar(6, dtype=DataType.Int)
        S19 = fd.define_scalar(24, dtype=DataType.Int)
        S20 = fd.define_scalar(2, dtype=DataType.Int)
        T22 = fd.ops.reshape(T15, new_shape=[S16, S17, S18, S19, S20])
        T23 = fd.ops.slice(
            T22,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[128, 256, 6, 24, 1],
            strides=[1, 1, 1, 1, 1],
        )
        T24 = fd.ops.slice(
            T22,
            start_indices=[0, 0, 0, 0, 1],
            end_indices=[128, 256, 6, 24, 2],
            strides=[1, 1, 1, 1, 1],
        )
        T25 = fd.ops.sum(T24, dims=[4], keepdim=False, dtype=DataType.Null)
        T26 = fd.ops.sum(T23, dims=[4], keepdim=False, dtype=DataType.Null)
        T27 = fd.ops.mul(T25, T10)
        T28 = fd.ops.mul(T25, T11)
        T29 = fd.ops.neg(T26)
        T30 = fd.ops.mul(T29, T11)
        T31 = fd.ops.add(T27, T30)
        T32 = fd.ops.cast(T31, dtype=DataType.BFloat16)
        T33 = fd.ops.mul(T26, T10)
        T34 = fd.ops.add(T28, T33)
        T35 = fd.ops.cast(T34, dtype=DataType.BFloat16)
        S36 = fd.define_scalar(128, dtype=DataType.Int)
        S37 = fd.define_scalar(256, dtype=DataType.Int)
        S38 = fd.define_scalar(6, dtype=DataType.Int)
        S39 = fd.define_scalar(24, dtype=DataType.Int)
        S40 = fd.define_scalar(1, dtype=DataType.Int)
        T42 = fd.ops.broadcast_in_dim(
            T32, shape=[S36, S37, S38, S39, S40], broadcast_dims=[0, 1, 2, 3]
        )
        S43 = fd.define_scalar(128, dtype=DataType.Int)
        S44 = fd.define_scalar(256, dtype=DataType.Int)
        S45 = fd.define_scalar(6, dtype=DataType.Int)
        S46 = fd.define_scalar(24, dtype=DataType.Int)
        S47 = fd.define_scalar(1, dtype=DataType.Int)
        T49 = fd.ops.broadcast_in_dim(
            T35, shape=[S43, S44, S45, S46, S47], broadcast_dims=[0, 1, 2, 3]
        )
        S50 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T51 = fd.ops.pad(T42, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], S50)
        S52 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T53 = fd.ops.pad(T49, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], S52)
        T54 = fd.ops.cast(T51, dtype=DataType.Float)
        T55 = fd.ops.cast(T53, dtype=DataType.Float)
        T56 = fd.ops.add(T54, T55)
        T57 = fd.ops.cast(T56, dtype=DataType.BFloat16)
        T58 = fd.ops.cast(T12, dtype=DataType.Float)
        T59 = fd.ops.permute(T58, dims=[0, 2, 1, 3])
        S60 = fd.define_scalar(128, dtype=DataType.Int)
        S61 = fd.define_scalar(256, dtype=DataType.Int)
        S62 = fd.define_scalar(6, dtype=DataType.Int)
        S63 = fd.define_scalar(24, dtype=DataType.Int)
        S64 = fd.define_scalar(2, dtype=DataType.Int)
        T66 = fd.ops.reshape(T59, new_shape=[S60, S61, S62, S63, S64])
        T67 = fd.ops.slice(
            T66,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[128, 256, 6, 24, 1],
            strides=[1, 1, 1, 1, 1],
        )
        T68 = fd.ops.slice(
            T66,
            start_indices=[0, 0, 0, 0, 1],
            end_indices=[128, 256, 6, 24, 2],
            strides=[1, 1, 1, 1, 1],
        )
        T69 = fd.ops.sum(T68, dims=[4], keepdim=False, dtype=DataType.Null)
        T70 = fd.ops.sum(T67, dims=[4], keepdim=False, dtype=DataType.Null)
        T71 = fd.ops.mul(T69, T10)
        T72 = fd.ops.mul(T69, T11)
        T73 = fd.ops.neg(T70)
        T74 = fd.ops.mul(T73, T11)
        T75 = fd.ops.add(T71, T74)
        T76 = fd.ops.cast(T75, dtype=DataType.BFloat16)
        T77 = fd.ops.mul(T70, T10)
        T78 = fd.ops.add(T72, T77)
        T79 = fd.ops.cast(T78, dtype=DataType.BFloat16)
        S80 = fd.define_scalar(128, dtype=DataType.Int)
        S81 = fd.define_scalar(256, dtype=DataType.Int)
        S82 = fd.define_scalar(6, dtype=DataType.Int)
        S83 = fd.define_scalar(24, dtype=DataType.Int)
        S84 = fd.define_scalar(1, dtype=DataType.Int)
        T86 = fd.ops.broadcast_in_dim(
            T76, shape=[S80, S81, S82, S83, S84], broadcast_dims=[0, 1, 2, 3]
        )
        S87 = fd.define_scalar(128, dtype=DataType.Int)
        S88 = fd.define_scalar(256, dtype=DataType.Int)
        S89 = fd.define_scalar(6, dtype=DataType.Int)
        S90 = fd.define_scalar(24, dtype=DataType.Int)
        S91 = fd.define_scalar(1, dtype=DataType.Int)
        T93 = fd.ops.broadcast_in_dim(
            T79, shape=[S87, S88, S89, S90, S91], broadcast_dims=[0, 1, 2, 3]
        )
        S94 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T95 = fd.ops.pad(T86, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], S94)
        S96 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T97 = fd.ops.pad(T93, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], S96)
        T98 = fd.ops.cast(T95, dtype=DataType.Float)
        T99 = fd.ops.cast(T97, dtype=DataType.Float)
        T100 = fd.ops.add(T98, T99)
        T101 = fd.ops.cast(T100, dtype=DataType.BFloat16)
        fd.add_output(T57)
        fd.add_output(T101)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue2275_repro1(nvfuser_direct_test):
    """
    Test for issue 2275 repro1 - tests unpadded concatenation operations with complex tensor manipulations.

    This test verifies that complex operations work correctly with:
    - Large strided tensors with complex memory layouts
    - BFloat16 precision operations
    - Complex tensor operations (cast, mul, sum, broadcast_in_dim, rsqrt, linear, reshape, permute, slice, neg, cat)
    - Broadcasting operations with different shapes
    - Linear operations with weight matrices
    - Multiple slice operations with different indices
    - Concatenation operations with negative dimensions
    - Proper handling of tensor contiguity and stride order
    """
    inputs = [
        torch.randn((4096,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (2, 4096, 4096), (0, 0, 1)
        ),
        torch.randn((33554432,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (2, 4096, 4096), (16777216, 4096, 1)
        ),
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (2, 32, 4096, 128), (0, 0, 128, 1)
        ),
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (2, 32, 4096, 128), (0, 0, 128, 1)
        ),
        torch.randn((25165824,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (6144, 4096), (4096, 1)
        ),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[None, None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T5 = fd.ops.cast(T1, dtype=DataType.Float)
        T6 = fd.ops.mul(T5, T5)
        T7 = fd.ops.sum(T6, dims=[2], keepdim=False, dtype=DataType.Null)
        S8 = fd.define_scalar(2, dtype=DataType.Int)
        S9 = fd.define_scalar(4096, dtype=DataType.Int)
        S10 = fd.define_scalar(1, dtype=DataType.Int)
        T12 = fd.ops.broadcast_in_dim(T7, shape=[S8, S9, S10], broadcast_dims=[0, 1])
        S13 = fd.define_scalar(4096.00, dtype=DataType.Double)
        S14 = fd.ops.reciprocal(S13)
        T15 = fd.ops.mul(T12, S14)
        S16 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
        T17 = fd.ops.add(T15, S16)
        T18 = fd.ops.rsqrt(T17)
        S19 = fd.define_scalar(2, dtype=DataType.Int)
        S20 = fd.define_scalar(4096, dtype=DataType.Int)
        S21 = fd.define_scalar(4096, dtype=DataType.Int)
        T23 = fd.ops.broadcast_in_dim(
            T18, shape=[S19, S20, S21], broadcast_dims=[0, 1, 2]
        )
        T24 = fd.ops.mul(T5, T23)
        T25 = fd.ops.cast(T0, dtype=DataType.Float)
        T26 = fd.ops.mul(T24, T25)
        T27 = fd.ops.cast(T26, dtype=DataType.BFloat16)
        T28 = fd.ops.linear(T27, T4)
        S29 = fd.define_scalar(2, dtype=DataType.Int)
        S30 = fd.define_scalar(4096, dtype=DataType.Int)
        S31 = fd.define_scalar(8, dtype=DataType.Int)
        S32 = fd.define_scalar(6, dtype=DataType.Int)
        S33 = fd.define_scalar(128, dtype=DataType.Int)
        T35 = fd.ops.reshape(T28, new_shape=[S29, S30, S31, S32, S33])
        T36 = fd.ops.permute(T35, dims=[0, 2, 3, 1, 4])
        T37 = fd.ops.slice(
            T36,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[2, 8, 4, 4096, 128],
            strides=[1, 1, 1, 1, 1],
        )

        S47 = fd.define_scalar(2, dtype=DataType.Int)
        S48 = fd.define_scalar(32, dtype=DataType.Int)
        S49 = fd.define_scalar(4096, dtype=DataType.Int)
        S50 = fd.define_scalar(128, dtype=DataType.Int)
        T52 = fd.ops.reshape(T37, new_shape=[S47, S48, S49, S50])
        T59 = fd.ops.slice(
            T52,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 128],
            strides=[1, 1, 1, 1],
        )
        T60 = fd.ops.slice(
            T59,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 64],
            strides=[1, 1, 1, 1],
        )
        T61 = fd.ops.slice(
            T59,
            start_indices=[0, 0, 0, 64],
            end_indices=[2, 32, 4096, 128],
            strides=[1, 1, 1, 1],
        )
        T62 = fd.ops.cast(T61, dtype=DataType.Float)
        T63 = fd.ops.neg(T62)
        T64 = fd.ops.cast(T63, dtype=DataType.BFloat16)
        T65 = fd.ops.cat([T64, T60], dim=-1)
        T66 = fd.ops.cast(T59, dtype=DataType.Float)
        T67 = fd.ops.cast(T2, dtype=DataType.Float)
        T68 = fd.ops.mul(T66, T67)
        T69 = fd.ops.cast(T65, dtype=DataType.Float)
        T70 = fd.ops.cast(T3, dtype=DataType.Float)
        T71 = fd.ops.mul(T69, T70)
        T72 = fd.ops.add(T68, T71)
        T73 = fd.ops.cast(T72, dtype=DataType.BFloat16)

        T87 = fd.ops.slice(
            T52,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 0],
            strides=[1, 1, 1, 1],
        )
        T88 = fd.ops.cat([T73, T87], dim=-1)

        fd.add_output(T88)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue2275_repro2(nvfuser_direct_test):
    """
    Test for issue 2275 repro2 - tests unpadded concatenation operations with trigonometric functions.

    This test verifies that complex operations work correctly with:
    - Large tensors with BFloat16 precision
    - Multiple slice operations with different indices
    - Trigonometric operations (sin, cos)
    - Negation and casting operations
    - Concatenation operations with negative dimensions
    - Proper handling of tensor shapes and operations
    """
    inputs = [torch.randn((2, 32, 4096, 128), dtype=torch.bfloat16, device="cuda:0")]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])

        T1 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 128],
            strides=[1, 1, 1, 1],
        )
        T2 = fd.ops.slice(
            T1,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 64],
            strides=[1, 1, 1, 1],
        )
        T3 = fd.ops.slice(
            T1,
            start_indices=[0, 0, 0, 64],
            end_indices=[2, 32, 4096, 128],
            strides=[1, 1, 1, 1],
        )
        T4 = fd.ops.cast(fd.ops.neg(T3), DataType.BFloat16)
        T5 = fd.ops.cat([T4, T2], dim=-1)
        T6 = fd.ops.add(fd.ops.sin(T5), fd.ops.cos(T5))
        T7 = fd.ops.cast(T6, DataType.BFloat16)

        T100 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 0, 0],
            end_indices=[2, 32, 4096, 0],
            strides=[1, 1, 1, 1],
        )
        T101 = fd.ops.cat([T7, T100], dim=-1)
        fd.add_output(T101)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


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


def test_issue2755(nvfuser_direct_test):
    """
    Test for issue 2755 - tests slice operations with negation.

    This test verifies that slice operations work correctly with:
    - Basic tensor slicing with different start and end indices
    - Negation operations on sliced tensors
    - Multiple slice operations in sequence
    - Proper handling of tensor shapes and operations
    """

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.define_tensor(shape=[-1])
        t1 = fd.ops.slice(
            t0,
            start_indices=[0],
            end_indices=[5],
        )
        t2 = fd.ops.neg(t1)
        t3 = fd.ops.slice(
            t2,
            start_indices=[0],
            end_indices=[2],
        )
        t4 = fd.ops.neg(t3)
        fd.add_output(t4)

    inputs = [torch.randn((10,), dtype=torch.float32, device="cuda:0")]
    nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_issue3292(nvfuser_direct_test):
    """
    Test for issue 3292 - tests complex tensor operations with manual normalization and padding.

    This test verifies that complex operations work correctly with:
    - Tensor reshaping and permutation operations
    - Multiple slice operations with manual normalization
    - Negation and concatenation operations with manual padding
    - Multiplication and addition operations
    - Complex tensor manipulation sequences
    - Proper handling of tensor shapes and operations
    """
    inputs = [
        torch.testing.make_tensor((5, 5, 576), dtype=torch.float32, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T2 = fd.define_tensor(
            shape=[5, 5, 576],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T30 = fd.ops.reshape(T2, new_shape=[5, 5, 1, 9, 64])
        T31 = fd.ops.permute(T30, dims=[0, 2, 3, 1, 4])
        T50 = fd.ops.slice(
            T31,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[5, 1, 7, 5, 64],
            strides=[1, 1, 1, 1, 1],
            manual_normalization=0,
        )
        T108 = fd.ops.reshape(T50, new_shape=[5, 7, 5, 64])
        T136 = fd.ops.slice(
            T108,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 7, 5, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T152 = fd.ops.slice(
            T108,
            start_indices=[0, 0, 0, 32],
            end_indices=[5, 7, 5, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T153 = fd.ops.neg(T152)
        T154 = fd.ops.cat([T153, T136], dim=-1, manual_padding=0)
        T161 = fd.ops.mul(T108, T108)
        T168 = fd.ops.mul(T154, T154)
        T169 = fd.ops.add(T161, T168)
        T185 = fd.ops.slice(
            T108,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 7, 5, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T201 = fd.ops.slice(
            T108,
            start_indices=[0, 0, 0, 32],
            end_indices=[5, 7, 5, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T202 = fd.ops.neg(T201)
        T203 = fd.ops.cat([T202, T185], dim=-1, manual_padding=0)
        T205 = fd.ops.mul(T203, T203)
        T222 = fd.ops.slice(
            T108,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 7, 5, 0],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T223 = fd.ops.cat([T169, T222], dim=-1, manual_padding=0)
        fd.add_output(T223)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
