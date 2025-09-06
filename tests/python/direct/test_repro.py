# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser_direct import FusionDefinition, DataType
from nvfuser.pytorch_utils import RecordTorchMemory

# Use smaller range for torch.testing.make_tensor for nvfuser_direct.validate
LOW_VAL = -2
HIGH_VAL = 2


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
    Test for issue 2545 - tests empty tensor handling with concatenation operations.

    This test verifies that operations with empty tensors work correctly,
    particularly when concatenating tensors where one or more inputs
    are empty tensors.
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

    This test verifies that broadcast_in_dim operations work correctly
    with division operations, particularly when broadcasting tensors
    with different shapes.
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


def test_issue4444(nvfuser_direct_test):
    """
    Test for issue 4444 - complex tensor operations with multiple slice operations,
    manual normalization, padding, and reshaping operations.

    This test validates:
    - Multiple slice operations with manual normalization
    - Complex tensor reshaping and permutation operations
    - Manual padding operations with scalar values
    - Broadcast operations with specific dimensions
    - Cast operations between different data types
    - Complex mathematical sequences involving negation, addition, and multiplication
    - Proper handling of tensor shapes and operations
    - Scalar definition and vector operations
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 64, 16384, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[16384, 128],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[16384, 128],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 64, 16384, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 64, 16384, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T20 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 64, 16384, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T21 = fd.ops.cast(T20, dtype=DataType.Float)
        T27 = fd.ops.broadcast_in_dim(
            T1, shape=[1, 64, 16384, 128], broadcast_dims=[2, 3]
        )
        T28 = fd.ops.mul(T27, T21)
        T29 = fd.ops.cast(T28, dtype=DataType.BFloat16)
        T35 = fd.ops.broadcast_in_dim(
            T2, shape=[1, 64, 16384, 128], broadcast_dims=[2, 3]
        )
        T51 = fd.ops.slice(
            T29,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 64, 16384, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T52 = fd.ops.mul(T35, T21)
        S53 = fd.define_scalar(0, dtype=DataType.Int)
        T59 = fd.ops.full(
            shape=[1, 64, 16384, 0], fill_value=S53, dtype=DataType.BFloat16
        )
        T75 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 64, 16384, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T76 = fd.ops.cast(T51, dtype=DataType.Float)
        S77 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T87 = fd.ops.pad(T59, [0, 128, 0, 0, 0, 0, 0, 0], S77)
        T88 = fd.ops.cast(T75, dtype=DataType.Float)
        T89 = fd.ops.neg(T76)
        T90 = fd.ops.cast(T87, dtype=DataType.Float)
        T91 = fd.ops.mul(T27, T88)
        T92 = fd.ops.cast(T89, dtype=DataType.BFloat16)
        T93 = fd.ops.add(T90, T52)
        T94 = fd.ops.cast(T91, dtype=DataType.BFloat16)
        S95 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T105 = fd.ops.pad(T92, [64, 0, 0, 0, 0, 0, 0, 0], S95)
        T121 = fd.ops.slice(
            T94,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 64, 16384, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T122 = fd.ops.mul(T35, T88)
        T123 = fd.ops.cast(T105, dtype=DataType.Float)
        T124 = fd.ops.cast(T121, dtype=DataType.Float)
        T140 = fd.ops.slice(
            T29,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 64, 16384, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T141 = fd.ops.add(T93, T123)
        T142 = fd.ops.neg(T124)
        S143 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T153 = fd.ops.pad(T140, [0, 64, 0, 0, 0, 0, 0, 0], S143)
        T154 = fd.ops.cast(T142, dtype=DataType.BFloat16)
        T155 = fd.ops.add(T90, T122)
        T156 = fd.ops.cast(T153, dtype=DataType.Float)
        S157 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T167 = fd.ops.pad(T154, [64, 0, 0, 0, 0, 0, 0, 0], S157)
        T168 = fd.ops.add(T141, T156)
        T169 = fd.ops.cast(T167, dtype=DataType.Float)
        T170 = fd.ops.cast(T168, dtype=DataType.BFloat16)
        T186 = fd.ops.slice(
            T94,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 64, 16384, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T187 = fd.ops.add(T155, T169)
        T194 = fd.ops.reshape(T4, new_shape=[1, 8, 8, 16384, 128])
        T201 = fd.ops.reshape(T170, new_shape=[1, 8, 8, 16384, 128])
        S202 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T212 = fd.ops.pad(T186, [0, 64, 0, 0, 0, 0, 0, 0], S202)
        T213 = fd.ops.cast(T194, dtype=DataType.Float)
        T214 = fd.ops.cast(T201, dtype=DataType.Float)
        T215 = fd.ops.cast(T212, dtype=DataType.Float)
        T216 = fd.ops.sum(T213, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T217 = fd.ops.sum(T214, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T218 = fd.ops.add(T187, T215)
        T219 = fd.ops.cast(T216, dtype=DataType.BFloat16)
        T220 = fd.ops.cast(T217, dtype=DataType.BFloat16)
        T221 = fd.ops.cast(T218, dtype=DataType.BFloat16)
        T228 = fd.ops.broadcast_in_dim(
            T219, shape=[1, 8, 1, 16384, 128], broadcast_dims=[1, 3, 4]
        )
        T235 = fd.ops.broadcast_in_dim(
            T220, shape=[1, 8, 1, 16384, 128], broadcast_dims=[1, 3, 4]
        )
        T242 = fd.ops.reshape(T221, new_shape=[1, 8, 8, 16384, 128])
        T243 = fd.ops.cat([T242, T235, T228], dim=2, manual_padding=0)
        T244 = fd.ops.permute(T243, dims=[0, 3, 1, 2, 4])
        T249 = fd.ops.reshape(T244, new_shape=[1, 16384, 10240])
        T253 = fd.ops.reshape(T249, new_shape=[16384, 10240])
        T254 = fd.ops.permute(T253, dims=[1, 0])
        fd.add_output(T253)
        fd.add_output(T254)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (1, 64, 16384, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (16384, 128),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (16384, 128),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 64, 16384, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 64, 16384, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_issue4459(nvfuser_direct_test):
    """
    Test for issue 4459 - complex tensor operations with broadcast, reshape, and mathematical operations.

    This test verifies complex tensor operations involving broadcast operations,
    tensor reshaping, and mathematical sequences with multiple operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[4, 32],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0, 1],
        )
        T1 = fd.define_tensor(
            shape=[4, 32, 1, 1, 1],
            contiguity=[True, True, None, None, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[4, 3, 2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[4, 32, 10, 64, 64],
            contiguity=[True, True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[4, 3, 2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[320],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T4 = fd.define_tensor(
            shape=[320],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T5 = fd.define_tensor(
            shape=[4, 320, 66, 66],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T12 = fd.ops.broadcast_in_dim(T0, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1])
        T19 = fd.ops.broadcast_in_dim(
            T12, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T26 = fd.ops.broadcast_in_dim(
            T1, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T27 = fd.ops.sub(T2, T19)
        T33 = fd.ops.reshape(T3, new_shape=[1, 320, 1, 1])
        T34 = fd.ops.mul(T27, T26)
        T40 = fd.ops.reshape(T4, new_shape=[1, 320, 1, 1])
        T46 = fd.ops.broadcast_in_dim(
            T33, shape=[4, 320, 64, 64], broadcast_dims=[0, 1, 2, 3]
        )
        T52 = fd.ops.reshape(T34, new_shape=[4, 320, 64, 64])
        T58 = fd.ops.broadcast_in_dim(
            T40, shape=[4, 320, 64, 64], broadcast_dims=[0, 1, 2, 3]
        )
        T59 = fd.ops.mul(T52, T46)
        T60 = fd.ops.add(T59, T58)
        T61 = fd.ops.neg(T60)
        T62 = fd.ops.exp(T61)
        S63 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T73 = fd.ops.pad(T5, [-1, -1, -1, -1, 0, 0, 0, 0], S63)
        S74 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T75 = fd.ops.add(S74, T62)
        T76 = fd.ops.mul(T60, T73)
        T77 = fd.ops.reciprocal(T75)
        T78 = fd.ops.neg(T76)
        T79 = fd.ops.mul(T78, T77)
        T80 = fd.ops.mul(T79, T77)
        T81 = fd.ops.mul(T80, T62)
        T82 = fd.ops.neg(T81)
        T83 = fd.ops.mul(T77, T73)
        T84 = fd.ops.add(T83, T82)
        T85 = fd.ops.mul(T46, T84)
        T92 = fd.ops.reshape(T85, new_shape=[4, 32, 10, 64, 64])
        T93 = fd.ops.mul(T27, T92)
        T94 = fd.ops.sum(T93, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
        T101 = fd.ops.broadcast_in_dim(
            T94, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
        )
        S102 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T103 = fd.ops.pow(T1, S102)
        S104 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T105 = fd.ops.mul(S104, T101)
        T106 = fd.ops.mul(T26, T92)
        T107 = fd.ops.mul(T105, T103)
        T108 = fd.ops.neg(T106)
        T109 = fd.ops.sum(T107, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
        T110 = fd.ops.sum(T108, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
        T117 = fd.ops.broadcast_in_dim(
            T0, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
        )
        T124 = fd.ops.broadcast_in_dim(
            T109, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
        )
        T131 = fd.ops.broadcast_in_dim(
            T110, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
        )
        T138 = fd.ops.broadcast_in_dim(
            T117, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T145 = fd.ops.broadcast_in_dim(
            T124, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T146 = fd.ops.sum(T131, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
        T147 = fd.ops.sub(T2, T138)
        S148 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T149 = fd.ops.mul(S148, T145)
        T156 = fd.ops.broadcast_in_dim(
            T146, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
        )
        T157 = fd.ops.mul(T149, T147)
        T164 = fd.ops.broadcast_in_dim(
            T156, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
        )
        S165 = fd.define_scalar(40960.0, dtype=DataType.Double)
        S166 = fd.ops.reciprocal(S165)
        T167 = fd.ops.mul(T157, S166)
        S168 = fd.define_scalar(2.44141e-05, dtype=DataType.Double)
        T169 = fd.ops.mul(S168, T164)
        T170 = fd.ops.add(T169, T167)
        T171 = fd.ops.add(T106, T170)
        T177 = fd.ops.reshape(T171, new_shape=[4, 320, 64, 64])
        T184 = fd.ops.reshape(T177, new_shape=[1, 4, 320, 64, 64])
        T185 = fd.ops.permute(T184, dims=[0, 3, 4, 1, 2])
        T192 = fd.ops.reshape(T185, new_shape=[1, 1, 4096, 4, 320])
        T193 = fd.ops.mul(T52, T84)
        T194 = fd.ops.sum(T192, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T195 = fd.ops.sum(T193, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
        T196 = fd.ops.sum(T84, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
        T200 = fd.ops.reshape(T194, new_shape=[16384, 320])
        T206 = fd.ops.broadcast_in_dim(T195, shape=[1, 320, 1, 1], broadcast_dims=[1])
        T212 = fd.ops.broadcast_in_dim(T196, shape=[1, 320, 1, 1], broadcast_dims=[1])
        T213 = fd.ops.permute(T200, dims=[1, 0])
        T214 = fd.ops.sum(T194, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T217 = fd.ops.reshape(T206, new_shape=[320])
        T220 = fd.ops.reshape(T212, new_shape=[320])
        fd.add_output(T177)
        fd.add_output(T200)
        fd.add_output(T213)
        fd.add_output(T214)
        fd.add_output(T217)
        fd.add_output(T220)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.randn(128, dtype=torch.float32, device="cuda:0").as_strided(
            (4, 32), (1, 4)
        ),
        torch.testing.make_tensor(
            (4, 32, 1, 1, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (4, 32, 10, 64, 64),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (320,),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (320,),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (4, 320, 66, 66),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_issue4670(nvfuser_direct_test):
    """
    Test for issue 4670 - iota operations with broadcast and comparison operations.

    This test verifies iota operations with scalar parameters, broadcast operations,
    and comparison operations with conditional logic.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(129, dtype=DataType.Int)
        S1 = fd.define_scalar(0, dtype=DataType.Int)
        S2 = fd.define_scalar(1, dtype=DataType.Int)
        T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int)
        T4 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
        S5 = fd.define_scalar(128, dtype=DataType.Int)
        S6 = fd.ops.size(T3, dim=0)
        T8 = fd.ops.expand(T4, shape=[S5, S6])
        S9 = fd.define_scalar(128, dtype=DataType.Int)
        S10 = fd.define_scalar(0, dtype=DataType.Int)
        S11 = fd.define_scalar(1, dtype=DataType.Int)
        T12 = fd.ops.iota(S9, S10, S11, dtype=DataType.Int)
        T13 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
        S14 = fd.ops.size(T12, dim=0)
        S15 = fd.define_scalar(129, dtype=DataType.Int)
        T17 = fd.ops.expand(T13, shape=[S14, S15])
        T18 = fd.ops.gt(T8, T17)
        T19 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
        S20 = fd.define_scalar(128, dtype=DataType.Int)
        T22 = fd.ops.expand(T19, shape=[S20, S6])
        T23 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
        S24 = fd.define_scalar(129, dtype=DataType.Int)
        T26 = fd.ops.expand(T23, shape=[S14, S24])
        T27 = fd.ops.sub(T22, T26)
        S28 = fd.define_scalar(1, dtype=DataType.Int)
        T29 = fd.ops.ge(T27, S28)
        S30 = fd.define_scalar(-3.38953e38, dtype=DataType.BFloat16)
        S31 = fd.define_scalar(128, dtype=DataType.Int)
        S32 = fd.define_scalar(129, dtype=DataType.Int)
        T34 = fd.ops.full(shape=[S31, S32], fill_value=S30, dtype=DataType.BFloat16)
        S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T36 = fd.ops.where(T29, T34, S35)
        fd.add_output(T18)
        fd.add_output(T36)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = []
    fd.validate(inputs)


def test_ws_tma_normalization1(nvfuser_direct_test):
    """
    Test for issue 5374765 - Gemma-7b model failure with vectorized domains.

    This test verifies complex tensor operations with BFloat16 data type,
    including reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[4096, 3072],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[4096, 3072],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[3072],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 1],
            contiguity=[None, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.ops.reshape(T0, new_shape=[1, 4096, 3072])
        T15 = fd.ops.reshape(T1, new_shape=[1, 4096, 3072])
        T16 = fd.ops.cast(T2, dtype=DataType.Float)
        T17 = fd.ops.cast(T10, dtype=DataType.Float)
        T18 = fd.ops.cast(T15, dtype=DataType.Float)
        S19 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T20 = fd.ops.add(S19, T16)
        T21 = fd.ops.add(T18, T17)
        T26 = fd.ops.broadcast_in_dim(T20, shape=[1, 4096, 3072], broadcast_dims=[2])
        T27 = fd.ops.mul(T26, T21)
        T28 = fd.ops.cast(T3, dtype=DataType.Float)
        T29 = fd.ops.mul(T28, T27)
        T30 = fd.ops.sum(T29, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T35 = fd.ops.broadcast_in_dim(T30, shape=[1, 4096, 1], broadcast_dims=[1])
        S36 = fd.define_scalar(3.00000, dtype=DataType.Float)
        T37 = fd.ops.pow(T4, S36)
        S38 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T39 = fd.ops.mul(S38, T35)
        T40 = fd.ops.mul(T39, T37)
        S41 = fd.define_scalar(3072.00, dtype=DataType.Double)
        S42 = fd.ops.reciprocal(S41)
        T43 = fd.ops.mul(T40, S42)
        T44 = fd.ops.sum(T43, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T48 = fd.ops.broadcast_in_dim(T44, shape=[1, 4096], broadcast_dims=[1])
        T53 = fd.ops.broadcast_in_dim(T48, shape=[1, 4096, 1], broadcast_dims=[0, 1])
        T58 = fd.ops.broadcast_in_dim(
            T53, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T63 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T64 = fd.ops.mul(T28, T58)
        T65 = fd.ops.mul(T63, T27)
        T66 = fd.ops.add(T65, T64)
        T67 = fd.ops.add(T66, T64)
        T68 = fd.ops.cast(T5, dtype=DataType.Float)
        T69 = fd.ops.add(T68, T67)
        T70 = fd.ops.mul(T28, T63)
        T71 = fd.ops.mul(T70, T21)
        T72 = fd.ops.sum(T71, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T73 = fd.ops.cast(T69, dtype=DataType.BFloat16)
        T74 = fd.ops.cast(T72, dtype=DataType.BFloat16)
        T78 = fd.ops.reshape(T73, new_shape=[4096, 3072])
        T79 = fd.ops.permute(T78, dims=[1, 0])
        fd.add_output(T79)
        fd.add_output(T78)
        fd.add_output(T73)
        fd.add_output(T74)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (3072,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_ws_tma_normalization2(nvfuser_direct_test):
    """
    Test for issue 5374766 - multiple model failures with circular-buffer errors.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[147456, 128],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[128],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[288, 512],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.define_tensor(
            shape=[288, 512, 128],
            contiguity=[True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[288, 512, 1],
            contiguity=[True, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T9 = fd.ops.reshape(T0, new_shape=[288, 512, 128])
        T14 = fd.ops.broadcast_in_dim(T1, shape=[288, 512, 128], broadcast_dims=[2])
        T19 = fd.ops.broadcast_in_dim(T2, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T20 = fd.ops.cast(T9, dtype=DataType.Float)
        T21 = fd.ops.cast(T14, dtype=DataType.Float)
        T26 = fd.ops.broadcast_in_dim(
            T19, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
        )
        T27 = fd.ops.cast(T3, dtype=DataType.Float)
        T28 = fd.ops.mul(T21, T20)
        T29 = fd.ops.sub(T27, T26)
        T30 = fd.ops.mul(T29, T28)
        T31 = fd.ops.sum(T30, dims=[2], keepdim=False, dtype=DataType.Null)
        T36 = fd.ops.broadcast_in_dim(T31, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T41 = fd.ops.broadcast_in_dim(
            T4, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
        )
        S42 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T43 = fd.ops.pow(T4, S42)
        S44 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T45 = fd.ops.mul(S44, T36)
        T46 = fd.ops.mul(T41, T28)
        T47 = fd.ops.mul(T45, T43)
        T48 = fd.ops.neg(T46)
        T49 = fd.ops.sum(T47, dims=[2], keepdim=False, dtype=DataType.Null)
        T50 = fd.ops.sum(T48, dims=[2], keepdim=False, dtype=DataType.Null)
        T55 = fd.ops.broadcast_in_dim(T2, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T60 = fd.ops.broadcast_in_dim(T49, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T65 = fd.ops.broadcast_in_dim(T50, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T70 = fd.ops.broadcast_in_dim(
            T55, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
        )
        T75 = fd.ops.broadcast_in_dim(
            T60, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
        )
        T76 = fd.ops.sum(T65, dims=[2], keepdim=False, dtype=DataType.Null)
        T77 = fd.ops.sub(T27, T70)
        S78 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T79 = fd.ops.mul(S78, T75)
        T84 = fd.ops.broadcast_in_dim(T76, shape=[288, 512, 1], broadcast_dims=[0, 1])
        T85 = fd.ops.mul(T79, T77)
        T90 = fd.ops.broadcast_in_dim(
            T84, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
        )
        S91 = fd.define_scalar(128.000, dtype=DataType.Double)
        S92 = fd.ops.reciprocal(S91)
        T93 = fd.ops.mul(T85, S92)
        S94 = fd.define_scalar(0.00781250, dtype=DataType.Double)
        T95 = fd.ops.mul(S94, T90)
        T96 = fd.ops.add(T95, T93)
        T97 = fd.ops.add(T46, T96)
        T98 = fd.ops.cast(T97, dtype=DataType.BFloat16)
        T99 = fd.ops.mul(T29, T41)
        T100 = fd.ops.mul(T99, T20)
        T101 = fd.ops.cast(T9, dtype=DataType.Float)
        T105 = fd.ops.reshape(T98, new_shape=[147456, 128])
        T106 = fd.ops.sum(T97, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T107 = fd.ops.sum(T100, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T108 = fd.ops.sum(T101, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T109 = fd.ops.permute(T105, dims=[1, 0])
        T110 = fd.ops.cast(T106, dtype=DataType.BFloat16)
        T111 = fd.ops.cast(T107, dtype=DataType.BFloat16)
        T112 = fd.ops.cast(T108, dtype=DataType.BFloat16)
        fd.add_output(T109)
        fd.add_output(T110)
        fd.add_output(T105)
        fd.add_output(T98)
        fd.add_output(T111)
        fd.add_output(T112)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (147456, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (128,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (288, 512),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (288, 512, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (288, 512, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_ws_tma_normalization3(nvfuser_direct_test):
    """
    Test for issue 5374767 - Mistral-7B-v0.1 failure with allocation domain errors.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[4096, 4096],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[4096],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[1, 4096, 4096],
            contiguity=[True, None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 2, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 1],
            contiguity=[None, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 4096],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T9 = fd.ops.reshape(T0, new_shape=[1, 4096, 4096])
        T10 = fd.ops.cast(T1, dtype=DataType.Float)
        T11 = fd.ops.cast(T9, dtype=DataType.Float)
        T16 = fd.ops.broadcast_in_dim(T10, shape=[1, 4096, 4096], broadcast_dims=[2])
        T17 = fd.ops.mul(T16, T11)
        T18 = fd.ops.cast(T2, dtype=DataType.Float)
        T19 = fd.ops.mul(T18, T17)
        T20 = fd.ops.sum(T19, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T25 = fd.ops.broadcast_in_dim(T20, shape=[1, 4096, 1], broadcast_dims=[1])
        S26 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T27 = fd.ops.pow(T3, S26)
        S28 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T29 = fd.ops.mul(S28, T25)
        T30 = fd.ops.mul(T29, T27)
        S31 = fd.define_scalar(4096.00, dtype=DataType.Double)
        S32 = fd.ops.reciprocal(S31)
        T33 = fd.ops.mul(T30, S32)
        T34 = fd.ops.sum(T33, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T38 = fd.ops.broadcast_in_dim(T34, shape=[1, 4096], broadcast_dims=[1])
        T43 = fd.ops.broadcast_in_dim(T38, shape=[1, 4096, 1], broadcast_dims=[0, 1])
        T48 = fd.ops.broadcast_in_dim(
            T43, shape=[1, 4096, 4096], broadcast_dims=[0, 1, 2]
        )
        T53 = fd.ops.broadcast_in_dim(
            T3, shape=[1, 4096, 4096], broadcast_dims=[0, 1, 2]
        )
        T54 = fd.ops.mul(T18, T48)
        T55 = fd.ops.mul(T53, T17)
        T56 = fd.ops.add(T55, T54)
        T57 = fd.ops.add(T56, T54)
        T58 = fd.ops.mul(T18, T53)
        T59 = fd.ops.cast(T4, dtype=DataType.Float)
        T60 = fd.ops.mul(T58, T11)
        T61 = fd.ops.add(T59, T57)
        T62 = fd.ops.sum(T60, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T63 = fd.ops.cast(T61, dtype=DataType.BFloat16)
        T64 = fd.ops.cast(T62, dtype=DataType.BFloat16)
        fd.add_output(T64)
        fd.add_output(T63)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (4096, 4096),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (4096,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 4096),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 4096),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_ws_tma_normalization4(nvfuser_direct_test):
    """
    Test for issue 5374768 - multiple model failures with boundary index errors.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[28672, 2048],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[2048],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[14, 2048, 2048],
            contiguity=[True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[14, 2048, 1],
            contiguity=[True, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T8 = fd.ops.reshape(T0, new_shape=[14, 2048, 2048])
        T9 = fd.ops.cast(T1, dtype=DataType.Float)
        T10 = fd.ops.cast(T8, dtype=DataType.Float)
        T15 = fd.ops.broadcast_in_dim(T9, shape=[14, 2048, 2048], broadcast_dims=[2])
        T16 = fd.ops.mul(T15, T10)
        T17 = fd.ops.cast(T2, dtype=DataType.Float)
        T18 = fd.ops.mul(T17, T16)
        T19 = fd.ops.sum(T18, dims=[2], keepdim=False, dtype=DataType.Null)
        T24 = fd.ops.broadcast_in_dim(T19, shape=[14, 2048, 1], broadcast_dims=[0, 1])
        S25 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T26 = fd.ops.pow(T3, S25)
        S27 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T28 = fd.ops.mul(S27, T24)
        T29 = fd.ops.mul(T28, T26)
        S30 = fd.define_scalar(2048.00, dtype=DataType.Double)
        S31 = fd.ops.reciprocal(S30)
        T32 = fd.ops.mul(T29, S31)
        T33 = fd.ops.sum(T32, dims=[2], keepdim=False, dtype=DataType.Null)
        T38 = fd.ops.broadcast_in_dim(T33, shape=[14, 2048, 1], broadcast_dims=[0, 1])
        T43 = fd.ops.broadcast_in_dim(
            T38, shape=[14, 2048, 2048], broadcast_dims=[0, 1, 2]
        )
        T48 = fd.ops.broadcast_in_dim(
            T3, shape=[14, 2048, 2048], broadcast_dims=[0, 1, 2]
        )
        T49 = fd.ops.mul(T17, T43)
        T50 = fd.ops.mul(T48, T16)
        T51 = fd.ops.add(T50, T49)
        T52 = fd.ops.add(T51, T49)
        T53 = fd.ops.cast(T52, dtype=DataType.BFloat16)
        T54 = fd.ops.mul(T17, T48)
        T55 = fd.ops.mul(T54, T10)
        T59 = fd.ops.reshape(T53, new_shape=[28672, 2048])
        T60 = fd.ops.sum(T55, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T61 = fd.ops.permute(T59, dims=[1, 0])
        T62 = fd.ops.cast(T60, dtype=DataType.BFloat16)
        fd.add_output(T61)
        fd.add_output(T59)
        fd.add_output(T53)
        fd.add_output(T62)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (28672, 2048),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (2048,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (14, 2048, 2048),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (14, 2048, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_ws_tma_normalization5(nvfuser_direct_test):
    """
    Test for issue 5374769 - stablecode-completion-alpha-3b failure with allocation domain errors.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[16384, 2560],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[2560],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[1, 16384],
            contiguity=[None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 16384, 2560],
            contiguity=[True, None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 2, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 16384, 1],
            contiguity=[None, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[16384, 2560],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T6 = fd.define_tensor(
            shape=[2560],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T7 = fd.define_tensor(
            shape=[1, 16384, 2560],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.reshape(T0, new_shape=[1, 16384, 2560])
        T17 = fd.ops.broadcast_in_dim(T1, shape=[1, 16384, 2560], broadcast_dims=[2])
        T22 = fd.ops.broadcast_in_dim(T2, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T23 = fd.ops.cast(T12, dtype=DataType.Float)
        T24 = fd.ops.cast(T17, dtype=DataType.Float)
        T29 = fd.ops.broadcast_in_dim(
            T22, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T30 = fd.ops.cast(T3, dtype=DataType.Float)
        T31 = fd.ops.mul(T24, T23)
        T32 = fd.ops.sub(T30, T29)
        T33 = fd.ops.mul(T32, T31)
        T34 = fd.ops.sum(T33, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T39 = fd.ops.broadcast_in_dim(T34, shape=[1, 16384, 1], broadcast_dims=[1])
        T44 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T49 = fd.ops.reshape(T5, new_shape=[1, 16384, 2560])
        T54 = fd.ops.broadcast_in_dim(T6, shape=[1, 16384, 2560], broadcast_dims=[2])
        S55 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T56 = fd.ops.pow(T4, S55)
        S57 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T58 = fd.ops.mul(S57, T39)
        T59 = fd.ops.mul(T44, T31)
        T60 = fd.ops.cast(T49, dtype=DataType.Float)
        T61 = fd.ops.cast(T54, dtype=DataType.Float)
        T62 = fd.ops.mul(T58, T56)
        T63 = fd.ops.neg(T59)
        T64 = fd.ops.mul(T61, T60)
        T65 = fd.ops.sum(T62, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T66 = fd.ops.sum(T63, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T67 = fd.ops.mul(T32, T64)
        T71 = fd.ops.broadcast_in_dim(T65, shape=[1, 16384], broadcast_dims=[1])
        T76 = fd.ops.broadcast_in_dim(T66, shape=[1, 16384, 1], broadcast_dims=[1])
        T77 = fd.ops.sum(T67, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T82 = fd.ops.broadcast_in_dim(T2, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T87 = fd.ops.broadcast_in_dim(T71, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T88 = fd.ops.sum(T76, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T93 = fd.ops.broadcast_in_dim(T77, shape=[1, 16384, 1], broadcast_dims=[1])
        T98 = fd.ops.broadcast_in_dim(
            T82, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T103 = fd.ops.broadcast_in_dim(
            T87, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T107 = fd.ops.broadcast_in_dim(T88, shape=[1, 16384], broadcast_dims=[1])
        S108 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T109 = fd.ops.mul(S108, T93)
        T110 = fd.ops.mul(T44, T64)
        T111 = fd.ops.sub(T30, T98)
        S112 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T113 = fd.ops.mul(S112, T103)
        T118 = fd.ops.broadcast_in_dim(T107, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T119 = fd.ops.mul(T109, T56)
        T120 = fd.ops.neg(T110)
        T121 = fd.ops.mul(T113, T111)
        T126 = fd.ops.broadcast_in_dim(
            T118, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T127 = fd.ops.sum(T119, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T128 = fd.ops.sum(T120, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        S129 = fd.define_scalar(2560.00, dtype=DataType.Double)
        S130 = fd.ops.reciprocal(S129)
        T131 = fd.ops.mul(T121, S130)
        S132 = fd.define_scalar(0.000390625, dtype=DataType.Double)
        T133 = fd.ops.mul(S132, T126)
        T134 = fd.ops.cast(T7, dtype=DataType.Float)
        T138 = fd.ops.broadcast_in_dim(T127, shape=[1, 16384], broadcast_dims=[1])
        T143 = fd.ops.broadcast_in_dim(T128, shape=[1, 16384, 1], broadcast_dims=[1])
        T144 = fd.ops.add(T133, T131)
        T145 = fd.ops.add(T134, T59)
        T150 = fd.ops.broadcast_in_dim(T138, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T151 = fd.ops.sum(T143, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T156 = fd.ops.broadcast_in_dim(
            T150, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        T160 = fd.ops.broadcast_in_dim(T151, shape=[1, 16384], broadcast_dims=[1])
        S161 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T162 = fd.ops.mul(S161, T156)
        T167 = fd.ops.broadcast_in_dim(T160, shape=[1, 16384, 1], broadcast_dims=[0, 1])
        T168 = fd.ops.add(T145, T144)
        T169 = fd.ops.mul(T162, T111)
        T174 = fd.ops.broadcast_in_dim(
            T167, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
        )
        S175 = fd.define_scalar(2560.00, dtype=DataType.Double)
        S176 = fd.ops.reciprocal(S175)
        T177 = fd.ops.mul(T169, S176)
        S178 = fd.define_scalar(0.000390625, dtype=DataType.Double)
        T179 = fd.ops.mul(S178, T174)
        T180 = fd.ops.mul(T32, T44)
        T181 = fd.ops.add(T179, T177)
        T182 = fd.ops.add(T168, T110)
        T183 = fd.ops.mul(T180, T60)
        T184 = fd.ops.mul(T180, T23)
        T185 = fd.ops.cast(T49, dtype=DataType.Float)
        T186 = fd.ops.cast(T12, dtype=DataType.Float)
        T187 = fd.ops.add(T182, T181)
        T188 = fd.ops.sum(T183, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T189 = fd.ops.sum(T185, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T190 = fd.ops.sum(T184, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T191 = fd.ops.sum(T186, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T192 = fd.ops.cast(T187, dtype=DataType.BFloat16)
        T193 = fd.ops.cast(T188, dtype=DataType.BFloat16)
        T194 = fd.ops.cast(T189, dtype=DataType.BFloat16)
        T195 = fd.ops.cast(T190, dtype=DataType.BFloat16)
        T196 = fd.ops.cast(T191, dtype=DataType.BFloat16)
        fd.add_output(T196)
        fd.add_output(T195)
        fd.add_output(T194)
        fd.add_output(T193)
        fd.add_output(T192)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (16384, 2560),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (2560,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 16384),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 16384, 2560),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 16384, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (16384, 2560),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (2560,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 16384, 2560),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_loop_promotion_cyclic_war(nvfuser_direct_test):
    """
    Test for loop promotion with cyclic WAR (Write-After-Read) dependencies.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including reshape, cast, broadcast, and mathematical operations with slice operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[4096, 128],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[4096, 128],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 4096, 5120],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 640],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 640],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 4096, 16640],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T6 = fd.define_tensor(
            shape=[1, 4096, 16640],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 4096, 5120],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T8 = fd.define_tensor(
            shape=[1, 4096, 640],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T9 = fd.define_tensor(
            shape=[1, 4096, 640],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.define_tensor(
            shape=[1, 4096, 16640],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T11 = fd.define_tensor(
            shape=[1, 4096, 16640],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T16 = fd.ops.reshape(T0, new_shape=[1, 4096, 128])
        T22 = fd.ops.broadcast_in_dim(
            T16, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T27 = fd.ops.reshape(T1, new_shape=[1, 4096, 128])
        T33 = fd.ops.broadcast_in_dim(
            T27, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T39 = fd.ops.broadcast_in_dim(
            T22, shape=[1, 40, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T40 = fd.ops.cast(T39, dtype=DataType.Float)
        T46 = fd.ops.broadcast_in_dim(
            T33, shape=[1, 40, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T47 = fd.ops.cast(T46, dtype=DataType.Float)
        T48 = fd.ops.cast(T2, dtype=DataType.Float)
        T49 = fd.ops.cast(T3, dtype=DataType.Float)
        T50 = fd.ops.cast(T4, dtype=DataType.Float)
        T51 = fd.ops.cast(T5, dtype=DataType.Float)
        T52 = fd.ops.cast(T6, dtype=DataType.Float)
        S53 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T54 = fd.ops.mul(T7, S53)
        S55 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T56 = fd.ops.mul(T8, S55)
        S57 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T58 = fd.ops.mul(T9, S57)
        S59 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T60 = fd.ops.mul(T10, S59)
        S61 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T62 = fd.ops.mul(T11, S61)
        T63 = fd.ops.add(T48, T54)
        T64 = fd.ops.add(T49, T56)
        T65 = fd.ops.add(T50, T58)
        T66 = fd.ops.add(T51, T60)
        T67 = fd.ops.add(T52, T62)
        T68 = fd.ops.cast(T63, dtype=DataType.BFloat16)
        T74 = fd.ops.reshape(T68, new_shape=[1, 4096, 40, 128])
        T75 = fd.ops.cast(T64, dtype=DataType.BFloat16)
        T81 = fd.ops.reshape(T75, new_shape=[1, 4096, 5, 128])
        T82 = fd.ops.cast(T65, dtype=DataType.BFloat16)
        T88 = fd.ops.reshape(T82, new_shape=[1, 4096, 5, 128])
        T89 = fd.ops.cast(T66, dtype=DataType.BFloat16)
        T90 = fd.ops.neg(T66)
        T91 = fd.ops.cast(T67, dtype=DataType.BFloat16)
        T92 = fd.ops.permute(T74, dims=[0, 2, 1, 3])
        T93 = fd.ops.permute(T81, dims=[0, 2, 1, 3])
        T94 = fd.ops.permute(T88, dims=[0, 2, 1, 3])
        T95 = fd.ops.exp(T90)
        T105 = fd.ops.broadcast_in_dim(
            T93, shape=[1, 1, 8, 5, 1, 4096, 1, 128], broadcast_dims=[1, 3, 5, 7]
        )
        T111 = fd.ops.reshape(T105, new_shape=[1, 40, 4096, 128])
        T121 = fd.ops.broadcast_in_dim(
            T94, shape=[1, 1, 8, 5, 1, 4096, 1, 128], broadcast_dims=[1, 3, 5, 7]
        )
        T127 = fd.ops.reshape(T121, new_shape=[1, 40, 4096, 128])
        T128 = fd.ops.cast(T92, dtype=DataType.Float)
        T144 = fd.ops.slice(
            T92,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 40, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T160 = fd.ops.slice(
            T92,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 40, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T161 = fd.ops.cast(T160, dtype=DataType.Float)
        T162 = fd.ops.neg(T161)
        T163 = fd.ops.cast(T162, dtype=DataType.BFloat16)
        T164 = fd.ops.cast(T111, dtype=DataType.Float)
        T180 = fd.ops.slice(
            T111,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 40, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T196 = fd.ops.slice(
            T111,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 40, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T197 = fd.ops.cast(T196, dtype=DataType.Float)
        T198 = fd.ops.neg(T197)
        T199 = fd.ops.cast(T198, dtype=DataType.BFloat16)
        S200 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T201 = fd.ops.add(S200, T95)
        T202 = fd.ops.mul(T128, T40)
        T203 = fd.ops.cat([T163, T144], dim=-1, manual_padding=0)
        T204 = fd.ops.mul(T164, T40)
        T205 = fd.ops.cat([T199, T180], dim=-1, manual_padding=0)
        T206 = fd.ops.reciprocal(T201)
        T207 = fd.ops.cast(T203, dtype=DataType.Float)
        T208 = fd.ops.cast(T205, dtype=DataType.Float)
        T209 = fd.ops.mul(T207, T47)
        T210 = fd.ops.mul(T208, T47)
        T211 = fd.ops.mul(T66, T206)
        T212 = fd.ops.add(T202, T209)
        T213 = fd.ops.add(T204, T210)
        T214 = fd.ops.mul(T211, T67)
        T215 = fd.ops.cast(T212, dtype=DataType.BFloat16)
        T216 = fd.ops.cast(T213, dtype=DataType.BFloat16)
        T217 = fd.ops.cast(T214, dtype=DataType.BFloat16)
        fd.add_output(T89)
        fd.add_output(T91)
        fd.add_output(T127)
        fd.add_output(T215)
        fd.add_output(T216)
        fd.add_output(T217)
        fd.add_output(T214)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (4096, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (4096, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 5120),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 640),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 640),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 16640),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 16640),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 5120),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 640),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 640),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 16640),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 16640),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_reshape_cancellation(nvfuser_direct_test):
    """
    Test for reshape cancellation operations with complex tensor manipulations.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including slice operations, concatenation, reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 2048, 24, 32],
            contiguity=[None, True, True, False],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 2048, 24, 32],
            contiguity=[None, True, True, False],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 2048, 24, 32],
            contiguity=[None, True, True, False],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 2048, 4, 4608],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 2048, 24, 32],
            contiguity=[None, True, True, False],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 2048, 24, 64],
            contiguity=[None, True, None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T6 = fd.define_tensor(
            shape=[1, 2048, 24, 64],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 2048, 24, 64],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T8 = fd.ops.cast(T0, dtype=DataType.Float)
        T9 = fd.ops.neg(T8)
        T10 = fd.ops.cast(T9, dtype=DataType.BFloat16)
        T17 = fd.ops.broadcast_in_dim(
            T1, shape=[1, 2048, 24, 32, 1], broadcast_dims=[0, 1, 2, 3]
        )
        T24 = fd.ops.broadcast_in_dim(
            T10, shape=[1, 2048, 24, 32, 1], broadcast_dims=[0, 1, 2, 3]
        )
        T25 = fd.ops.cast(T2, dtype=DataType.Float)
        T41 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 3072],
            end_indices=[1, 2048, 4, 4608],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T42 = fd.ops.cat([T24, T17], dim=-1, manual_padding=0)
        T43 = fd.ops.neg(T25)
        T50 = fd.ops.reshape(T41, new_shape=[1, 2048, 4, 6, 256])
        T56 = fd.ops.reshape(T42, new_shape=[1, 2048, 24, 64])
        T57 = fd.ops.cast(T43, dtype=DataType.BFloat16)
        T63 = fd.ops.reshape(T50, new_shape=[1, 2048, 24, 256])
        T64 = fd.ops.cast(T56, dtype=DataType.Float)
        T71 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 2048, 24, 32, 1], broadcast_dims=[0, 1, 2, 3]
        )
        T78 = fd.ops.broadcast_in_dim(
            T57, shape=[1, 2048, 24, 32, 1], broadcast_dims=[0, 1, 2, 3]
        )
        T94 = fd.ops.slice(
            T63,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 2048, 24, 256],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T95 = fd.ops.mul(T64, T5)
        T111 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 2048, 4, 1536],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T112 = fd.ops.cat([T78, T71], dim=-1, manual_padding=0)
        T113 = fd.ops.cast(T94, dtype=DataType.Float)
        T114 = fd.ops.add(T6, T95)
        T121 = fd.ops.reshape(T111, new_shape=[1, 2048, 4, 6, 256])
        T127 = fd.ops.reshape(T112, new_shape=[1, 2048, 24, 64])
        T128 = fd.ops.cat([T114, T113], dim=-1, manual_padding=0)
        T134 = fd.ops.reshape(T121, new_shape=[1, 2048, 24, 256])
        T135 = fd.ops.cast(T127, dtype=DataType.Float)
        T136 = fd.ops.permute(T128, dims=[0, 2, 1, 3])
        T152 = fd.ops.slice(
            T134,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 2048, 24, 256],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T153 = fd.ops.mul(T135, T5)
        T154 = fd.ops.cast(T136, dtype=DataType.BFloat16)
        T155 = fd.ops.cast(T152, dtype=DataType.Float)
        T156 = fd.ops.add(T7, T153)
        T157 = fd.ops.cat([T156, T155], dim=-1, manual_padding=0)
        T158 = fd.ops.permute(T136, dims=[0, 1, 3, 2])
        T159 = fd.ops.permute(T157, dims=[0, 2, 1, 3])
        fd.add_output(T159)
        fd.add_output(T154)
        fd.add_output(T158)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.randn(3145727, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 24, 32), (3145728, 1536, 64, 2)
        ),
        torch.randn(3145727, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 24, 32), (3145728, 1536, 64, 2)
        ),
        torch.randn(3145727, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 24, 32), (3145728, 1536, 64, 2)
        ),
        torch.testing.make_tensor(
            (1, 2048, 4, 4608),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.randn(3145727, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 24, 32), (3145728, 1536, 64, 2)
        ),
        torch.randn(131072, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 24, 64), (131072, 64, 0, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 24, 64),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 2048, 24, 64),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_reduction_reference_missing_input_ids(nvfuser_direct_test):
    """
    Test for issue 4840 - reduction reference missing input IDs.

    This test verifies complex tensor operations with Half and Float data types,
    including slice operations, concatenation, reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 16, 4096, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 4096, 16, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 16, 4096, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 16, 128],
            contiguity=[None, True, None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 6144],
            contiguity=[None, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 16, 4096, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T6 = fd.define_tensor(
            shape=[1, 4096, 16, 128],
            contiguity=[None, True, None, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 4096, 16, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.Half,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T8 = fd.ops.permute(T0, dims=[0, 2, 1, 3])
        T9 = fd.ops.cast(T8, dtype=DataType.Float)
        T10 = fd.ops.cast(T1, dtype=DataType.Float)
        T11 = fd.ops.add(T10, T9)
        T12 = fd.ops.permute(T2, dims=[0, 2, 1, 3])
        T13 = fd.ops.cast(T12, dtype=DataType.Float)
        T29 = fd.ops.slice(
            T11,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4096, 16, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T45 = fd.ops.slice(
            T13,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4096, 16, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T46 = fd.ops.mul(T3, T29)
        T47 = fd.ops.mul(T3, T45)
        T63 = fd.ops.slice(
            T46,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4096, 16, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T79 = fd.ops.slice(
            T47,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4096, 16, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T95 = fd.ops.slice(
            T46,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4096, 16, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T96 = fd.ops.neg(T63)
        T112 = fd.ops.slice(
            T47,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4096, 16, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T113 = fd.ops.neg(T79)
        T120 = fd.ops.broadcast_in_dim(
            T95, shape=[1, 4096, 16, 1, 64], broadcast_dims=[0, 1, 2, 4]
        )
        T127 = fd.ops.broadcast_in_dim(
            T96, shape=[1, 4096, 16, 1, 64], broadcast_dims=[0, 1, 2, 4]
        )
        T134 = fd.ops.broadcast_in_dim(
            T112, shape=[1, 4096, 16, 1, 64], broadcast_dims=[0, 1, 2, 4]
        )
        T141 = fd.ops.broadcast_in_dim(
            T113, shape=[1, 4096, 16, 1, 64], broadcast_dims=[0, 1, 2, 4]
        )
        S142 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T154 = fd.ops.pad(T120, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], S142)
        S155 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T167 = fd.ops.pad(T127, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], S155)
        S168 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T180 = fd.ops.pad(T134, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], S168)
        S181 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T193 = fd.ops.pad(T141, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], S181)
        T206 = fd.ops.slice(
            T4,
            start_indices=[0, 0, 0],
            end_indices=[1, 4096, 2048],
            strides=[1, 1, 1],
            manual_normalization=0,
        )
        T219 = fd.ops.slice(
            T4,
            start_indices=[0, 0, 2048],
            end_indices=[1, 4096, 4096],
            strides=[1, 1, 1],
            manual_normalization=0,
        )
        T220 = fd.ops.add(T167, T154)
        T221 = fd.ops.add(T193, T180)
        T227 = fd.ops.reshape(T206, new_shape=[1, 4096, 16, 128])
        T233 = fd.ops.reshape(T219, new_shape=[1, 4096, 16, 128])
        T234 = fd.ops.permute(T5, dims=[0, 2, 1, 3])
        S235 = fd.define_scalar(0, dtype=DataType.Int)
        T241 = fd.ops.full(
            shape=[1, 4096, 16, 0], fill_value=S235, dtype=DataType.Float
        )
        T242 = fd.ops.mul(T6, T29)
        T248 = fd.ops.reshape(T220, new_shape=[1, 4096, 16, 128])
        T249 = fd.ops.mul(T6, T45)
        T255 = fd.ops.reshape(T221, new_shape=[1, 4096, 16, 128])
        T256 = fd.ops.cast(T227, dtype=DataType.Float)
        T257 = fd.ops.cast(T233, dtype=DataType.Float)
        T258 = fd.ops.cast(T234, dtype=DataType.Float)
        T259 = fd.ops.cast(T7, dtype=DataType.Float)
        S260 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T270 = fd.ops.pad(T241, [0, 128, 0, 0, 0, 0, 0, 0], S260)
        T271 = fd.ops.add(T248, T242)
        T272 = fd.ops.add(T255, T249)
        T279 = fd.ops.reshape(T256, new_shape=[1, 4096, 16, 2, 64])
        T286 = fd.ops.reshape(T257, new_shape=[1, 4096, 16, 2, 64])
        T287 = fd.ops.add(T259, T258)
        T288 = fd.ops.add(T271, T270)
        T289 = fd.ops.add(T272, T270)
        T308 = fd.ops.slice(
            T279,
            start_indices=[0, 0, 0, 1, 0],
            end_indices=[1, 4096, 16, 2, 64],
            strides=[1, 1, 1, 1, 1],
            manual_normalization=0,
        )
        T327 = fd.ops.slice(
            T286,
            start_indices=[0, 0, 0, 1, 0],
            end_indices=[1, 4096, 16, 2, 64],
            strides=[1, 1, 1, 1, 1],
            manual_normalization=0,
        )
        T328 = fd.ops.cast(T287, dtype=DataType.Half)
        T329 = fd.ops.cast(T288, dtype=DataType.Half)
        T330 = fd.ops.cast(T289, dtype=DataType.Half)
        T349 = fd.ops.slice(
            T279,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[1, 4096, 16, 1, 64],
            strides=[1, 1, 1, 1, 1],
            manual_normalization=0,
        )
        T350 = fd.ops.squeeze(T308, dims=[3], squeeze_expanded=False)
        T369 = fd.ops.slice(
            T286,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[1, 4096, 16, 1, 64],
            strides=[1, 1, 1, 1, 1],
            manual_normalization=0,
        )
        T370 = fd.ops.squeeze(T327, dims=[3], squeeze_expanded=False)
        T375 = fd.ops.reshape(T328, new_shape=[1, 4096, 2048])
        T380 = fd.ops.reshape(T329, new_shape=[1, 4096, 2048])
        T385 = fd.ops.reshape(T330, new_shape=[1, 4096, 2048])
        T386 = fd.ops.squeeze(T349, dims=[3], squeeze_expanded=False)
        T387 = fd.ops.neg(T350)
        T388 = fd.ops.squeeze(T369, dims=[3], squeeze_expanded=False)
        T389 = fd.ops.neg(T370)
        T390 = fd.ops.cat([T385, T380, T375], dim=2, manual_padding=0)
        T391 = fd.ops.cat([T387, T386], dim=-1, manual_padding=0)
        T392 = fd.ops.cat([T389, T388], dim=-1, manual_padding=0)
        T393 = fd.ops.cast(T390, dtype=DataType.Float)
        T394 = fd.ops.mul(T256, T45)
        T395 = fd.ops.mul(T257, T29)
        T396 = fd.ops.mul(T391, T45)
        T397 = fd.ops.mul(T392, T29)
        S398 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T399 = fd.ops.mul(S398, T393)
        T400 = fd.ops.sum(T394, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T401 = fd.ops.sum(T395, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T402 = fd.ops.sum(T396, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T403 = fd.ops.sum(T397, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T407 = fd.ops.reshape(T399, new_shape=[4096, 6144])
        T413 = fd.ops.broadcast_in_dim(
            T400, shape=[1, 4096, 1, 128], broadcast_dims=[1, 3]
        )
        T419 = fd.ops.broadcast_in_dim(
            T401, shape=[1, 4096, 1, 128], broadcast_dims=[1, 3]
        )
        T425 = fd.ops.broadcast_in_dim(
            T402, shape=[1, 4096, 1, 128], broadcast_dims=[1, 3]
        )
        T431 = fd.ops.broadcast_in_dim(
            T403, shape=[1, 4096, 1, 128], broadcast_dims=[1, 3]
        )
        T432 = fd.ops.permute(T407, dims=[1, 0])
        T436 = fd.ops.reshape(T390, new_shape=[4096, 6144])
        T437 = fd.ops.add(T419, T413)
        T438 = fd.ops.add(T431, T425)
        fd.add_output(T432)
        fd.add_output(T407)
        fd.add_output(T436)
        fd.add_output(T437)
        fd.add_output(T438)

    with FusionDefinition() as fd:
        fusion_func(fd)

    # input range is revised to [-1, 1] to avoid small validation errors
    # which highly likely caused by the strict tolerance values in
    # ValidationConstants.
    inputs = [
        torch.testing.make_tensor(
            (8388608,), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ).as_strided((1, 16, 4096, 128), (8388608, 128, 2048, 1)),
        torch.testing.make_tensor(
            (1, 4096, 16, 128), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ),
        torch.testing.make_tensor(
            (8388608,), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ).as_strided((1, 16, 4096, 128), (8388608, 128, 2048, 1)),
        torch.testing.make_tensor(
            (524288,), dtype=torch.float32, device="cuda:0", low=-1, high=1
        ).as_strided((1, 4096, 16, 128), (1048576, 128, 0, 1)),
        torch.testing.make_tensor(
            (1, 4096, 6144), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ),
        torch.testing.make_tensor(
            (8388608,), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ).as_strided((1, 16, 4096, 128), (8388608, 128, 2048, 1)),
        torch.testing.make_tensor(
            (524288,), dtype=torch.float32, device="cuda:0", low=-1, high=1
        ).as_strided((1, 4096, 16, 128), (1048576, 128, 0, 1)),
        torch.testing.make_tensor(
            (1, 4096, 16, 128), dtype=torch.float16, device="cuda:0", low=-1, high=1
        ),
    ]
    fd.validate(inputs)


def test_ws_tma_normalization6(nvfuser_direct_test):
    """
    Test for scalar input handling in TMA normalization.

    This test verifies complex tensor operations with BFloat16 and Float data types,
    including scalar tensor operations, reshape, cast, broadcast, and mathematical operations.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[3072],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T1 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 1],
            contiguity=[None, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False
        )
        T6 = fd.ops.cast(T0, dtype=DataType.Float)
        S7 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T8 = fd.ops.add(S7, T6)
        T9 = fd.ops.cast(T1, dtype=DataType.Float)
        T14 = fd.ops.broadcast_in_dim(T8, shape=[1, 4096, 3072], broadcast_dims=[2])
        T15 = fd.ops.mul(T14, T9)
        T16 = fd.ops.cast(T2, dtype=DataType.Float)
        T17 = fd.ops.mul(T16, T15)
        T18 = fd.ops.sum(T17, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T23 = fd.ops.broadcast_in_dim(T18, shape=[1, 4096, 1], broadcast_dims=[1])
        S24 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T25 = fd.ops.pow(T3, S24)
        S26 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T27 = fd.ops.mul(S26, T23)
        T28 = fd.ops.mul(T27, T25)
        S29 = fd.define_scalar(3072.00, dtype=DataType.Double)
        S30 = fd.ops.reciprocal(S29)
        T31 = fd.ops.mul(T28, S30)
        T32 = fd.ops.sum(T31, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T36 = fd.ops.broadcast_in_dim(T32, shape=[1, 4096], broadcast_dims=[1])
        T41 = fd.ops.broadcast_in_dim(T36, shape=[1, 4096, 1], broadcast_dims=[0, 1])
        T46 = fd.ops.broadcast_in_dim(
            T41, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T51 = fd.ops.broadcast_in_dim(
            T3, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T52 = fd.ops.mul(T16, T46)
        T53 = fd.ops.mul(T51, T15)
        T54 = fd.ops.add(T53, T52)
        T55 = fd.ops.add(T54, T52)
        T56 = fd.ops.cast(T4, dtype=DataType.Float)
        T57 = fd.ops.mul(T16, T51)
        T58 = fd.ops.add(T56, T55)
        T59 = fd.ops.mul(T57, T9)
        T60 = fd.ops.sum(T59, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T61 = fd.ops.cast(T60, dtype=DataType.BFloat16)
        T62 = fd.ops.mul(T5, T58)
        T63 = fd.ops.cast(T62, dtype=DataType.BFloat16)
        fd.add_output(T61)
        fd.add_output(T63)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor(
            (3072,),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 1),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (1, 4096, 3072),
            dtype=torch.bfloat16,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
        torch.testing.make_tensor(
            (),
            dtype=torch.float32,
            device="cuda:0",
            low=LOW_VAL,
            high=HIGH_VAL,
        ),
    ]
    fd.validate(inputs)


def test_domain_map_hang(nvfuser_direct_test):
    """
    Test for issue 4960 - domain map hang in complex tensor operations.

    This test verifies complex tensor operations with Float and BFloat16 data types,
    including iota operations, broadcast, mathematical operations, index_select, reshape, cast, slice operations, and concatenation.
    """

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[16],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T1 = fd.define_tensor(
            shape=[16],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[4096, 4096],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T6 = fd.define_tensor(
            shape=[1, 4096, 10240],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T8 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T9 = fd.define_tensor(
            shape=[1, 4096, 2560],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.define_tensor(
            shape=[1, 4096, 10240],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        S11 = fd.define_scalar(4096, dtype=DataType.Int)
        S12 = fd.define_scalar(0, dtype=DataType.Int)
        S13 = fd.define_scalar(1, dtype=DataType.Int)
        T14 = fd.ops.iota(S11, S12, S13, dtype=DataType.Int)
        T18 = fd.ops.broadcast_in_dim(T14, shape=[1, 4096], broadcast_dims=[1])
        T19 = fd.ops.cast(T14, dtype=DataType.Float)
        T23 = fd.ops.broadcast_in_dim(T19, shape=[4096, 1], broadcast_dims=[0])
        T27 = fd.ops.broadcast_in_dim(T0, shape=[1, 16], broadcast_dims=[1])
        T31 = fd.ops.broadcast_in_dim(T23, shape=[4096, 16], broadcast_dims=[0, 1])
        T35 = fd.ops.broadcast_in_dim(T27, shape=[4096, 16], broadcast_dims=[0, 1])
        T39 = fd.ops.broadcast_in_dim(T1, shape=[1, 16], broadcast_dims=[1])
        T43 = fd.ops.broadcast_in_dim(T39, shape=[4096, 16], broadcast_dims=[0, 1])
        S44 = fd.define_scalar(0, dtype=DataType.Int)
        T45 = fd.ops.lt(T18, S44)
        T46 = fd.ops.mul(T31, T35)
        T47 = fd.ops.mul(T31, T43)
        S48 = fd.define_scalar(4096, dtype=DataType.Int)
        S49 = fd.define_scalar(0, dtype=DataType.Int)
        T50 = fd.ops.where(T45, S48, S49)
        T51 = fd.ops.cast(T50, dtype=DataType.Int)
        T52 = fd.ops.cat([T46, T46], dim=-1, manual_padding=0)
        T53 = fd.ops.cat([T47, T47], dim=-1, manual_padding=0)
        T54 = fd.ops.add(T18, T51)
        T55 = fd.ops.cos(T52)
        T56 = fd.ops.sin(T52)
        T57 = fd.ops.cos(T53)
        T58 = fd.ops.sin(T53)
        T59 = fd.ops.cast(T55, dtype=DataType.BFloat16)
        T60 = fd.ops.cast(T56, dtype=DataType.BFloat16)
        T63 = fd.ops.reshape(T54, new_shape=[4096])
        T64 = fd.ops.cast(T57, dtype=DataType.BFloat16)
        T65 = fd.ops.cast(T58, dtype=DataType.BFloat16)
        T66 = fd.ops.index_select(T59, T63, dim=0)
        T67 = fd.ops.index_select(T60, T63, dim=0)
        T68 = fd.ops.index_select(T64, T63, dim=0)
        T69 = fd.ops.index_select(T65, T63, dim=0)
        T74 = fd.ops.reshape(T66, new_shape=[1, 4096, 32])
        T80 = fd.ops.broadcast_in_dim(
            T74, shape=[1, 1, 4096, 32], broadcast_dims=[0, 2, 3]
        )
        T85 = fd.ops.reshape(T67, new_shape=[1, 4096, 32])
        T91 = fd.ops.broadcast_in_dim(
            T85, shape=[1, 1, 4096, 32], broadcast_dims=[0, 2, 3]
        )
        T97 = fd.ops.broadcast_in_dim(
            T80, shape=[1, 32, 4096, 32], broadcast_dims=[0, 1, 2, 3]
        )
        T98 = fd.ops.cast(T97, dtype=DataType.Float)
        T104 = fd.ops.broadcast_in_dim(
            T91, shape=[1, 32, 4096, 32], broadcast_dims=[0, 1, 2, 3]
        )
        T105 = fd.ops.cast(T104, dtype=DataType.Float)
        T110 = fd.ops.reshape(T68, new_shape=[1, 4096, 32])
        T116 = fd.ops.broadcast_in_dim(
            T110, shape=[1, 1, 4096, 32], broadcast_dims=[0, 2, 3]
        )
        T121 = fd.ops.reshape(T69, new_shape=[1, 4096, 32])
        T127 = fd.ops.broadcast_in_dim(
            T121, shape=[1, 1, 4096, 32], broadcast_dims=[0, 2, 3]
        )
        T133 = fd.ops.broadcast_in_dim(
            T116, shape=[1, 32, 4096, 32], broadcast_dims=[0, 1, 2, 3]
        )
        T134 = fd.ops.cast(T133, dtype=DataType.Float)
        T140 = fd.ops.broadcast_in_dim(
            T127, shape=[1, 32, 4096, 32], broadcast_dims=[0, 1, 2, 3]
        )
        T141 = fd.ops.cast(T140, dtype=DataType.Float)
        T142 = fd.ops.cast(T2, dtype=DataType.BFloat16)
        T143 = fd.ops.cast(T3, dtype=DataType.Float)
        T144 = fd.ops.cast(T4, dtype=DataType.Float)
        T145 = fd.ops.cast(T5, dtype=DataType.Float)
        T146 = fd.ops.cast(T6, dtype=DataType.Float)
        S147 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T148 = fd.ops.mul(T7, S147)
        S149 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T150 = fd.ops.mul(T8, S149)
        S151 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T152 = fd.ops.mul(T9, S151)
        S153 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T154 = fd.ops.mul(T10, S153)
        T155 = fd.ops.add(T143, T148)
        T156 = fd.ops.add(T144, T150)
        T157 = fd.ops.add(T145, T152)
        T158 = fd.ops.add(T146, T154)
        T159 = fd.ops.cast(T155, dtype=DataType.BFloat16)
        T160 = fd.ops.cast(T156, dtype=DataType.BFloat16)
        T161 = fd.ops.cast(T157, dtype=DataType.BFloat16)
        T167 = fd.ops.reshape(T159, new_shape=[1, 4096, 32, 80])
        T173 = fd.ops.reshape(T160, new_shape=[1, 4096, 32, 80])
        T179 = fd.ops.reshape(T161, new_shape=[1, 4096, 32, 80])
        T180 = fd.ops.cast(T158, dtype=DataType.BFloat16)
        T181 = fd.ops.permute(T167, dims=[0, 2, 1, 3])
        T182 = fd.ops.permute(T173, dims=[0, 2, 1, 3])
        T183 = fd.ops.permute(T179, dims=[0, 2, 1, 3])
        S184 = fd.define_scalar(0.500000, dtype=DataType.Double)
        T185 = fd.ops.mul(S184, T158)
        S186 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T187 = fd.ops.pow(T158, S186)
        T203 = fd.ops.slice(
            T181,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 32, 4096, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T219 = fd.ops.slice(
            T181,
            start_indices=[0, 0, 0, 32],
            end_indices=[1, 32, 4096, 80],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T235 = fd.ops.slice(
            T182,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 32, 4096, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T251 = fd.ops.slice(
            T182,
            start_indices=[0, 0, 0, 32],
            end_indices=[1, 32, 4096, 80],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T252 = fd.ops.cast(T203, dtype=DataType.Float)
        T268 = fd.ops.slice(
            T203,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 32, 4096, 16],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T284 = fd.ops.slice(
            T203,
            start_indices=[0, 0, 0, 16],
            end_indices=[1, 32, 4096, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T285 = fd.ops.cast(T284, dtype=DataType.Float)
        T286 = fd.ops.neg(T285)
        T287 = fd.ops.cast(T286, dtype=DataType.BFloat16)
        T288 = fd.ops.cast(T235, dtype=DataType.Float)
        T304 = fd.ops.slice(
            T235,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 32, 4096, 16],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T320 = fd.ops.slice(
            T235,
            start_indices=[0, 0, 0, 16],
            end_indices=[1, 32, 4096, 32],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T321 = fd.ops.cast(T320, dtype=DataType.Float)
        T322 = fd.ops.neg(T321)
        T323 = fd.ops.cast(T322, dtype=DataType.BFloat16)
        T324 = fd.ops.mul(T252, T98)
        T325 = fd.ops.cat([T287, T268], dim=-1, manual_padding=0)
        T326 = fd.ops.mul(T288, T98)
        T327 = fd.ops.cat([T323, T304], dim=-1, manual_padding=0)
        S328 = fd.define_scalar(0.0447150, dtype=DataType.Double)
        T329 = fd.ops.mul(S328, T187)
        T330 = fd.ops.cast(T325, dtype=DataType.Float)
        T331 = fd.ops.cast(T327, dtype=DataType.Float)
        T332 = fd.ops.mul(T330, T105)
        T333 = fd.ops.mul(T331, T105)
        T334 = fd.ops.add(T158, T329)
        T335 = fd.ops.add(T324, T332)
        T336 = fd.ops.add(T326, T333)
        S337 = fd.define_scalar(0.797885, dtype=DataType.Double)
        T338 = fd.ops.mul(S337, T334)
        T339 = fd.ops.cast(T335, dtype=DataType.BFloat16)
        T340 = fd.ops.cast(T336, dtype=DataType.BFloat16)
        T341 = fd.ops.cat([T339, T219], dim=-1, manual_padding=0)
        T342 = fd.ops.cat([T340, T251], dim=-1, manual_padding=0)
        T343 = fd.ops.tanh(T338)
        T344 = fd.ops.cast(T341, dtype=DataType.Float)
        T345 = fd.ops.cast(T342, dtype=DataType.Float)
        T346 = fd.ops.permute(T345, dims=[0, 1, 3, 2])
        S347 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T348 = fd.ops.add(S347, T343)
        T349 = fd.ops.mul(T185, T348)
        T350 = fd.ops.cast(T349, dtype=DataType.BFloat16)
        fd.add_output(T59)
        fd.add_output(T60)
        fd.add_output(T64)
        fd.add_output(T65)
        fd.add_output(T80)
        fd.add_output(T91)
        fd.add_output(T134)
        fd.add_output(T141)
        fd.add_output(T142)
        fd.add_output(T180)
        fd.add_output(T183)
        fd.add_output(T342)
        fd.add_output(T344)
        fd.add_output(T346)
        fd.add_output(T350)
        fd.add_output(T349)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.testing.make_tensor((16,), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((16,), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((4096, 4096), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 10240), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 2560), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 4096, 10240), dtype=torch.float32, device="cuda:0"
        ),
    ]
    fd.validate(inputs)

    # https://github.com/NVIDIA/Fuser/issues/3290
    def test_execution_order(self):
        N_PARALLEL_PATHS = 10

        with FusionDefinition() as fd:
            T0s = [
                fd.define_tensor(
                    shape=[256, 256],
                    contiguity=[True, True],
                    dtype=DataType.Float,
                    is_cpu=False,
                    stride_order=[1, 0],
                )
                for _ in range(N_PARALLEL_PATHS)
            ]
            a = fd.define_tensor(
                shape=[256, 256],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            for T0 in T0s:
                T1 = fd.ops.relu(T0)
                T2 = fd.ops.matmul(T1, T1)
                T3 = fd.ops.relu(T2)
                a = fd.ops.matmul(T3, a)
            fd.add_output(a)

        t0s = [
            torch.randn(256, 256, device="cuda") for _ in range(N_PARALLEL_PATHS)
        ]  # 0.25 MiB * N_PARALLEL_PATHS
        a = torch.randn(256, 256, device="cuda")  # 0.25 MiB

        # Warm up
        fd.execute([*t0s, a])

        with RecordTorchMemory() as nvf_mem:
            fd.execute([*t0s, a])

        def eager_func(t0s, a):
            for t0 in t0s:
                t1 = torch.nn.functional.relu(t0)
                del t0
                t2 = torch.matmul(t1, t1)
                del t1
                t3 = torch.nn.functional.relu(t2)
                del t2
                a = torch.matmul(t3, a)
                del t3
            return a

        with RecordTorchMemory() as eager_mem:
            eager_func(t0s, a)

        assert nvf_mem == eager_mem
