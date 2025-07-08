# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
import torch._refs as refs
import torch._prims as prims
from functools import partial

from typing import List
import itertools

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

from nvfuser_direct.testing.utils import (
    is_pre_ampere,
    is_pre_volta,
)

from utils import (
    NVFuserTest,
)


@pytest.mark.skipif(is_pre_volta(), reason="Only supported on Volta and newer devices.")
class TestNvFuserFrontend(NVFuserTest):
    def test_basic(self):
        inputs = [
            torch.ones(2, 4, 8, device="cuda"),
            torch.ones(2, 4, 8, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
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
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

    def test_basic_fp16(self):
        inputs = [
            torch.ones(2, 4, 8, device="cuda", dtype=torch.float16),
            torch.ones(2, 4, 8, device="cuda", dtype=torch.float16),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            c0 = fd.define_scalar(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            t5 = fd.ops.cast(t4, DataType.Half)
            fd.add_output(t5)

        # t0 and t1 are ones(2, 4, 8) tensors.
        # t2 = t0 + t1 = twos(2, 4, 8)
        # t3 = t2 * 3.0 = sixes(2,4,8)
        # t4 = sum(t3, dim=-1) = forty-eights(2, 4)
        # t5 = cast(t4, DataType.Half) = forty-eights(2, 4)
        # Expected Output is a tensor of 48's
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

    def test_cast_double_to_half(self):
        inputs = [
            torch.randn(2, 4, device="cuda", dtype=torch.float64),
            torch.randn(2, 4, device="cuda", dtype=torch.float64),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0h = fd.ops.cast(t0, DataType.Half)
            t1h = fd.ops.cast(t1, DataType.Half)
            t2 = fd.ops.add(t0h, t1h)
            t3 = fd.ops.relu(t2)
            t4 = fd.ops.cast(t3, DataType.Half)

            fd.add_output(t4)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0].to(torch.half) + inputs[1].to(torch.half))
        self.assertEqual(eager_out, nvf_out[0])

    # TODO Add test_cast_fp8

    def test_promote_to_double(self):
        inputs = [
            torch.randn(2, 4, device="cuda", dtype=torch.float16),
            torch.randn(2, 4, device="cuda", dtype=torch.float64),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t2 = fd.ops.add(t0, t1)
            t5 = fd.ops.relu(t2)

            fd.add_output(t5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0] + inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_matmul(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.matmul(inputs[0], inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_linear_with_bias(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.nn.functional.linear(inputs[0], inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_linear_without_bias(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.nn.functional.linear(inputs[0], inputs[1], inputs[2])
        self.assertEqual(eager_out, nvf_out[0])

    def test_tensor_ndim(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum(inputs[0].reshape(new_shape), dim=3)
        self.assertEqual(eager_out, nvf_out[0])

    def test_execute_with_tuple_and_list(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs_with_list)
        self.assertEqual(eager_out, nvf_out[0])

        inputs_with_tuple = [tensor, tuple(new_shape)]
        # expect to reuse fusion
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs_with_tuple)
        self.assertEqual(eager_out, nvf_out[0])

    def test_dynamic_reshape(self):
        def dynamic_reshape(fd: FusionDefinition) -> None:
            x = fd.define_tensor([-1, -1], [True, True])
            d0 = fd.ops.size(x, 0)
            d1 = fd.define_scalar(dtype=DataType.Int32)
            d2 = fd.define_scalar(dtype=DataType.Int32)
            y = fd.ops.reshape(x, [d0, d1, d2])
            fd.add_output(y)

        x = torch.rand(3, 4, device="cuda")
        ys, _ = self.exec_nvfuser(dynamic_reshape, [x, 2, 2])
        self.assertEqual(len(ys), 1)
        y = ys[0]

        self.assertEqual(y.shape, torch.Size([3, 2, 2]))
        self.assertEqual(x.flatten(), y.flatten())

    def test_reshape_dynamic(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    # Test empty symbolic tensors can be reshaped
    # See https://github.com/NVIDIA/Fuser/issues/2362
    def test_empty_reshape(self):
        inputs = [
            torch.randint(0, 10, (0, 1, 2, 3, 4), dtype=torch.int64, device="cuda:0")
        ]

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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    def test_squeeze(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        v1 = torch.sum(inputs[1], [0, -1])
        v2 = torch.sum(inputs[2], [0, 1])
        eager_out = inputs[0] * v1 * v2
        self.assertEqual(eager_out, nvf_out[0])

    def test_expand(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = inputs[0].expand(inputs[1].size()) + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_broadcast(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1]
        )
        self.assertEqual(eager_out, nvf_out[0])

    def test_implicit_broadcast_input(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1]
        )
        self.assertEqual(eager_out, nvf_out[0])

    def test_explicit_broadcast_input(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            prims.broadcast_in_dim(inputs[0], inputs[1].size(), [0, 1, 2]), inputs[1]
        )
        self.assertEqual(eager_out, nvf_out[0])

    def test_broadcast_mixing(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], [3, 3], [0]))
        self.assertEqual(eager_out, nvf_out[0])

    def test_prim_layer_norm_fwd(self):
        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = "cuda"
        inputs = [
            torch.randn(*input_size, device=device, requires_grad=True),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
        ]

        def primitive_definition(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            normalization_axis: int,
            keepdim: bool,
        ) -> torch.Tensor:
            mean = inputs.mean(normalization_axis, keepdim=keepdim)
            diff = inputs - mean
            diff_sq = diff * diff
            var = diff_sq.mean(normalization_axis, keepdim=keepdim)
            pre_shift_scale_norm_output = (inputs - mean) / torch.sqrt(var + 1e-12)
            norm_output = weight * pre_shift_scale_norm_output + bias
            return norm_output

        def nvfuser_fusion(
            fd: FusionDefinition,
            normalization_axis: int,
            norm_size: int,
            input_shape: List[int],
            eps: float,
            keepDim: bool,
        ) -> None:
            inputs = fd.define_tensor(
                shape=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
            )
            weights = fd.define_tensor(
                shape=[-1], contiguity=[True], dtype=DataType.Float
            )
            bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float)
            sum0 = fd.ops.sum(inputs, dims=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_scalar(norm_size)
            mean = fd.ops.div(sum0, norm_const)
            diff = fd.ops.sub(inputs, mean)
            diff_sq = fd.ops.mul(diff, diff)
            sum1 = fd.ops.sum(diff_sq, dims=[normalization_axis], keepdim=keepDim)
            var = fd.ops.div(sum1, norm_const)
            eps_const = fd.define_scalar(eps)

    def test_index_select(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.index_select(inputs[0] + inputs[1], dim, inputs[2])
            self.assertEqual(eager_out, nvf_out[0])

        test_fn(0)
        test_fn(1)

    def test_index_select_scalar_indices(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.index_select(inputs[0], dim, inputs[1])
            self.assertEqual(eager_out, nvf_out[0])

        test_fn(0)
        test_fn(1)

    def test_select(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.select(inputs[0], dim, inputs[1])
            self.assertEqual(eager_out, nvf_out[0])

        test_fn(0)
        test_fn(1)

    def test_where(self):
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
            nv_result, _ = self.exec_nvfuser(fusion_func, [pred, a, b])
            torch_result = torch.where(pred, a, b)
            self.assertEqual(nv_result[0], torch_result)

    def test_where_dtypes(self):
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
        ), _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.where(inputs[0], 3.0, 5.0)

        # explicit Float dtype matches torch.where behavior
        self.assertEqual(eager_out, nf)

        assert n.dtype == torch.float64
        assert nf.dtype == torch.float32
        assert nd.dtype == torch.float64
        assert ni.dtype == torch.int32
        assert nl.dtype == torch.int64
        assert nc.dtype == torch.complex128
        assert ncf.dtype == torch.complex64
        assert ncd.dtype == torch.complex128
        assert nb.dtype == torch.bool

    def test_addcmul(self):
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

        nvfout, _ = self.exec_nvfuser(fusion_func, inputs)

        torch_out = torch.addcmul(*inputs, value=0.1)

        self.assertEqual(nvfout[0], torch_out)

    def test_slice(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        for out in nvf_out:
            self.assertTrue(out.allclose(x[:, 1:, 2:]))

    def test_iota(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, [])

        eager_out1 = torch.tensor([0, 2], dtype=torch.long, device="cuda")
        eager_out2 = torch.tensor([100, 101, 102], dtype=torch.int, device="cuda")
        self.assertEqual(eager_out1, nvf_out[0])
        self.assertEqual(eager_out2, nvf_out[1])

    def test_scalar_only_inputs(self):
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

        # TODO: full is broken and does not print its proper definition
        # Issue: https://github.com/csarofeen/pytorch/issues/2502
        nvf_out = fd.execute([2.0, 3.0])
        eager_out = torch.full([2, 2], 1.0) * 5.0
        self.assertEqual(eager_out, nvf_out[0])

    def test_nanogpt_split_mha_linears(self):
        inputs = [
            torch.randn(16, 128, 3072, device="cuda"),
        ]

        def nvfuser_fusion_0(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T0_slice1 = fd.ops.slice(T0, [0, 0, 0], [16, 128, 1024], [1, 1, 1])
            T0_slice2 = fd.ops.slice(T0, [0, 0, 1024], [16, 128, 2048], [1, 1, 1])
            T0_slice3 = fd.ops.slice(T0, [0, 0, 2048], [16, 128, 3072], [1, 1, 1])
            T1_slice1 = fd.ops.reshape(T0_slice1, [16, 128, 16, 64])
            T1_slice2 = fd.ops.reshape(T0_slice2, [16, 128, 16, 64])
            T1_slice3 = fd.ops.reshape(T0_slice3, [16, 128, 16, 64])
            T2_slice1 = fd.ops.permute(T1_slice1, [0, 2, 1, 3])
            T2_slice2 = fd.ops.permute(T1_slice2, [0, 2, 1, 3])
            T2_slice3 = fd.ops.permute(T1_slice3, [0, 2, 1, 3])
            fd.add_output(T2_slice1)
            fd.add_output(T2_slice2)
            fd.add_output(T2_slice3)

        def torch_def_0(acts, n_embd, n_head):
            B, T, C = acts.size()
            q, k, v = acts.split(n_embd, dim=2)
            k = k.view(B, T, n_head, (C // 3) // n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            q = q.view(B, T, n_head, (C // 3) // n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            v = v.view(B, T, n_head, (C // 3) // n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            return (
                q,
                k,
                v,
            )

        def nvfuser_fusion_1(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            T1 = fd.ops.slice(
                T0,
                start_indices=[0, 0, 0],
                end_indices=[16, 128, 1024],
                strides=[1, 1, 1],
            )
            T2 = fd.ops.slice(
                T0,
                start_indices=[0, 0, 1024],
                end_indices=[16, 128, 2048],
                strides=[1, 1, 1],
            )
            T3 = fd.ops.slice(
                T0,
                start_indices=[0, 0, 2048],
                end_indices=[16, 128, 3072],
                strides=[1, 1, 1],
            )
            fd.add_output(T1)
            fd.add_output(T2)
            fd.add_output(T3)

        def torch_def_1(acts, n_embd, n_head):
            B, T, C = acts.size()
            q, k, v = acts.split(n_embd, dim=2)
            return (
                q,
                k,
                v,
            )

        tests = [
            (nvfuser_fusion_0, torch_def_0),
            (nvfuser_fusion_1, torch_def_1),
        ]

        for nvf_func, torch_func in tests:
            nvf_out, _ = self.exec_nvfuser(nvf_func, inputs)
            eager_out = torch_func(*inputs, 1024, 16)
            for idx in range(len(eager_out)):
                self.assertEqual(eager_out[idx], nvf_out[idx])

    def test_alias_output_to_input(self):
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

        out_tensors, _ = self.exec_nvfuser(fusion_func, in_tensors)

        # t1 is an alias and therefore is hidden.
        self.assertEqual(len(out_tensors), 1)
        self.assertEqual(out_tensors[0], torch.full((4, 4), 4.0, device="cuda"))
        self.assertEqual(in_tensors[0], torch.full((4, 4), 2.0, device="cuda"))

    def test_welford(self):
        num_elem = 2
        inputs = [torch.randn(2, num_elem, device="cuda")]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            mean, var_sum, n = fd.ops.welford(t0, [-1])
            var = fd.ops.div(var_sum, n)
            fd.add_output(var)
            fd.add_output(mean)

        fuser_result, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_result = torch.var_mean(inputs[0], [-1], correction=0)
        self.assertEqual(fuser_result, torch_result)

    def test_gather(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
            self.assertEqual(eager_out, nvf_out[0])

    def test_pad(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(torch.nn.functional.pad(inputs[0], [1, 1, 1, 1]), nvf_out[0])
        self.assertEqual(torch.nn.functional.pad(inputs[0], [0, 0, 2, 3]), nvf_out[1])
        self.assertEqual(torch.nn.functional.pad(inputs[0], [0, 0, 0, 0]), nvf_out[2])
        self.assertEqual(torch.nn.functional.pad(inputs[0], [2, 3]), nvf_out[3])
        self.assertEqual(
            torch.nn.functional.pad(inputs[0], [2, 3], "constant", 2.0), nvf_out[4]
        )
        self.assertEqual(
            torch.nn.functional.pad(inputs[0], [2, 3, 0, 0, 0, 0]), nvf_out[5]
        )

    def test_pad_dynamic(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(
            torch.nn.functional.pad(inputs[0], [17, 17, 17, 17]), nvf_out[0]
        )

    def test_cat(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(torch.cat([inputs[0], inputs[1]], dim=1), nvf_out[0])
        self.assertEqual(torch.cat([inputs[0], inputs[2]], dim=0), nvf_out[1])

    def test_sdpa_fwd(self):
        def fusion_func(
            fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
        ):
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            dropout_p, is_causal, scale = None, None, None
            if has_dropout:
                dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
            if has_causal:
                is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
            if has_scale:
                scale = fd.define_scalar(value=None, dtype=DataType.Double)
            attn, *_ = fd.ops.sdpfa_fwd(q, k, v, dropout_p, is_causal, scale)
            fd.add_output(attn)

        N, H, L, S, E = 4, 8, 16, 16, 8
        qkv = [
            torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
        ]

        dropout_vals = [None, 0.0, 0.2]
        is_causal_vals = [None, True, False]
        scale_vals = [None, 1 / E**0.5, 1e-3]
        # TODO: Try to move this to pytest_ops.py. Currently, it does not work since the API between nvFuser and torch differs.
        for dropout_p, is_causal, scale in itertools.product(
            dropout_vals, is_causal_vals, scale_vals
        ):
            with self.subTest(dropout_p=dropout_p, is_causal=is_causal, scale=scale):
                from torch.nn.attention import SDPBackend, sdpa_kernel

                has_dropout = True if dropout_p is not None else False
                has_causal = True if is_causal is not None else False
                has_scale = True if scale is not None else False
                inputs = [*qkv]
                for param in [dropout_p, is_causal, scale]:
                    if param is not None:
                        inputs.append(param)
                nvf_out, _ = self.exec_nvfuser(
                    partial(
                        fusion_func,
                        has_dropout=has_dropout,
                        has_causal=has_causal,
                        has_scale=has_scale,
                    ),
                    inputs,
                )

                # Torch does not accept NoneType dropout_p, is_causal.
                dropout_p = 0.0 if dropout_p is None else dropout_p
                is_causal = False if is_causal is None else is_causal

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.manual_seed(0)
                    ref_out = torch.nn.functional.scaled_dot_product_attention(
                        *qkv, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                    )
                torch.testing.assert_close(nvf_out[0], ref_out)

    def test_sdpa_fwd_bwd(self):
        N, H, L, S, E = 4, 8, 16, 16, 8

        dropout_vals = [None, 0.0, 0.2]
        is_causal_vals = [None, True, False]
        scale_vals = [None, 1 / E**0.5, 1e-3]

        def fusion_func(
            fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
        ):
            q = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            k = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            v = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )
            grad_out = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
            )

            dropout_p, is_causal, scale = None, None, None
            if has_dropout:
                dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
            if has_causal:
                is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
            if has_scale:
                scale = fd.define_scalar(value=None, dtype=DataType.Double)

            output, log_sumexp, philox_seed, philox_offset = fd.ops.sdpfa_fwd(
                q, k, v, dropout_p, is_causal, scale
            )
            grad_query, grad_key, grad_value = fd.ops.sdpfa_bwd(
                grad_out,
                q,
                k,
                v,
                output,
                log_sumexp,
                dropout_p,
                is_causal,
                philox_seed,
                philox_offset,
                scale,
            )

            fd.add_output(output)
            fd.add_output(grad_query)
            fd.add_output(grad_key)
            fd.add_output(grad_value)

        for dropout_p, is_causal, scale in itertools.product(
            dropout_vals, is_causal_vals, scale_vals
        ):
            with self.subTest(dropout_p=dropout_p, is_causal=is_causal, scale=scale):
                from torch.nn.attention import SDPBackend, sdpa_kernel

                q = torch.randn(
                    (N, H, L, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                k = torch.randn(
                    (N, H, S, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                v = torch.randn(
                    (N, H, S, E),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                    requires_grad=True,
                )
                grad_output = torch.randn(
                    (N, H, L, E), dtype=torch.bfloat16, device="cuda:0"
                )

                has_dropout = True if dropout_p is not None else False
                has_causal = True if is_causal is not None else False
                has_scale = True if scale is not None else False

                inputs = [q, k, v, grad_output]
                for param in [dropout_p, is_causal, scale]:
                    if param is not None:
                        inputs.append(param)

                # Torch does not accept NoneType dropout_p, is_causal.
                dropout_p = 0.0 if dropout_p is None else dropout_p
                is_causal = False if is_causal is None else is_causal

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.manual_seed(0)
                    ref_out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                    )
                    ref_out.backward(grad_output)

                nvf_out, _ = self.exec_nvfuser(
                    partial(
                        fusion_func,
                        has_dropout=has_dropout,
                        has_causal=has_causal,
                        has_scale=has_scale,
                    ),
                    inputs,
                )
                torch.testing.assert_close(nvf_out[0], ref_out)
                torch.testing.assert_close(nvf_out[1], q.grad)
                torch.testing.assert_close(nvf_out[2], k.grad)
                torch.testing.assert_close(nvf_out[3], v.grad)

    def test_normal(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        # Is there a better way to test distribution?!
        self.assertTrue(
            nvf_out[0]
            .mean()
            .cpu()
            .float()
            .isclose(torch.tensor(mean), rtol=1e-2, atol=1e-2)
            .item()
        )
        self.assertTrue(
            nvf_out[0]
            .std()
            .cpu()
            .float()
            .isclose(torch.tensor(std), rtol=1e-2, atol=1e-2)
            .item()
        )

    def test_uniform(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        # Is there a better way to test distribution?!
        self.assertTrue(
            nvf_out[0]
            .mean()
            .cpu()
            .float()
            .isclose(torch.tensor((hi - lo) / 2.0), rtol=1e-2, atol=1e-2)
            .item()
        )
        self.assertTrue(
            nvf_out[0]
            .min()
            .cpu()
            .float()
            .isclose(torch.tensor(lo), rtol=1e-2, atol=1e-2)
            .item()
        )
        self.assertTrue(
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
