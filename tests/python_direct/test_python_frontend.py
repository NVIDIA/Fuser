# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
import torch._refs as refs
import torch._prims as prims
from typing import List

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)

from nvfuser_direct.testing.utils import is_pre_volta
from utils import NVFuserTest


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
