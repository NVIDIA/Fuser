# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from functools import partial
import itertools
import io
import math
import re
import random
import sys
from typing import List

import torch
import torch.nn.functional as F
import torch._refs as refs
import torch._prims as prims

from nvfuser import (
    FusionDefinition,
    FusionCache,
    DataType,
    Tensor,
    version,
    compute_contiguity,
    compute_tensor_descriptor,
)
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

from python.utils import (
    is_pre_volta,
    is_pre_ampere,
    is_pre_hopper,
    is_pre_blackwell,
    debug_serde,
    NVFuserTest,
    verify_stride_order,
)
import pytest


@pytest.mark.skipif(is_pre_volta(), reason="Only supported on Volta and newer devices.")
class TestNvFuserFrontend(NVFuserTest):
    def test_basic(self):
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

        # Expected Output is a tensor of 48's
        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)

        # Create a new fusion with the same definition, it should hit the cache!
        nvf_out2, fd2 = self.exec_nvfuser(
            fusion_func, inputs, new_fusion_expected=False
        )

        # Create a fusion from a fusion id and make sure it executes!
        fd3 = FusionDefinition(fd2.id())
        nvf_out3 = fd3.execute(inputs)

        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out1[0])
        self.assertEqual(eager_out, nvf_out2[0])
        self.assertEqual(eager_out, nvf_out3[0])

    def test_basic_fp16(self):
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

        # Expected Output is a tensor of 48's
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

    def test_cast_double_to_half(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0].to(torch.half) + inputs[1].to(torch.half))
        self.assertEqual(eager_out, nvf_out[0])

    @pytest.mark.skipif(
        is_pre_hopper(), reason="Only supported on Hopper and newer devices."
    )
    def test_cast_fp8(self):
        def fn(in_type, out_type):
            inputs = [
                torch.randn([5, 5], device="cuda").to(in_type),
            ]

            def fusion_func(fd: FusionDefinition) -> None:
                T0 = fd.from_pytorch(inputs[0])
                T1 = fd.ops.cast(T0, dtype=torch_dtype_to_nvfuser_dtype(out_type))
                fd.add_output(T1)

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
            eager_out = inputs[0].to(out_type)
            if in_type == torch.float8_e8m0fnu or out_type == torch.float8_e8m0fnu:
                # Eager mode uses manual bit manipulation, and nvFuser uses
                # hardware instructions. Unfortunately, these implementations
                # do not match exactly. e8m0 can only represent 2^x, so we are
                # asserting that the x of the two results are off by at most 1.
                nvf_out_fp32 = nvf_out[0].to(torch.float32)
                eager_out_fp32 = eager_out.to(torch.float32)
                rel_err = eager_out_fp32.div(nvf_out_fp32).max().item()
                self.assertTrue(rel_err <= 2 and rel_err >= 0.5)
            else:
                self.assertEqual(eager_out, nvf_out[0])

        for type0 in [torch.double, torch.float32, torch.float16, torch.bfloat16]:
            type1_list = [torch.float8_e4m3fn, torch.float8_e5m2]
            if not is_pre_blackwell():
                type1_list.append(torch.float8_e8m0fnu)
            for type1 in type1_list:
                fn(type0, type1)
                fn(type1, type0)

    def test_promote_to_double(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0] + inputs[1])
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

    def test_ops_broadcast(self):
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
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, shape=input_shape, broadcast_dims=[2]
            )
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(
                bias, shape=input_shape, broadcast_dims=[2]
            )
            out = fd.ops.add(scale, bias_bcast)
            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)

        def nvfuser_fusion_var_mean(
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
            var, mean = fd.ops.var_mean(
                inputs, dims=[normalization_axis], correction=0, keepdim=keepDim
            )
            eps_const = fd.define_scalar(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            diff = fd.ops.sub(inputs, mean)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, shape=input_shape, broadcast_dims=[2]
            )
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(
                bias, shape=input_shape, broadcast_dims=[2]
            )
            out = fd.ops.add(scale, bias_bcast)
            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)

        fusion_func_1 = partial(
            nvfuser_fusion,
            normalization_axis=2,
            norm_size=inputs[0].size()[2],
            input_shape=inputs[0].size(),
            eps=1e-12,
            keepDim=True,
        )
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs)

        fusion_func_2 = partial(
            nvfuser_fusion_var_mean,
            normalization_axis=2,
            norm_size=inputs[0].size()[2],
            input_shape=inputs[0].size(),
            eps=1e-12,
            keepDim=True,
        )
        nvf_var_mean_out, _ = self.exec_nvfuser(fusion_func_2, inputs)

        eager_out = primitive_definition(inputs[0], inputs[1], inputs[2], 2, True)

        self.assertEqual(eager_out, nvf_out[0])
        self.assertEqual(eager_out, nvf_var_mean_out[0])

    def test_prim_rms_norm_fwd(self):
        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = "cuda"
        inputs = [
            torch.randn(*input_size, device=device, requires_grad=True),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
        ]

        def primitive_definition(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            normalization_axis: int,
            keepdim: bool,
        ) -> torch.Tensor:
            var = inputs.mul(inputs).mean(normalization_axis, keepdim)
            pre_shift_scale_norm_output = inputs / torch.sqrt(var + 1e-12)
            norm_output = weight * pre_shift_scale_norm_output
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
            inputs_sq = fd.ops.mul(inputs, inputs)
            sum0 = fd.ops.sum(inputs_sq, dims=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_scalar(norm_size)
            var = fd.ops.div(sum0, norm_const)
            eps_const = fd.define_scalar(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale = fd.ops.mul(inputs, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, shape=input_shape, broadcast_dims=[2]
            )
            out = fd.ops.mul(pre_scale, weights_bcast)
            fd.add_output(out)
            fd.add_output(invstd)

        fusion_func = partial(
            nvfuser_fusion,
            normalization_axis=2,
            norm_size=inputs[0].size()[2],
            input_shape=inputs[0].size(),
            eps=1e-12,
            keepDim=True,
        )
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = primitive_definition(inputs[0], inputs[1], 2, True)

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
        nvf_out, _ = self.exec_nvfuser(
            fusion_func, inputs_with_tuple, new_fusion_expected=False
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where a broadcast requires a symbolic output shape
    def test_tensor_shape(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.sub(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where no broadcast is needed
    def test_tensor_shape_nobcast(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1])
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where each arg of a binary op has broadcast.
    def test_tensor_size_both_args_bcast(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            prims.broadcast_in_dim(
                inputs[0], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]
            ),
            prims.broadcast_in_dim(
                inputs[1], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]
            ),
        )
        self.assertEqual(eager_out, nvf_out[0])

    def test_broadcast_in_dim_with_dynamic_shapes(self):
        inputs_1 = [
            torch.randn(2, 3, 4, device="cuda"),
            torch.randn(4, device="cuda"),
        ]
        inputs_2 = [
            torch.randn(2, 3, 1024, device="cuda"),
            torch.randn(1024, device="cuda"),
        ]

        def fusion_func_1(fd: FusionDefinition):
            t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
            t1 = fd.define_tensor(shape=[-1], contiguity=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_2(fd: FusionDefinition):
            t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
            t1 = fd.define_tensor(shape=[-1], contiguity=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_1[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_3(fd: FusionDefinition):
            t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
            t1 = fd.define_tensor(shape=[-1], contiguity=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_2[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        # Func_1 uses tensor.shape() to propagate dynamic size, therefore, it is
        # expected that test 2 should be cached based on test 2

        # Test 1
        inputs = inputs_1
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

        # Test 2
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs, new_fusion_expected=False)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

        # Func_2 and Func_3 are nearly identical except that have a different
        # concrete output shape for their broadcast_in_dim.  Therefore, test 4
        # should not be cached.
        # Note: It is assumed that definition will change with Tensor Size with
        # concrete shapes.

        # Test 3
        inputs = inputs_1
        nvf_out, _ = self.exec_nvfuser(fusion_func_2, inputs)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

        # Test 4
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func_3, inputs)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where the broadcast is necessary to realize the output
    def test_tensor_shape_with_output_bcast(self):
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
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = prims.broadcast_in_dim(
            torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1]
        )
        self.assertEqual(eager_out, nvf_out[0])

        # Testing Dynamic usage of same Fusion
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, new_fusion_expected=False)
        eager_out = prims.broadcast_in_dim(
            torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1]
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing an expand followed by a  broadcast
    def test_tensor_shape_expand_bcast(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out1 = prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1, 2])
        eager_out2 = prims.broadcast_in_dim(inputs[2], eager_out1.size(), [0, 1, 2])
        self.assertEqual(eager_out2, nvf_out[0])

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

    def test_gather(self):
        inputs = [
            torch.randn(8, 16, device="cuda"),
            torch.randn(8, 16, device="cuda"),
            torch.randint(0, 8, (4, 4), device="cuda").to(dtype=torch.long),
        ]

        def test_fn(dim):
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

        test_fn(0)
        test_fn(1)

    def test_take_along_axis(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
            self.assertEqual(eager_out, nvf_out[0])

        test_fn(0)
        test_fn(1)

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

    def test_from_pytorch_fails_on_cpu_tensor(self):
        inputs = [
            torch.randn(4, 4, device="cpu"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.relu(t0)
            fd.add_output(t1)

        try:
            with FusionDefinition() as fd:
                fusion_func(fd)
            raise RuntimeError(
                "FusionDefinition.from_pytorch should have raised an error for a CPU Tensor!"
            )
        except ValueError:
            pass

    def test_no_definition(self):
        inputs = [
            torch.randn(4, 4, device="cpu"),
        ]

        # A FusionDefinition object is constructed but not defined, should trip an error
        try:
            fd = FusionDefinition()
            out = fd.execute(inputs)
            raise RuntimeError(
                "Expecting an error for a lack of a child class defining a definition!"
            )
        except NotImplementedError:
            pass

    def test_func_definition(self):
        inputs = [
            torch.randn(4, 4, device="cuda"),
        ]

        class MyFusion(FusionDefinition):
            def definition(self):
                t0 = self.from_pytorch(inputs[0])
                t1 = self.ops.sigmoid(t0)
                self.add_output(t1)

        fd = MyFusion()
        nvf_out = fd.execute(inputs)
        eager_out = torch.sigmoid(inputs[0])
        self.assertEqual(eager_out, nvf_out[0])

    def test_python_version_API(self):
        from nvfuser.nvfuser_version import Version

        self.assertTrue(version() > "0.0.0")
        self.assertTrue(version() > Version("0.0.0"))

    def test_zero_size_dim(self):
        inputs = [
            torch.ones(0, 0, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.define_tensor(
                shape=[0, 0], contiguity=[True, True], dtype=DataType.Float
            )
            t1 = fd.ops.relu(t0)
            fd.add_output(t1)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0])
        self.assertEqual(eager_out.numel(), nvf_out[0].numel())

    def test_static_tensor_sizes(self):
        inputs = [
            torch.randn(4, 5, 1, device="cuda"),
            torch.randn(1, 5, 6, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            t1 = fd.from_pytorch(inputs[1], static_sizes=True)
            t2 = fd.ops.mul(t0, t1)
            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.mul(inputs[0], inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

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

    def test_complex_constants(self):
        inputs = [
            torch.arange(2, device="cuda").type(torch.complex64),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_scalar(complex(3.0, 0.5))
            t1 = fd.ops.mul(t0, c0)
            fd.add_output(t1)

        (n,), _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = inputs[0] * (3.0 + 0.5j)

        self.assertEqual(eager_out, n)
        assert n.dtype == torch.complex64

    def test_where_op(self):
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

    def test_iota(self):
        inputs = [
            (2, 0, 2, DataType.Int),
            (3, 100, 1, DataType.Int32),
            # TODO: How do I that that? I am getting the following error:
            # NameError: name 'None0' is not defined
            # (4, None, None, DataType.Int),
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
        eager_out3 = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
        self.assertEqual(eager_out1, nvf_out[0])
        self.assertEqual(eager_out2, nvf_out[1])
        # self.assertEqual(eager_out3, nvf_out[2])

    def test_triu(self):
        inputs = [
            torch.randn(4, 16, device="cuda", dtype=torch.float16),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.triu(t0, -1)
            fd.add_output(t1)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out0 = torch.triu(inputs[0], -1)
        self.assertEqual(eager_out0, nvf_out[0])

    def test_complex_rsqrt(self):
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

        (rfloat, rdouble), _ = self.exec_nvfuser(fusion_func, inputs)

        at_rfloat = inputs[0].rsqrt()
        at_rdouble = inputs[1].rsqrt()

        self.assertEqual(at_rfloat, rfloat)
        self.assertEqual(at_rdouble, rdouble)

    def test_reduction_complex_number(self):
        def test_dtype(torch_dtype):
            inputs = [torch.randn(2, 32, device="cuda", dtype=torch_dtype)]

            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                t1 = fd.ops.sum(
                    t0, [-1], False, torch_dtype_to_nvfuser_dtype(torch_dtype)
                )
                fd.add_output(t1)

            nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)
            eager_out = torch.sum(inputs[0], dim=-1)
            self.assertEqual(eager_out, nvf_out1[0])

        list_of_dtype = [torch.complex64, torch.complex128]
        for torch_dtype in list_of_dtype:
            test_dtype(torch_dtype)

    def test_segmentation_reduction_pointwise_epilogue(self):
        inputs = [
            torch.randn(2, 32, device="cuda", dtype=torch.float32),
            torch.randn(2, 128, device="cuda", dtype=torch.float32),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.sum(t0, [-1], True, torch_dtype_to_nvfuser_dtype(torch.float32))
            t3 = fd.ops.add(t1, t2)
            fd.add_output(t3)

        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum(inputs[0], dim=-1, keepdim=True) + inputs[1]
        self.assertEqual(eager_out, nvf_out1[0])

        with FusionDefinition() as fd:
            fusion_func(fd)

        # create segments
        segments = fd.segment(inputs)

        # Each segment has a map from its segment index space to the original
        # fusion's index space. The original fusion creates two segments.
        # Check that segment maps match expected behavior.
        assert len(fd.segment_index_space_maps) == 2

        # First Segment:
        # def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
        #    T0 = fd.define_tensor(shape=[-1, -1],
        #                          contiguity=[True, True],
        #                          dtype=DataType.Float,
        #                          is_cpu=False)
        #    T1 = fd.ops.sum(T0, dims=[1], keepdim=False, dtype=DataType.Float)
        #    T2 = fd.ops.broadcast(T1, is_broadcast_dim=[False, True])
        #    fd.add_output(T2)
        #
        assert fd.segment_index_space_maps[segments[0]] == {2: 2, 0: 0}

        # Second Segment:
        # def nvfuser_fusion_id3(fd : FusionDefinition) -> None :
        #    T0 = fd.define_tensor(shape=[-1, -1],
        #                          contiguity=[True, True],
        #                          dtype=DataType.Float,
        #                          is_cpu=False)
        #    T1 = fd.define_tensor(shape=[-1, 1],
        #                          contiguity=[True, None],
        #                          dtype=DataType.Float,
        #                          is_cpu=False)
        #    T2 = fd.ops.add(T0, T1)
        #    fd.add_output(T2)
        assert fd.segment_index_space_maps[segments[1]] == {2: 3, 1: 2, 0: 1}

    def test_arithmetic_ops(self):
        inputs = [
            torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])

            c0 = fd.define_scalar(1.0)

            t1 = -t0
            t2 = abs(t0)
            c1 = -c0
            c2 = abs(c0)

            # Using literals like this will work once
            # https://github.com/csarofeen/pytorch/pull/2449 is merged
            # t3 = -t1 * (1 + t0 ** 2) / t2 + c2 ** c1 - 1.0
            t3 = -t1 * (c0 - t0 * t0) / t2 + c2**c1 - c0

            fd.add_output(t1)
            fd.add_output(t2)
            fd.add_output(t3)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        at_out0 = -inputs[0]
        at_out1 = abs(inputs[0])
        at_out2 = inputs[0] * (1.0 - inputs[0] * inputs[0]) / abs(inputs[0])

        self.assertEqual(at_out0, nvf_out[0])
        self.assertEqual(at_out1, nvf_out[1])
        self.assertEqual(at_out2, nvf_out[2])

    def test_signbit(self):
        inputs = [
            torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
            torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.where(fd.ops.signbit(t0), -abs(t1), abs(t1))
            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        at_out = torch.where(
            torch.signbit(inputs[0]), -torch.abs(inputs[1]), torch.abs(inputs[1])
        )
        self.assertEqual(at_out, nvf_out[0])

    def test_all_dim_var_mean(self):
        inputs = [torch.randn(2, 2, 2, device="cuda")]

        # use decorator to create fusion_func
        def fusion_decorator(correction):
            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                t1, t2 = fd.ops.var_mean(t0, [0, 1, 2], correction)
                fd.add_output(t1)
                fd.add_output(t2)

            return fusion_func

        list_of_test_cases = [0, 1]
        for correction in list_of_test_cases:
            fuser_result, _ = self.exec_nvfuser(fusion_decorator(correction), inputs)
            torch_result = torch.var_mean(inputs[0], [0, 1, 2], bool(correction))
            self.assertEqual(fuser_result, torch_result)

    def test_var_mean_correction(self):
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

        for correction in range(num_elem + 5):
            fuser_result, _ = self.exec_nvfuser(fusion_decorator(correction), inputs)
            torch_result = torch.var_mean(inputs[0], [-1], correction=correction)
            self.assertEqual(fuser_result, torch_result)

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

    def test_var_correction(self):
        num_elem = 2
        inputs = [torch.randn(2, num_elem, device="cuda")]

        # use decorator to create fusion_func
        def fusion_decorator(correction):
            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                t1 = fd.ops.var(t0, [-1], correction)
                fd.add_output(t1)

            return fusion_func

        for correction in range(num_elem + 5):
            fuser_result, _ = self.exec_nvfuser(fusion_decorator(correction), inputs)
            torch_result = torch.var(inputs[0], [-1], correction=correction)
            self.assertEqual(fuser_result[0], torch_result)

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

    def test_compute_contiguity(self):
        sizes = [2, 1, 3, 1, 4, 5, 6]
        strides = [80, 30, 30, 456456465465, 0, 6, 1]
        contiguity = [False, None, True, None, None, True, True]
        self.assertEqual(compute_contiguity(sizes, strides), contiguity)
        strides = [800, 300, 300, 456456465465, 0, 60, 10]
        contiguity = [False, None, True, None, None, True, False]
        self.assertEqual(compute_contiguity(sizes, strides), contiguity)

    def test_compute_tensor_descriptor(self):
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
            self.assertEqual(computed_contiguity, contiguity)
            self.assertEqual(computed_stride_order, stride_order)

    def test_stride_order_with_explicit_broadcast(self):
        inputs = [
            torch.randn(3, device="cuda").unsqueeze(-1),
            torch.randn(2, 3, device="cuda")
            .unsqueeze(-1)
            .expand(2, 3, 4)
            .transpose(2, 0),
            torch.randn(5 * 960, device="cuda").as_strided(
                (5, 4, 1, 5, 16), (960, 48, 16, 192, 1)
            ),
            torch.randn(6, device="cuda").as_strided((2, 16, 3), (3, 0, 1)),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.define_tensor(
                shape=[-1, 16, 3],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                stride_order=[1, 2, 0],
                is_cpu=False,
            )

            t0_b = fd.ops.broadcast(t0, [True, False, False])
            t4 = fd.ops.add(t0_b, t1)
            c0 = fd.define_scalar(3.0)
            t5 = fd.ops.add(t2, c0)
            t6 = fd.ops.mul(t3, c0)

            fd.add_output(t4)
            fd.add_output(t5)
            fd.add_output(t6)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(nvf_out[0], inputs[0] + inputs[1])
        self.assertEqual(nvf_out[1], inputs[2] + 3.0)
        self.assertEqual(nvf_out[2], inputs[3] * 3.0)

    def test_prod(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_outs = [
            torch.prod(inputs[0], dtype=torch.float32),
            torch.prod(inputs[0], 1, False, dtype=torch.float32),
            torch.prod(inputs[0], 1, True, dtype=torch.float32),
            torch.prod(inputs[0], -1, False, dtype=torch.float32),
        ]
        assert len(nvf_out) == len(eager_outs)

        for n, e in zip(nvf_out, eager_outs):
            self.assertEqual(n, e)

    def test_output_stride_order(self):
        inputs = [
            torch.arange(0, 120).reshape(2, 3, 4, 5).cuda().float(),
        ]
        eager_out = inputs[0] + 3.0

        for perm in itertools.permutations(range(4), 4):
            # testing stride_order in add_output
            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                c0 = fd.define_scalar(3.0)
                t1 = fd.ops.add(t0, c0)
                fd.add_output(t1, perm)

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
            self.assertEqual(eager_out, nvf_out[0])

            nvf_stride = nvf_out[0].stride()
            verify_stride_order(nvf_stride, perm)

            # testing stride_order in set
            def fusion_set_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                c0 = fd.define_scalar(3.0)
                t1 = fd.ops.add(t0, c0)
                t2 = fd.ops.stride_order(t1, perm)
                fd.add_output(t2)

            nvf_out, _ = self.exec_nvfuser(fusion_set_func, inputs)
            self.assertEqual(eager_out, nvf_out[0])

            nvf_stride = nvf_out[0].stride()
            verify_stride_order(nvf_stride, perm)

    def test_output_stride_order_with_reduction(self):
        inputs = [torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float)]

        for stride_order in itertools.permutations(range(3), 3):

            def fusion_add_output(fd: FusionDefinition) -> None:
                T0 = fd.from_pytorch(inputs[0])
                T1 = fd.ops.sum(T0, dims=[2])
                fd.add_output(T1, stride_order=stride_order)

            with FusionDefinition() as fd:
                fusion_add_output(fd)

            out = fd.execute(inputs)[0]
            verify_stride_order(out.stride(), stride_order)

            def fusion_stride_order_op(fd: FusionDefinition) -> None:
                T0 = fd.from_pytorch(inputs[0])
                T1 = fd.ops.sum(T0, dims=[2])
                T2 = fd.ops.stride_order(T1, stride_order)
                fd.add_output(T2)

            with FusionDefinition() as fd:
                fusion_stride_order_op(fd)

            out = fd.execute(inputs)[0]
            verify_stride_order(out.stride(), stride_order)

    def test_expanded_bcast_tensor(self):
        inputs = [
            torch.tensor(1.5, device="cuda"),
            torch.randn(5, 5, 5, device="cuda"),
            torch.randint(0, 1, (5, 5), device="cuda")
            .bool()
            .unsqueeze(-1)
            .expand(5, 5, 5),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.from_pytorch(inputs[2])
            T3 = fd.ops.add(T0, T1)
            T4 = fd.ops.add(T2, T3)
            fd.add_output(T4)

        eager_out = inputs[0] + inputs[1] + inputs[2]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        self.assertEqual(eager_out, nvf_out[0])

    def test_segment_set(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        self.assertEqual(eager_out, nvf_out[0])

    def test_fix_2549(self):
        a = torch.ones(4, 1, dtype=torch.double, device="cuda")
        b = torch.ones(4, 4, dtype=torch.double, device="cuda")

        def nvfuser_fusion_id(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                sizes=a.shape, strides=a.stride(), dtype=DataType.Double, is_cpu=False
            )
            T1 = fd.define_tensor(
                sizes=b.shape, strides=b.stride(), dtype=DataType.Double, is_cpu=False
            )
            T2 = fd.ops.broadcast_in_dim(T0, shape=[4, 4], broadcast_dims=[0, 1])
            T3 = fd.ops.div(T1, T2)
            fd.add_output(T3)

        with FusionDefinition() as fd:
            nvfuser_fusion_id(fd)

        out = fd.execute([a, b])
        self.assertEqual(out[0], b / a)

    def test_real_imag(self):
        for dtype in [torch.complex128, torch.complex64]:
            inputs = [
                torch.randn(5, dtype=dtype, device="cuda"),
            ]

            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                fd.add_output(fd.ops.real(t0))
                fd.add_output(fd.ops.imag(t0))

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            self.assertEqual(torch.real(inputs[0]), nvf_out[0])
            self.assertEqual(torch.imag(inputs[0]), nvf_out[1])

    def test_cuda_code_and_scheduled_fusion_ir_strings(self):
        inputs = [
            torch.randn(2, 2, 2, 2, device="cuda"),
        ]
        big_inputs = [
            torch.randn(64, 64, 64, 64, device="cuda"),
        ]

        # Function only based definition
        class DefFuncFusion(FusionDefinition):
            def definition(self):
                t0 = self.from_pytorch(inputs[0])
                t1 = self.ops.relu(t0)
                self.add_output(t1)

        # Function based definition plus a user schedule
        class UserSchedFusion(FusionDefinition):
            def definition(self):
                t0 = self.from_pytorch(inputs[0])
                t1 = self.ops.sinh(t0)
                self.add_output(t1)

            def schedule(self):
                pass

        # Context Based Definition
        ctx_fusion = FusionDefinition()
        with ctx_fusion:
            t0 = ctx_fusion.from_pytorch(inputs[0])
            t1 = ctx_fusion.ops.tanh(t0)
            ctx_fusion.add_output(t1)

        # Context Based Definition with a segmented fusion
        ctx_seg_fusion = FusionDefinition()
        with ctx_seg_fusion:
            t0 = ctx_seg_fusion.from_pytorch(inputs[0])
            t1 = ctx_seg_fusion.ops.sum(t0, dim=0)
            t2 = ctx_seg_fusion.ops.sum(t0, dim=-1)
            ctx_seg_fusion.add_output(t1)
            ctx_seg_fusion.add_output(t2)

        test_defs = [DefFuncFusion(), UserSchedFusion(), ctx_fusion, ctx_seg_fusion]

        for fd in test_defs:
            # Attempting to get the cuda code for an un-executed FusionDefinition
            # should trigger a RuntimeError and not a segfault
            with self.assertRaisesRegex(
                RuntimeError, "(Invalid fusion definition!|never been executed)"
            ):
                _ = fd.last_cuda_code()
            with self.assertRaisesRegex(
                RuntimeError, "(Invalid fusion definition!|never been executed)"
            ):
                _ = fd.last_scheduled_fusion_ir()
            # Only make this check for function based definitions
            if hasattr(super(type(self), self), "definition"):
                with self.assertRaisesRegex(RuntimeError, "Invalid fusion definition!"):
                    _ = fd.fusion_ir()

            _ = fd.execute(inputs)

            code_len = len(fd.sched.user_schedule_ir())
            self.assertTrue(code_len > 0, "User schedule is not defined.")
            code_len = len(fd.last_cuda_code())
            self.assertTrue(code_len > 0, "Cuda Code was not produced!")
            code_len = len(fd.last_cuda_code(intrinsic_code=True))
            self.assertTrue(code_len > 0, "Cuda Code was not produced!")
            sched_ir_len = len(fd.last_scheduled_fusion_ir())
            self.assertTrue(code_len > 0, "Scheduled Fusion IR was not produced!")
            sched_ir_len = len(fd.last_scheduled_fusion_ir(tensor_transforms=True))
            self.assertTrue(code_len > 0, "Scheduled Fusion IR was not produced!")
            sched_ir_len = len(fd.fusion_ir())
            self.assertTrue(code_len > 0, "Unscheduled Fusion IR was not produced!")

            code_len = len(fd.cuda_code_for(inputs))
            self.assertTrue(code_len > 0, "Cuda Code was not produced!")
            code_len = len(fd.cuda_code_for(inputs, intrinsic_code=True))
            self.assertTrue(code_len > 0, "Cuda Code was not produced!")
            sched_ir_len = len(fd.scheduled_fusion_ir_for(inputs))
            self.assertTrue(code_len > 0, "Scheduled Fusion IR was not produced!")
            sched_ir_len = len(
                fd.scheduled_fusion_ir_for(inputs, tensor_transforms=True)
            )
            self.assertTrue(code_len > 0, "Scheduled Fusion IR was not produced!")

            # Attempt to get strings for inputs that do not heuristically match
            # and a new fusion has not been compiled
            with self.assertRaisesRegex(
                RuntimeError, "(not compiled|never been executed)"
            ):
                _ = fd.cuda_code_for(big_inputs)
            with self.assertRaisesRegex(
                RuntimeError, "(not compiled|never been executed)"
            ):
                _ = fd.scheduled_fusion_ir_for(big_inputs)

        # It is necessary to reset the Fusion Cache
        # so serialization/deserialization does not exhibit the same error across tests.
        fc = FusionCache.get()
        fc.reset()

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

        self.assertEqual(F.pad(inputs[0], [1, 1, 1, 1]), nvf_out[0])
        self.assertEqual(F.pad(inputs[0], [0, 0, 2, 3]), nvf_out[1])
        self.assertEqual(F.pad(inputs[0], [0, 0, 0, 0]), nvf_out[2])
        self.assertEqual(F.pad(inputs[0], [2, 3]), nvf_out[3])
        self.assertEqual(F.pad(inputs[0], [2, 3], "constant", 2.0), nvf_out[4])
        self.assertEqual(F.pad(inputs[0], [2, 3, 0, 0, 0, 0]), nvf_out[5])

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
            V18 = fd.define_vector([S16, S16, S16, S16], dtype=DataType.Int)

            t1 = fd.ops.pad(t0, V18)
            fd.add_output(t1)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(F.pad(inputs[0], [17, 17, 17, 17]), nvf_out[0])

    def test_pad_cache(self):
        """Test that using different pad widths causes a cache miss.

        cf. https://github.com/NVIDIA/Fuser/pull/10#pullrequestreview-1352667557
        """
        inputs = [
            torch.testing.make_tensor((2, 3), dtype=torch.float32, device="cuda"),
        ]

        def fusion_func_pad1(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.pad(t0, [1, 1])
            fd.add_output(t1)

        nvf_out1, _ = self.exec_nvfuser(
            fusion_func_pad1, inputs, new_fusion_expected=True
        )
        _ = self.exec_nvfuser(fusion_func_pad1, inputs, new_fusion_expected=False)

        def fusion_func_pad2(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.pad(t0, [2, 2])
            fd.add_output(t1)

        nvf_out2, _ = self.exec_nvfuser(
            fusion_func_pad2, inputs, new_fusion_expected=True
        )

        def fusion_func_pad3(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            fill_val = fd.define_scalar(2.0)
            t1 = fd.ops.pad(t0, [1, 1], fill_val)
            fd.add_output(t1)

        nvf_out3, _ = self.exec_nvfuser(
            fusion_func_pad3, inputs, new_fusion_expected=True
        )
        _ = self.exec_nvfuser(fusion_func_pad3, inputs, new_fusion_expected=False)

        self.assertEqual(F.pad(inputs[0], [1, 1]), nvf_out1[0])
        # Erroneous cache miss would use kernel 1 instead of 2
        self.assertEqual(F.pad(inputs[0], [2, 2]), nvf_out2[0])
        # Erroneous cache hit based on fill value would use kernel1
        self.assertEqual(F.pad(inputs[0], [1, 1], "constant", 2.0), nvf_out3[0])

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

            # torch.cat accepts empty tensors (size 0 in the concat dimension),
            # which do not affect the output.
            # The below fails with RuntimeError: mapped_id_resize != nullptr
            # INTERNAL ASSERT FAILED at
            # "/opt/pytorch/nvfuser/csrc/lower_index_compute.cpp":1306
            # t5 = fd.ops.cat([t0, t3], 0)
            # fd.add_output(t5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(torch.cat([inputs[0], inputs[1]], dim=1), nvf_out[0])
        self.assertEqual(torch.cat([inputs[0], inputs[2]], dim=0), nvf_out[1])
        # self.assertEqual(torch.cat([inputs[0], inputs[3]], dim=0), nvf_out[2])

    def test_pad_prior_cat(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        # pad tensors t0 and t1, so their first dimension are size 10.
        pad_input0 = torch.nn.functional.pad(inputs[0], [0, 0, 0, 8])
        pad_input1 = torch.nn.functional.pad(inputs[1], [0, 0, 0, 7])
        self.assertEqual(torch.cat([pad_input0, pad_input1], dim=1), nvf_out[0])

    def test_nextafter(self):
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
                if isinstance(t, Tensor):
                    # ...but skip outputting scalars, which we don't support
                    fd.add_output(t)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

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
            self.assertEqual(n, torch_out)

    def test_nanogpt_mha_dpa(self):
        inputs = [
            torch.randn(16, 16, 128, 128, device="cuda"),
            torch.randn(1, 1, 1024, 1024, device="cuda"),
        ]

        def nvfuser_fusion(fd: FusionDefinition, prob) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            T1 = fd.define_tensor(
                shape=[1, 1, -1, -1],
                contiguity=[None, None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            S2 = fd.define_scalar(0.125000, dtype=DataType.Double)
            T3 = fd.ops.mul(T0, S2)
            T4 = fd.ops.slice(
                T1,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 1, 128, 128],
                strides=[1, 1, 1, 1],
            )
            S5 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T6 = fd.ops.eq(S5, T4)
            T7 = fd.ops.broadcast_in_dim(
                T6, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            S8 = fd.define_scalar(float("-inf"), dtype=DataType.Double)
            T9 = fd.ops.where(T7, S8, T3)
            S10 = fd.define_scalar(-1, dtype=DataType.Int)
            S11 = fd.define_scalar(4, dtype=DataType.Int)
            S12 = fd.ops.add(S10, S11)
            T13 = fd.ops.max(T9, dims=[3], keepdim=False, dtype=DataType.Null)
            T14 = fd.ops.broadcast_in_dim(
                T13, shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
            )
            T15 = fd.ops.broadcast_in_dim(
                T14, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T16 = fd.ops.sub(T9, T15)
            T17 = fd.ops.exp(T16)
            S18 = fd.define_scalar(-1, dtype=DataType.Int)
            S19 = fd.define_scalar(4, dtype=DataType.Int)
            S20 = fd.ops.add(S18, S19)
            T21 = fd.ops.sum(T17, dims=[3], keepdim=False, dtype=DataType.Null)
            T22 = fd.ops.broadcast_in_dim(
                T21, shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
            )
            T23 = fd.ops.broadcast_in_dim(
                T22, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T24 = fd.ops.div(T17, T23)
            S25 = fd.define_scalar(16, dtype=DataType.Int)
            S26 = fd.define_scalar(16, dtype=DataType.Int)
            S27 = fd.define_scalar(128, dtype=DataType.Int)
            S28 = fd.define_scalar(128, dtype=DataType.Int)
            S29 = fd.define_scalar(0.00000, dtype=DataType.Double)
            S30 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T31 = fd.ops.uniform(
                S29, S30, shape=[S25, S26, S27, S28], dtype=DataType.Float
            )
            S32 = fd.define_scalar(1.0 - prob, dtype=DataType.Double)
            T33 = fd.ops.lt(T31, S32)
            T34 = fd.ops.cast(T33, dtype=DataType.Float)
            T35 = fd.ops.mul(T24, T34)
            S36 = fd.define_scalar(1.0 / (1.0 - prob), dtype=DataType.Double)
            T37 = fd.ops.mul(T35, S36)
            fd.add_output(T37)

        def torch_def(acts, bias, n_seq_len, n_head_dim, prob):
            att = acts * (1.0 / math.sqrt(n_head_dim))
            att = att.masked_fill(
                bias[:, :, :n_seq_len, :n_seq_len] == 0, float("-inf")
            )
            att = torch.nn.functional.softmax(att, dim=-1)
            att = torch.nn.functional.dropout(att, p=prob)
            return att

        # NOTE: The dropout probabilities need to be set to 0 elements zeroed out
        # in order to match implementations as eager and nvFuser do not have matching
        # blocking.
        nvf_out, _ = self.exec_nvfuser(partial(nvfuser_fusion, prob=0.0), inputs)
        eager_out = torch_def(inputs[0], inputs[1], 128, 64, 0.0)

        for idx in range(len(nvf_out)):
            self.assertEqual(eager_out, nvf_out[idx])

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

    def test_slice_error_checks(self):
        inputs = [
            [torch.randn(10, 10, device="cuda")],
            [torch.randn(5, 5, device="cuda")],
        ]

        def check_start_indices(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[-1, -2], end_indices=[5, 5], strides=[7, 7]
            )
            fd.add_output(T1)

        def check_end_indices(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[3, 4], end_indices=[1, 2], strides=[1, 1]
            )
            fd.add_output(T1)

        def check_strides(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[0, 0], end_indices=[5, 5], strides=[5, 5]
            )
            fd.add_output(T1)

        def check_tensor_dims(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[0, 0, 0], end_indices=[4, 4, 4], strides=[1, 1, 1]
            )
            fd.add_output(T1)

        def check_slice_dims_start(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[0, 0, 0], end_indices=[4, 4], strides=[1, 1]
            )
            fd.add_output(T1)

        def check_slice_dims_end(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[0, 0], end_indices=[4, 4, 4], strides=[1, 1]
            )
            fd.add_output(T1)

        def check_slice_dims_stride(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[0, 0], end_indices=[4, 4], strides=[1, 1, 1]
            )
            fd.add_output(T1)

        def check_nostrides(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(T0, start_indices=[2, 2], end_indices=[4, 4])
            fd.add_output(T1)

        # TODO: Currently, this check fails to produce a zero-element tensor whne the tensor
        # is smaller than the index range of the slize.  Therefore, it is disabled.
        # Issue: https://github.com/NVIDIA/Fuser/issues/52
        def legal(fd: FusionDefinition, acts) -> None:
            T0 = fd.from_pytorch(acts[0])
            T1 = fd.ops.slice(
                T0, start_indices=[6, 6], end_indices=[8, 8], strides=[1, 1]
            )
            fd.add_output(T1)

        checks = [
            (
                check_start_indices,
                "Slice operation start_indices must be greater than or equal to 0. .*",
            ),
            (
                check_end_indices,
                "Slice operation end_indices must be greater than or equal to start_indices. .*",
            ),
            (
                check_strides,
                "nvFuser Limitation: All slice operation strides must be of const size 1.*",
            ),
            (
                check_tensor_dims,
                "Number of tensor dimensions does not match slice dimensions! .*",
            ),
            (
                check_slice_dims_start,
                "Slice start_indices and strides don't match! .*",
            ),
            (
                check_slice_dims_end,
                "Slice indexing attribute dimensions don't match! .*",
            ),
            (
                check_slice_dims_stride,
                "Slice start_indices and strides don't match! .*",
            ),
            (check_nostrides, None),
            # (legal, None),
        ]

        first_check = True
        for inp in inputs:
            for check, error in checks:
                if error is None:
                    # First check is here on legal fusions since the second time
                    # through they should already be cached
                    out = self.exec_nvfuser(
                        partial(check, acts=inp),
                        inp,
                        new_fusion_expected=(first_check or debug_serde),
                    )
                else:
                    # When a fusion definition with errors is deserialized, it is recreated, triggering an error.
                    # skip_serde_check=True is necessary to skip these failing fusion definitions
                    # so serialization/deserialization does not exhibit the same errors in subsequent tests.
                    self.assertRaisesRegex(
                        RuntimeError,
                        error,
                        self.exec_nvfuser,
                        partial(check, acts=inp),
                        inp,
                        skip_serde_check=True,
                    )
            first_check = False

    def test_constant_nans(self):
        inputs = [
            torch.randn(4, 4, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_scalar(float("nan"))
            t1 = fd.ops.add(t0, c0)
            fd.add_output(t1)

        eager_out = inputs[0] + float("nan")

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        self.assertEqual(eager_out, nvf_out[0])

    def test_def_op_in_schedule(self):
        """
        Tests for an error when a definition op is used in a schedule
        """
        inputs = [
            torch.randn(4, 4, 4, device="cuda"),
        ]

        class SchedError(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.ops.tanh(self.t0)
                self.add_output(self.t1)

            def schedule(self):
                self.t2 = self.ops.relu(self.t1)

        with self.assertRaisesRegex(
            RuntimeError, "Attempting to add to a completed definition!"
        ):
            fd = SchedError()
            _ = fd.execute(inputs)

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="test_selected_device requires multiple GPUs",
    )
    def test_selected_device(self):
        """
        Run the Fusion on device 1
        """
        inputs = [
            torch.rand(2, 2, device="cuda:1"),
            torch.rand(2, 2, device="cuda:1"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.add(t0, t1)
            c0 = fd.define_scalar(1.0, DataType.Float)
            t3 = fd.ops.full(shape=[2, 2], fill_value=c0, dtype=DataType.Float)
            t4 = fd.ops.mul(t3, t2)
            fd.add_output(t4)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, device="cuda:1")
        eager_out = torch.full([2, 2], 1.0, device="cuda:1") * (inputs[0] + inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

        self.assertTrue(nvf_out[0].device.index == 1)

    def test_integer_division(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        self.assertEqual(
            nvf_out[0], torch.div(inputs[0], inputs[1], rounding_mode="trunc")
        )
        self.assertEqual(nvf_out[1], torch.true_divide(inputs[0], inputs[1]))

    def test_right_shift_arithmetic(self):
        inputs = [
            torch.tensor([-2147483648, 1073741824], dtype=torch.int32, device="cuda")
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_scalar(3)
            t1 = fd.ops.bitwise_right_shift(t0, c0)
            fd.add_output(t1)

        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.bitwise_right_shift(inputs[0], 3)
        self.assertEqual(eager_out, nvf_out1[0])

    def test_right_shift_logical(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, [current_input])
            self.assertEqual(nvf_out[0], expected_outputs[idx])

    def test_right_shift_logical_sizeof_dtype(self):
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
            expected_output = torch.zeros_like(current_input)

            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(current_input)
                c0 = fd.define_scalar(None, dtype=DataType.Int)
                t1 = fd.ops.logical_right_shift(t0, c0)
                fd.add_output(t1)

            nvf_out, _ = self.exec_nvfuser(fusion_func, [current_input, num_bits])
            self.assertEqual(nvf_out[0], expected_output)

    def test_gcd(self):
        inputs = [
            torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
            torch.testing.make_tensor(1024, device="cuda", dtype=torch.long),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.gcd(t0, t1)
            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        self.assertEqual(nvf_out[0], torch.gcd(inputs[0], inputs[1]))

    def test_input_scalar(self):
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
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    def test_debug_output(self):
        inputs = [
            torch.randn((3,), dtype=torch.float32, device="cuda:0"),
            0.1,
        ]

        with FusionDefinition() as fd:
            T0 = fd.from_pytorch(inputs[0])
            S1 = fd.define_scalar()
            T1 = fd.ops.div(T0, S1)
            fd.add_output(T1)

        out1 = fd.execute(inputs)
        self.assertIsNone(fd.debug_output())

        # If debug output is captured, getDebugOutput() will not return None.
        # The output will depend on the NVFUSER_DUMP environment variable in
        # such case
        out2 = fd.execute(inputs, capture_debug_output=True)
        self.assertIsNotNone(fd.debug_output())

    # Test that deterministic random ops (uniform, normal) give same results as
    # their stochastic versions
    def test_deterministic_random(self):
        input_size = [5, 9]
        dtype = torch.float32
        device = "cuda"
        inputs = [
            torch.randn(*input_size, device=device, dtype=dtype),
        ]

        for randopname in ["uniform", "normal"]:

            def fusion_func(fd: FusionDefinition, *, deterministic) -> None:
                t1 = fd.from_pytorch(inputs[0])
                a = fd.define_scalar(0.3, DataType.Float)
                b = fd.define_scalar(1.7, DataType.Float)
                randop = getattr(fd.ops, randopname)
                if deterministic:
                    rng_seed = fd.define_scalar(DataType.Int)
                    rng_offset = fd.define_scalar(DataType.Int)
                    u = randop(
                        a, b, shape=[5, 9], rng_seed=rng_seed, rng_offset=rng_offset
                    )
                else:
                    u = randop(a, b, shape=[5, 9])
                t2 = t1 * u
                fd.add_output(t2)

            # exec_nvfuser tests printing and serde, so run that for each definition first
            self.exec_nvfuser(partial(fusion_func, deterministic=False), inputs)
            self.exec_nvfuser(
                partial(fusion_func, deterministic=True), [inputs[0], 0, 0]
            )

            # Now instantiate FusionDefinitions in each mode
            with FusionDefinition() as fd_stoch:
                fusion_func(fd_stoch, deterministic=False)
            with FusionDefinition() as fd_det:
                fusion_func(fd_det, deterministic=True)

            # Get the current RNG state to restore after this test.
            state = torch.cuda.get_rng_state()
            # Test with three different random seeds
            for _ in range(3):
                max_seed = 2**63 - 1
                seed = random.randint(0, max_seed)
                torch.manual_seed(seed)

                stateful_sequence = [fd_stoch.execute(inputs) for _ in range(10)]
                # Each call to uniform with DataType::Float will advance the offset by one
                # See Note [Divide offset by 4] in rng.cpp for more information
                stateless_sequence = [
                    fd_det.execute([inputs[0], seed, rng_offset])
                    for rng_offset in range(10)
                ]

                for i, (sful, sless) in enumerate(
                    zip(stateful_sequence, stateless_sequence)
                ):
                    torch.testing.assert_close(sful[0], sless[0])
            # Restore the RNG state
            torch.cuda.set_rng_state(state)

    # Test expand to zero is replaced with expanded extent and not 1
    # see https://github.com/NVIDIA/Fuser/issues/603
    def test_expand_to_zero(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(nvf_out[0].shape, (0, 0))
        self.assertEqual(nvf_out[1].shape, (0, 0))

    # Test that a pad of an expanded empty tensor works properly
    # See https://github.com/NVIDIA/Fuser/issues/596#issuecomment-1714465618
    def test_pad_expanded_empty(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        torch_ref = F.pad(inputs[0], (0, 0, 1, 1, 1, 0), "constant", -3.70753)

        self.assertEqual(nvf_out[0], torch_ref)

    def test_dynamic_reshape(self):
        def dynamic_reshape(fd: FusionDefinition) -> None:
            x = fd.define_tensor([-1, -1], [True, True])
            d0 = fd.ops.size(x, 0)
            d1 = fd.define_scalar(dtype=DataType.Int32)
            d2 = fd.define_scalar(dtype=DataType.Int32)
            new_shape = fd.define_vector([d0, d1, d2])
            y = fd.ops.reshape(x, new_shape)
            fd.add_output(y)

        x = torch.rand(3, 4, device="cuda")
        ys, _ = self.exec_nvfuser(dynamic_reshape, [x, 2, 2])
        self.assertEqual(len(ys), 1)
        y = ys[0]

        self.assertEqual(y.shape, torch.Size([3, 2, 2]))
        self.assertEqual(x.flatten(), y.flatten())

    def test_allocation_domain_concretization(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_ref = inputs[0] * 2.0
        self.assertEqual(nvf_out[0], torch_ref)

    def test_allocation_domain_index_select(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_ref = torch.index_select(inputs[0], 1, inputs[1])
        self.assertEqual(nvf_out[0], torch_ref)

    # This tests that concretization will work properly with index_select
    def test_issue1129(self):
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
            V3 = fd.define_vector([S2], dtype=DataType.Int)
            T4 = fd.ops.reshape(T0, new_shape=V3)
            T5 = fd.ops.index_select(T1, T4, dim=0)
            S6 = fd.define_scalar(5, dtype=DataType.Int)
            S7 = fd.define_scalar(5, dtype=DataType.Int)
            S8 = fd.define_scalar(64, dtype=DataType.Int)
            V9 = fd.define_vector([S6, S7, S8], dtype=DataType.Int)
            T10 = fd.ops.reshape(T5, new_shape=V9)
            fd.add_output(T10)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_ref = torch.reshape(
            torch.index_select(inputs[1], 0, torch.reshape(inputs[0], [25])), [5, 5, 64]
        )
        self.assertEqual(nvf_out[0], torch_ref)

    # This test verifies aliases added by MarkAliasPass are still in effect
    # after serialization and deserialization.
    def test_mark_alias_pass(self):
        def reshape(fd: FusionDefinition) -> None:
            x = fd.define_tensor(
                [2, 3, 4], contiguity=[True, True, True], dtype=DataType.Float
            )
            y = fd.ops.reshape(x, [2, 12])
            fd.add_output(y)

        x = torch.rand(2, 3, 4, device="cuda")
        ys, _ = self.exec_nvfuser(reshape, [x])
        self.assertEqual(len(ys), 1)
        y = ys[0]

        self.assertEqual(y.data_ptr(), x.data_ptr())

    # Test that reshape to slice to sum with concrete sizes sets extents properly
    # https://github.com/NVIDIA/Fuser/issues/1221
    def test_sum_sliced_reshape_to_broadcast(self):
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
            V96 = fd.define_vector([S91, S92, S93, S94, S95], dtype=DataType.Int)
            T97 = fd.ops.reshape(T18, new_shape=V96)
            T98 = fd.ops.slice(
                T97,
                start_indices=[0, 0, 0, 0, 0],
                end_indices=[12, 128, 25, 32, 1],
                strides=[1, 1, 1, 1, 1],
            )
            T89 = fd.ops.sum(T98, dims=[4], keepdim=False, dtype=DataType.Null)
            fd.add_output(T89)

        # TODO Segmentation fails validateAllocationSizesAndStrides
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    # This tests no dead code at definition does not cause a problem due to
    # removal of empty tensors
    # See https://github.com/NVIDIA/Fuser/pull/1270
    def test_issue1270(self):
        inputs = [
            torch.randn(0, device="cuda", dtype=torch.bfloat16).as_strided(
                (5, 0), (1, 0)
            ),
            torch.randn(0, device="cuda", dtype=torch.bfloat16).as_strided(
                (5, 0), (0, 1)
            ),
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        t2 = inputs[1].type(torch.float32)
        t4 = torch.full([5, 0], 1.0, dtype=torch.bfloat16, device="cuda")
        t5 = t4.type(torch.float32)
        t6 = t2 * t5
        t7 = inputs[0].type(torch.float32)
        t8 = t7 * t5
        t24 = t6.sum([1])
        t11 = t8.sum([0])
        self.assertEqual(nvf_out[0], t24)
        self.assertEqual(nvf_out[1], t11)

    # This tests squeeze of dynamic input is handled properly
    def test_issue1273(self):
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
            T26 = fd.ops.broadcast_in_dim(
                T19, shape=[2, 1, 2], broadcast_dims=[0, 1, 2]
            )
            T27 = fd.ops.sub(T7, T26)
            T32 = fd.ops.broadcast_in_dim(
                T21, shape=[2, 1, 2], broadcast_dims=[0, 1, 2]
            )
            T33 = fd.ops.mul(T27, T32)
            T37 = fd.ops.reshape(T33, new_shape=[2, 2])
            fd.add_output(T37)

        # `supports_segmentation` is to work around #3856
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)
        t7 = inputs[0].reshape((2, 1, 2))
        t8 = t7.var(dim=2, unbiased=False)
        t9 = t7.mean(dim=2)
        t27 = t7 - t9.unsqueeze(-1).expand((2, 1, 2))
        t32 = torch.rsqrt(inputs[1] + t8.unsqueeze(-1)).expand((2, 1, 2))
        torch_ref = (t27 * t32).reshape((2, 2))
        self.assertEqual(nvf_out[0], torch_ref)

    # See https://github.com/NVIDIA/Fuser/issues/1246
    def test_issue1246(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
            torch_ref = torch.cat([2.0 * inputs[0], inputs[1]], dim=-1)
            self.assertEqual(nvf_out[0], torch_ref)

    # Test that inputs are properly forwarded when an input is used in multiple
    # UnaryOps, some having one and others having multiple further uses.
    # See https://github.com/NVIDIA/Fuser/issues/1301#issuecomment-1812470502
    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_issue1310(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        t14 = inputs[0].type(torch.float32)
        t16 = t14.sum([0, 1])
        t31 = t14.sum([2])
        self.assertEqual(nvf_out[0], t16)
        self.assertEqual(nvf_out[1], t16)  # T16 == T20
        self.assertEqual(nvf_out[2], t31)

    def test_issue1393(self):
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
            V10 = fd.define_vector([S7, S8, S9], dtype=DataType.Int)
            T11 = fd.ops.reshape(T6, new_shape=V10)
            S12 = fd.define_scalar(3, dtype=DataType.Int)
            S13 = fd.define_scalar(4, dtype=DataType.Int)
            S14 = fd.define_scalar(5, dtype=DataType.Int)
            V15 = fd.define_vector([S12, S13, S14], dtype=DataType.Int)
            T16 = fd.ops.broadcast_in_dim(T11, shape=V15, broadcast_dims=[0, 1, 2])
            T17 = fd.ops.cast(T16, dtype=DataType.Float)
            T18 = fd.ops.cast(T0, dtype=DataType.Float)
            T19 = fd.ops.mul(T17, T18)
            T20 = fd.ops.cast(T19, dtype=DataType.Half)
            fd.add_output(T20)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_ref = inputs[0] * (inputs[1] * inputs[2]).unsqueeze(-1)
        self.assertEqual(nvf_out[0], torch_ref)

    # We expect this to fail on branch `wjy/input` but to pass on ToT.
    def test_issue2755(self):
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
        self.exec_nvfuser(fusion_func, inputs)

    # Test that expand+pad does not cause indexing error, and that no scalars
    # are lost during segmentation.
    # See https://github.com/NVIDIA/Fuser/issues/1277
    def test_issue1277(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)
        # self.assertEqual(nvf_out[0], t24)

        # This fusion takes a long time to segment and schedule
        # because of the resized extents, which seem to stress the
        # expression simplifier a lot. Serializing this fusion would
        # significantly increase the test time as it would be
        # deserialized every time, which includes segmentation and
        # scheduling. Ideally, we should optimize the expression
        # simplifier, but for now resetting the cache should avoid the
        # issue.
        FusionCache.reset()

    # Test that symbolic IterDomains can be concatenated
    # https://github.com/NVIDIA/Fuser/issues/1554
    def test_cat_symbolic(self):
        inputs = [
            0.29730177875068026,
            0.29730177875068026,
            4,
            64,
            768,
            4,
            64,
            768,
            2,
            torch.randn([4, 6, 64, 128], dtype=torch.float32, device="cuda"),
            torch.randn([4, 6, 64, 128], dtype=torch.float32, device="cuda"),
            torch.randn([4, 64, 768], dtype=torch.float32, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            S0 = fd.define_scalar(None, dtype=DataType.Double)
            S1 = fd.define_scalar(None, dtype=DataType.Double)
            S2 = fd.define_scalar(None, dtype=DataType.Int)
            S3 = fd.define_scalar(None, dtype=DataType.Int)
            S4 = fd.define_scalar(None, dtype=DataType.Int)
            S5 = fd.define_scalar(None, dtype=DataType.Int)
            S6 = fd.define_scalar(None, dtype=DataType.Int)
            S7 = fd.define_scalar(None, dtype=DataType.Int)
            S8 = fd.define_scalar(None, dtype=DataType.Int)
            T9 = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T10 = fd.define_tensor(
                shape=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T11 = fd.define_tensor(
                shape=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T12 = fd.ops.mul(T10, S1)
            T13 = fd.ops.permute(T12, dims=[0, 1, 3, 2])
            T14 = fd.ops.mul(T9, S0)
            T15 = fd.ops.permute(T14, dims=[0, 2, 1, 3])
            S16 = fd.define_scalar(4, dtype=DataType.Int)
            S17 = fd.define_scalar(64, dtype=DataType.Int)
            S18 = fd.define_scalar(768, dtype=DataType.Int)
            V19 = fd.define_vector([S16, S17, S18], dtype=DataType.Int)
            T20 = fd.ops.reshape(T15, new_shape=V19)
            T21 = fd.ops.permute(T13, dims=[0, 2, 1, 3])
            S22 = fd.define_scalar(4, dtype=DataType.Int)
            S23 = fd.define_scalar(64, dtype=DataType.Int)
            S24 = fd.define_scalar(768, dtype=DataType.Int)
            V25 = fd.define_vector([S22, S23, S24], dtype=DataType.Int)
            T26 = fd.ops.reshape(T21, new_shape=V25)
            T27 = fd.ops.cat([T20, T26, T11], dim=2)
            T28 = fd.ops.sum(T27, [0, 1], keepdim=False, dtype=DataType.Null)
            fd.add_output(T27)
            fd.add_output(T28)

        # TODO: Support segmentation. See #3594.
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

        t12 = inputs[1] * inputs[-2]
        t13 = torch.permute(t12, [0, 1, 3, 2])
        t14 = inputs[0] * inputs[-3]
        t15 = torch.permute(t14, [0, 2, 1, 3])
        t20 = torch.reshape(t15, [4, 64, 768])
        t21 = torch.permute(t13, [0, 2, 1, 3])
        t26 = torch.reshape(t21, [4, 64, 768])
        t27 = torch.cat([t20, t26, inputs[-1]], dim=2)
        t28 = t27.sum([0, 1])

        torch.testing.assert_close(nvf_out[0], t27)
        torch.testing.assert_close(nvf_out[1], t28)

    # Test that trivial reshapes whose inputs are reductions are concretized
    # properly
    # See https://github.com/NVIDIA/Fuser/issues/1691
    def test_issue1691(self):
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
            V5 = fd.define_vector([S4], dtype=DataType.Int)
            T6 = fd.ops.reshape(T2, new_shape=V5)
            S7 = fd.define_scalar(4, dtype=DataType.Int)
            V8 = fd.define_vector([S7], dtype=DataType.Int)
            T9 = fd.ops.reshape(T3, new_shape=V8)
            T10 = fd.ops.mul(T6, T9)
            T11 = fd.ops.sum(T10, dims=[0], keepdim=False, dtype=DataType.Null)
            fd.add_output(T11)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        torch_ref = (inputs[0].sum(dim=[0, 1]) * inputs[1].sum(dim=1)).sum(dim=0)
        self.assertEqual(nvf_out[0], torch_ref)

    # Test that expanded dimensions can be reduced properly
    # See https://github.com/NVIDIA/Fuser/issues/1678
    def test_expanded_reduction(self):
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

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            self.assertEqual(nvf_out[0], inputs[0].sum(dim=0, keepdim=keepdim))

    def test_issue1872(self):
        def fusion_func(fd: FusionDefinition) -> None:
            S0 = fd.define_scalar(1.00000, dtype=DataType.Double)
            S1 = fd.define_scalar(5, dtype=DataType.Int)
            V2 = fd.define_vector([S1], dtype=DataType.Int)
            T3 = fd.ops.full(shape=V2, fill_value=S0, dtype=DataType.Float)
            T4 = fd.ops.slice(T3, start_indices=[0], end_indices=[2], strides=[1])
            T5 = fd.ops.cast(T4, dtype=DataType.Half)
            T6 = fd.ops.slice(T3, start_indices=[2], end_indices=[5], strides=[1])
            T7 = fd.ops.cast(T6, dtype=DataType.Half)
            fd.add_output(T5)
            fd.add_output(T7)

        self.exec_nvfuser(fusion_func, [])

    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_issue1706(self):
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
            V17 = fd.define_vector([S14, S15, S16], dtype=DataType.Int)
            T18 = fd.ops.broadcast_in_dim(T13, shape=V17, broadcast_dims=[0, 1, 2])
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
            V49 = fd.define_vector([S46, S47, S48], dtype=DataType.Int)
            T50 = fd.ops.broadcast_in_dim(T45, shape=V49, broadcast_dims=[1])
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
            V66 = fd.define_vector([S64, S65], dtype=DataType.Int)
            T67 = fd.ops.broadcast_in_dim(T63, shape=V66, broadcast_dims=[1])
            S68 = fd.define_scalar(1, dtype=DataType.Int)
            S69 = fd.define_scalar(4096, dtype=DataType.Int)
            S70 = fd.define_scalar(1, dtype=DataType.Int)
            V71 = fd.define_vector([S68, S69, S70], dtype=DataType.Int)
            T72 = fd.ops.broadcast_in_dim(T67, shape=V71, broadcast_dims=[0, 1])
            S73 = fd.define_scalar(1, dtype=DataType.Int)
            S74 = fd.define_scalar(4096, dtype=DataType.Int)
            S75 = fd.define_scalar(4096, dtype=DataType.Int)
            V76 = fd.define_vector([S73, S74, S75], dtype=DataType.Int)
            T77 = fd.ops.broadcast_in_dim(T72, shape=V76, broadcast_dims=[0, 1, 2])
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

        # check if serialization passes during segmentation
        # skip pytorch check because fusion is derived from llama2 network.
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    # https://github.com/NVIDIA/Fuser/issues/1953
    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_issue1953(self):
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
            V21 = fd.define_vector([S16, S17, S18, S19, S20], dtype=DataType.Int)
            T22 = fd.ops.reshape(T15, new_shape=V21)
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
            V41 = fd.define_vector([S36, S37, S38, S39, S40], dtype=DataType.Int)
            T42 = fd.ops.broadcast_in_dim(T32, shape=V41, broadcast_dims=[0, 1, 2, 3])
            S43 = fd.define_scalar(128, dtype=DataType.Int)
            S44 = fd.define_scalar(256, dtype=DataType.Int)
            S45 = fd.define_scalar(6, dtype=DataType.Int)
            S46 = fd.define_scalar(24, dtype=DataType.Int)
            S47 = fd.define_scalar(1, dtype=DataType.Int)
            V48 = fd.define_vector([S43, S44, S45, S46, S47], dtype=DataType.Int)
            T49 = fd.ops.broadcast_in_dim(T35, shape=V48, broadcast_dims=[0, 1, 2, 3])
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
            V65 = fd.define_vector([S60, S61, S62, S63, S64], dtype=DataType.Int)
            T66 = fd.ops.reshape(T59, new_shape=V65)
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
            V85 = fd.define_vector([S80, S81, S82, S83, S84], dtype=DataType.Int)
            T86 = fd.ops.broadcast_in_dim(T76, shape=V85, broadcast_dims=[0, 1, 2, 3])
            S87 = fd.define_scalar(128, dtype=DataType.Int)
            S88 = fd.define_scalar(256, dtype=DataType.Int)
            S89 = fd.define_scalar(6, dtype=DataType.Int)
            S90 = fd.define_scalar(24, dtype=DataType.Int)
            S91 = fd.define_scalar(1, dtype=DataType.Int)
            V92 = fd.define_vector([S87, S88, S89, S90, S91], dtype=DataType.Int)
            T93 = fd.ops.broadcast_in_dim(T79, shape=V92, broadcast_dims=[0, 1, 2, 3])
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    # A simple pointwise fusion, but passed misaligned input
    def test_misaligned_add(self):
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
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    # See https://github.com/NVIDIA/Fuser/issues/2275
    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_unpadded_catop_issue2275_repro1(self):
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
            V11 = fd.define_vector([S8, S9, S10], dtype=DataType.Int)
            T12 = fd.ops.broadcast_in_dim(T7, shape=V11, broadcast_dims=[0, 1])
            S13 = fd.define_scalar(4096.00, dtype=DataType.Double)
            S14 = fd.ops.reciprocal(S13)
            T15 = fd.ops.mul(T12, S14)
            S16 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
            T17 = fd.ops.add(T15, S16)
            T18 = fd.ops.rsqrt(T17)
            S19 = fd.define_scalar(2, dtype=DataType.Int)
            S20 = fd.define_scalar(4096, dtype=DataType.Int)
            S21 = fd.define_scalar(4096, dtype=DataType.Int)
            V22 = fd.define_vector([S19, S20, S21], dtype=DataType.Int)
            T23 = fd.ops.broadcast_in_dim(T18, shape=V22, broadcast_dims=[0, 1, 2])
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
            V34 = fd.define_vector([S29, S30, S31, S32, S33], dtype=DataType.Int)
            T35 = fd.ops.reshape(T28, new_shape=V34)
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
            V51 = fd.define_vector([S47, S48, S49, S50], dtype=DataType.Int)
            T52 = fd.ops.reshape(T37, new_shape=V51)
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    # See https://github.com/NVIDIA/Fuser/issues/2275
    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_unpadded_catop_issue2275_repro2(self):
        inputs = [
            torch.randn((2, 32, 4096, 128), dtype=torch.bfloat16, device="cuda:0")
        ]

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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    # See https://github.com/NVIDIA/Fuser/issues/2317
    @pytest.mark.skipif(
        is_pre_ampere(), reason="Only supported on Ampere and newer devices."
    )
    def test_reduction_transpose_sched_issue2317(self):
        inputs = [
            torch.randn((16, 25, 128, 64), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((16, 128, 1600), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((1600, 1600), dtype=torch.bfloat16, device="cuda:0"),
        ]

        def fusion_func(fd: FusionDefinition, inputs) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.from_pytorch(inputs[2])

            T10 = fd.ops.permute(T0, dims=[0, 2, 1, 3])
            T11 = fd.ops.stride_order(T10, stride_order=[3, 2, 1, 0])
            T16 = fd.ops.reshape(T11, new_shape=T1.shape())
            T17 = fd.ops.linear(T16, T2)
            T33 = fd.ops.add(T17, T1)

            T33 = fd.ops.cast(T33, dtype=DataType.BFloat16)
            T34 = fd.ops.linear(T33, T2)
            T35 = fd.ops.add(T34, T33)
            fd.add_output(T35)

        nvf_out, _ = self.exec_nvfuser(partial(fusion_func, inputs=inputs), inputs)

    @pytest.mark.skipif(reason="Temporarily disabled due to CUDA 13 compatibility.")
    def test_fusion_profiler(self):
        inputs = [
            torch.randn((2, 5), dtype=torch.float, device="cuda:0"),
            torch.randn((2, 5), dtype=torch.float, device="cuda:0"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.ops.add(T0, T1)
            T3 = fd.ops.sum(T2, dim=-1)
            T4 = fd.ops.sum(T3, dim=-1)
            fd.add_output(T4)

        with FusionDefinition() as fd:
            fusion_func(fd)

        # Testing returning a profile without profiling, expect an error!
        try:
            fd.profile()
            raise RuntimeError(
                "fd.profile() should have raised a ValueError because profile() was called before exeute()!"
            )
        except ValueError:
            pass

        # Testing that the profile returns 2 segments
        try:
            fd.execute(inputs, profile=True)
            prof = fd.profile()
            self.assertEqual(prof.segments, 2)
            self.assertEqual(len(prof.kernel_profiles), 2)
        except Exception as e:
            raise RuntimeError(
                "FusionDefinition's execute() did not run correctly with profile enabled!"
            )

    @pytest.mark.skipif(reason="Temporarily disabled due to CUDA 13 compatibility.")
    def test_fusion_profiler_user_schedule(self):
        inputs = [
            torch.randn((2, 5), dtype=torch.float, device="cuda:0"),
            torch.randn((2, 5), dtype=torch.float, device="cuda:0"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.ops.add(T0, T1)
            fd.add_output(T2)

        class MyFusion(FusionDefinition):
            def definition(self):
                fusion_func(fd)

            def schedule(self):
                pass

        fd = MyFusion()
        try:
            fd.execute(inputs, profile=True)
            self.assertTrue(fd.profile().fusion_id >= 0)
            self.assertEqual(fd.profile().kernel_profiles[0].scheduler, "user")
        except Exception as e:
            raise RuntimeError(
                "FusionDefinition's execute() did not run correctly with profile enabled!"
            )

    @pytest.mark.skipif(reason="Temporarily disabled due to CUDA 13 compatibility.")
    def test_fusion_profiler_with_noncodegen_kernels(self):
        inputs = [
            torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((16, 16), dtype=torch.bfloat16, device="cuda:0"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.from_pytorch(inputs[2])
            T3 = fd.ops.linear(T0, T2)
            T4 = fd.ops.add(T3, T1)
            fd.add_output(T4)

        class MyFusion(FusionDefinition):
            def definition(self):
                fusion_func(fd)

        fd = MyFusion()
        try:
            fd.execute(inputs, profile=True)
            self.assertTrue(fd.profile().fusion_id >= 0)
            self.assertEqual(len(fd.profile().kernel_profiles), 2)
            self.assertGreaterEqual(len(fd.profile().kernel_profiles[0].name), 0)
            self.assertGreaterEqual(len(fd.profile().kernel_profiles[1].name), 0)
        except Exception as e:
            raise RuntimeError(
                "FusionDefinition's execute() did not run correctly with profile enabled!"
            )

    # Small repro from https://github.com/NVIDIA/Fuser/issues/2359
    def test_reshape_squeeze_concretization(self):
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
            V4 = fd.define_vector([S2, S3], dtype=DataType.Int)
            V5 = fd.define_vector([S3], dtype=DataType.Int)
            T6 = fd.ops.reshape(T1, new_shape=V4)
            T7 = fd.ops.reshape(T6, new_shape=V5)
            # this works fine
            # T7 = fd.ops.reshape(T1, new_shape=V5)
            fd.add_output(T7)

        # TODO Segmentation fails validateAllocationSizesAndStrides
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

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
            V4 = fd.define_vector([S2, S3], dtype=DataType.Int)
            T5 = fd.ops.reshape(T0, new_shape=V4)
            fd.add_output(T5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    # Test that the range of generated uniform values spans the proper range
    # https://github.com/NVIDIA/Fuser/issues/1653
    def test_uniform_range(self):
        dtypes = [DataType.Double, DataType.Float, DataType.Half]
        if not is_pre_ampere():
            dtypes.append(DataType.BFloat16)

        def run_test(left: float, right: float, dtype: DataType):
            samples_per_run = 2**29

            def fusion_fn(fd: FusionDefinition):
                # Generate enough values to reasonably expect to sample the ends of the range
                shape = fd.define_vector([samples_per_run], dtype=DataType.Int)
                S0 = fd.define_scalar(left, dtype=DataType.Double)
                S1 = fd.define_scalar(right, dtype=DataType.Double)
                output = fd.ops.uniform(S0, S1, shape=shape, dtype=dtype)
                fd.add_output(output)

            with FusionDefinition() as fd:
                fusion_fn(fd)

            output = fd.execute([])[0]

            x = output.amax()
            m = output.amin()
            mu = output.type(torch.float64).mean()
            # Repeat to improve chances of sampling extreme values
            num_runs = 100
            num_samples = num_runs * samples_per_run
            for i in range(num_runs):
                u = fd.execute([])[0]
                x = torch.maximum(x, u.amax())
                m = torch.minimum(m, u.amin())
                mu = mu + (u.type(torch.float64).mean() - mu) / (i + 2)

            # round-trip cast to find expected min
            theomin = torch.tensor(left, dtype=output.dtype).item()
            theomu = 0.5 * (right + left)
            theomax = torch.nextafter(
                torch.tensor(right, dtype=output.dtype),
                torch.tensor(left, dtype=output.dtype),
            )

            assert (
                m.item() >= theomin
            ), f"{output.dtype} expected min generated value {theomin} but found {m.item()}"
            assert (
                m.item() <= theomax
            ), f"{output.dtype} expected max generated value {theomax} but found {x.item()}"

            # uniform distribution on [0, 1) has mean 0.5 and variance 1/12
            # The standard error of the mean is then 1/sqrt(12 *
            # num_samples). We use the precision at 1.0 as a surrogate for
            # the contribution of rounding to the standard error of the
            # finite-precision mean.
            assert abs(mu.item() - theomu) < (right - left) * max(
                right - x.item(), 3.0 / math.sqrt(12 * num_samples)
            ), f"{output.dtype} expected mean generated value {theomu} but found {mu.item()}"

            if dtype not in [DataType.Float, DataType.Double]:
                # For reduced precision types, check that we sample the extreme
                # values. We don't do this for full precision types since the
                # amount of samples required would be too large.
                assert (
                    m.item() == theomin
                ), f"{output.dtype} expected min generated value {theomin} but found {m.item()}"
                assert (
                    x.item() == theomax
                ), f"{output.dtype} expected max generated value {theomax} but found {x.item()}"

        # test standard and non-standard uniform
        for left, right in [[0.0, 1.0], [-1.5, 3.7]]:
            for dtype in dtypes:
                run_test(left, right, dtype)

    def test_random_distinct_values(self):
        dtypes = [DataType.Double, DataType.Float, DataType.Half]
        if not is_pre_ampere():
            dtypes.append(DataType.BFloat16)
        for dtype, randopname in itertools.product(dtypes, ["uniform", "normal"]):

            def fusion_fn(fd: FusionDefinition):
                # generate 4 values and check that they are all distinct
                shape = fd.define_vector([2, 2], dtype=DataType.Int)
                randop = getattr(fd.ops, randopname)
                S0 = fd.define_scalar(0.00000, dtype=DataType.Double)
                S1 = fd.define_scalar(1.00000, dtype=DataType.Double)
                output = randop(S0, S1, shape=shape, dtype=dtype)
                fd.add_output(output)

            with FusionDefinition() as fd:
                fusion_fn(fd)

            for i in range(100):
                output = fd.execute([])[0]

                # Rarely we might have a pair of matching lower precision
                # samples. However, it is extremely rare that we would have a
                # set of three matching elements in only 100 repeats unless we
                # have a bug.

                match = output.flatten().unsqueeze(0) == output.flatten().unsqueeze(1)
                match_pairs = (
                    match ^ torch.eye(4, dtype=torch.bool, device="cuda")
                ).sum() // 2

                assert (
                    match_pairs.item() < 3
                ), f"At least three entries match in {output}"

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
            V18 = fd.define_vector([S0, S17], dtype=DataType.Int)
            T19 = fd.ops.reshape(T1, new_shape=V18)
            T20 = fd.ops.sum(T19, dims=[1], keepdim=False, dtype=DataType.Null)
            fd.add_output(T20)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

    # Test that we do not hit segfaults when replacing an empty tensor that has multiple uses
    # https://github.com/NVIDIA/Fuser/issues/2545
    def test_remove_empty_issue_2545(self):
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

        # TODO: Support segmentation. See #3594.
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    def test_returning_aliased_outputs(self):
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        num_out = len(nvf_out)
        self.assertEqual(num_out, 3)
        for i in range(num_out):
            self.assertEqual(nvf_out[i].data_ptr(), inputs[0].data_ptr())

    # Test that we properly raise an error when passing inputs with the wrong types
    def test_mismatched_input_types(self):
        scalar_inp = 2.0
        tensor_inp = torch.rand((15,), dtype=torch.float32, device="cuda:0")

        def fusion_func(fd: FusionDefinition):
            T0 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[0],
            )
            s0 = fd.define_scalar()
            T1 = fd.ops.mul(T0, s0)
            fd.add_output(T1)

        with FusionDefinition() as fd:
            fusion_func(fd)

        try:
            import pytest
        except ImportError:
            self.skipTest("Could not import pytest")

        with pytest.raises(
            Exception,
            match="Expected input 0, .*, to be an at::Tensor but got scalar 2",
        ):
            nvf_out = fd.execute([scalar_inp, scalar_inp])

        with pytest.raises(
            Exception,
            match=re.escape(
                "Expected input 1, d2, to be a scalar but got float tensor of rank 1"
            ),
        ):
            nvf_out = fd.execute([tensor_inp, tensor_inp])

        with pytest.raises(
            Exception,
            match="Expected input 0, .*, to be bound to a tensor of dtype float, but got a tensor of dtype __half",
        ):
            wrong_tensor_inp = torch.rand((15,), dtype=torch.float16, device="cuda:0")
            nvf_out = fd.execute([wrong_tensor_inp, 2.0])

        with pytest.raises(
            Exception,
            match=re.escape(
                "Scalar value (2,1) is not compatible with the expected data type: double."
            ),
        ):
            nvf_out = fd.execute([tensor_inp, 2.0 + 1.0j])

    def test_repro_script_generation(self):
        expected_str = """
import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
    T2 = fd.ops.mul(T0, T1)
    fd.add_output(T2)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.testing.make_tensor((4, 4), dtype=torch.float32, device='cuda:0'),
    torch.testing.make_tensor((4, 4), dtype=torch.float32, device='cuda:0'),
]
fd.execute(inputs)
"""

        inputs = [
            torch.randn(4, 4, device="cuda:0"),
            torch.randn(4, 4, device="cuda:0"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.mul(t0, t1)
            fd.add_output(t2)

        with FusionDefinition() as fd:
            fusion_func(fd)

        # Strip white space before and after script
        expected_str = expected_str.strip()
        expected_lines = expected_str.split("\n")

        # Test fd.execute(inputs, print_repro=True)
        old_stdout = sys.stdout
        test_stdout = sys.stdout = io.StringIO()

        fd.execute(inputs, print_repro=True)

        sys.stdout = old_stdout

        def canonicalize(input):
            lines = None
            if isinstance(input, str):
                comp_str = input.strip()
                lines = comp_str.split("\n")
            else:
                lines = input
            new_lines = []
            # Enable skipping the repro header that might change due to number of
            # or software version.
            for line in lines:
                if re.match(r"^# .*", line):
                    continue
                elif re.fullmatch(
                    r"^def nvfuser_fusion_id\d+\(fd : FusionDefinition\) -> None :",
                    line,
                ):
                    new_lines += [
                        "def nvfuser_fusion_id0(fd : FusionDefinition) -> None :"
                    ]
                elif re.fullmatch(r"^    nvfuser_fusion_id\d+\(fd\)", line):
                    new_lines += ["    nvfuser_fusion_id0(fd)"]
                else:
                    new_lines += [line]
            return new_lines

        comp_lines = canonicalize(test_stdout.getvalue())

        assert len(expected_lines) == len(
            comp_lines
        ), "Reproduction script does not have the expected number of lines! Expected: {} Actual: {}".format(
            len(expected_lines), len(comp_lines)
        )
        # Don't compare the first 5 lines as they may be device and cuda specific
        assert (
            expected_lines == comp_lines
        ), "Reproduction script does not match expected output!"

        # Test fd.execute(inputs, save_repro_inputs=True) ; fd.last_repro_script()
        fd.execute(inputs, save_repro_inputs=True)

        comp_lines = canonicalize(fd.last_repro_script())

        assert len(expected_lines) == len(
            comp_lines
        ), "Reproduction script does not have the expected number of lines! Expected: {} Actual: {}".format(
            len(expected_lines), len(comp_lines)
        )
        assert (
            expected_lines == comp_lines
        ), "Reproduction script does not match expected output!"

        comp_lines = canonicalize(fd.last_repro_script())

        # Make sure that the fake_tensors are properly saved for repeated query
        assert len(expected_lines) == len(
            comp_lines
        ), "The second query of the eproduction script does not have the expected number of lines! Expected: {} Actual: {}".format(
            len(expected_lines), len(comp_lines)
        )
        assert (
            expected_lines == comp_lines
        ), "The second query of the reproduction script does not match expected output!"

    # Test that replaced sizes using input tensor metadata are successfully computed
    # See https://github.com/NVIDIA/Fuser/pull/2714 which surfaced this in
    # failing thunder test
    # thunder.tests.test_core.test_bsym_toposort_nvfuser_cuda_thunder.dtypes.float32
    def test_replaced_sizes_pr2714(self):
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
            V5 = fd.define_vector([S4], dtype=DataType.Int)
            T6 = fd.ops.reshape(T2, new_shape=V5)
            S7 = fd.define_scalar(4, dtype=DataType.Int)
            V8 = fd.define_vector([S7], dtype=DataType.Int)
            T9 = fd.ops.reshape(T3, new_shape=V8)
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

        self.exec_nvfuser(fusion_func, inputs)

    # testing that error thrown in finalizeDefinition is not accidentally cached as legit fusion.
    def test_fusion_definition_error_cache(self):
        def fusion_func(fd: FusionDefinition) -> None:
            # NOTE: it's important that the exception is thrown inside FusionDefinition::finalizeDefinition()
            # e.g. https://github.com/NVIDIA/Fuser/blob/adbbc75e58e6c53c606e90c8bc64f020b4b9df85/csrc/python_frontend/fusion_record.h#L1276
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.Int,
                is_cpu=True,
                stride_order=[1, 0],
            )

        for i in range(2):
            with pytest.raises(
                Exception,
                match="CPU non-scalar tensor is not supported!",
            ):
                with FusionDefinition() as fd:
                    fusion_func(fd)

    def test_slice_api(self):
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
            V_start = fd.define_vector(offset)
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

    def test_fusion_information(self):
        inputs = [
            torch.ones(2, 4, 8, device="cuda"),
            torch.ones(2, 4, 8, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            c2 = fd.define_scalar(3.0)

            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.mul(t3, c2)
            t5 = fd.ops.sum(t4, [-1], False, DataType.Float)

            fd.add_output(t5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

        with FusionDefinition() as fd:
            fusion_func(fd)

        nvf_out1 = fd.execute(inputs)
        self.assertEqual(eager_out, nvf_out1[0])

        # The input tensors are t0 and t1.
        self.assertEqual(fd.inputs(), [0, 1])
        # The output tensors is t5.
        self.assertEqual(fd.outputs(), [5])
        # The extents correspond with the dimensions for each input tensor.
        # There are two input tensors with three dimensions each, so the
        # extents range from [-1, -6].
        self.assertEqual(fd.extents(), [idx for idx in range(-1, -7, -1)])

    def test_issue3292(self):
        inputs = [
            torch.testing.make_tensor(
                (5, 5, 576), dtype=torch.float32, device="cuda:0"
            ),
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

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, supports_segmentation=False)

    def test_enable_disable_options(self):
        m = 24
        n = 16
        k = 8
        inps = [
            torch.randn(m, k, device="cuda", dtype=torch.float),
            torch.randn(k, n, device="cuda", dtype=torch.float),
        ]

        def fusion_func(fd: FusionDefinition, inps) -> None:
            t0 = fd.from_pytorch(inps[0])
            t1 = fd.from_pytorch(inps[1])
            t2 = fd.ops.matmul(t0, t1)
            fd.add_output(t2)

        with FusionDefinition() as fd:
            fusion_func(fd, inps=inps)

        # By default, matmul will be be run through expr_eval scheduler.
        # Through setting the enable and disable options as below,
        # we can execute it through matmul scheduler. The above fusion will not
        # be accepted by the matmul scheduler since the outputs are of type Float and raises a RuntimeError.
        # Note: We use this error-based test since for compatible dtypes (float16/bfloat16),
        # the matmul scheduler ran into a scheduling error on H100. This test might be more robust against
        # changes in matmul scheduler in the interim.

        self.assertRaisesRegex(
            RuntimeError,
            "Can not find a scheduler to schedule fusion segment",
            self.exec_nvfuser,
            partial(fusion_func, inps=inps),
            inps,
            _enable_options=["fuse_matmul"],
            _disable_options=["matmul_expr_eval"],
            skip_serde_check=True,
        )

        # Serializing error test cases corrupts the serialized binary causing subsequent tests to fail.
        # Reset the fusion cache to avoid this.
        FusionCache.reset()

    def test_issue1279(self):
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
            T5, T6 = fd.ops.var_mean(T4, dims=[1], correction=1, keepdim=False)
            T7 = fd.ops.cast(T5, dtype=DataType.Half)
            T8 = fd.ops.cast(T6, dtype=DataType.Half)
            fd.add_output(T7)
            fd.add_output(T8)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        a = inputs[0].type(torch.float32)
        b, c = torch.var_mean(a, dim=1)
        d = b.type(torch.float16)
        e = c.type(torch.float16)

        self.assertEqual(nvf_out[0], d)
        self.assertEqual(nvf_out[1], e)

    # See https://github.com/NVIDIA/Fuser/issues/3833
    def test_bcast_squeeze_replace_aliased_output(self):
        inputs = [
            torch.testing.make_tensor(
                (1, 1, 576), dtype=torch.bfloat16, device="cuda:0"
            ),
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

        nvf_out, _ = self.exec_nvfuser(
            fusion_func,
            inputs,
            skip_serde_check=True,
        )

        assert len(nvf_out) == 1
        self.assertEqual(nvf_out[0], inputs[0].squeeze(1))

    # See https://github.com/NVIDIA/Fuser/issues/3957
    # This test checks that our alias update keeps the output layout consistency
    def test_inplace_update_on_non_contiguous_inputs(self):
        inputs = [
            torch.randn(5, dtype=torch.float32, device="cuda:0").as_strided(
                (2, 2), (1, 3)
            ),
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

        # is_clonable is not supported yet, because the print out would explicitly mark output
        # with stride_order and overwrite its contiguity flag. This would violates the memory
        # layout required by the alias.
        nvf_out, _ = self.exec_nvfuser(
            fusion_func,
            inputs,
            is_clonable=False,
        )

        assert len(nvf_out) == 1
        self.assertEqual(nvf_out[0], inputs[0])
        self.assertEqual(nvf_out[0], ref_inp.relu())

    def test_import_conflict_nvfuser_then_direct(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import nvfuser  # noqa: F401
            import nvfuser_direct  # noqa: F401

            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "Be careful! You've imported nvfuser_direct when the nvfuser module is already imported."
                in str(w[-1].message)
            )


@pytest.mark.skip("https://github.com/NVIDIA/Fuser/issues/3740")
def test_cat_qwen2_v2():
    def qwen2_cat_fusion_2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2048, 512],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        T3 = fd.define_tensor(
            shape=[1, 4, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T5 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T6 = fd.define_tensor(
            shape=[1, 28, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 2048, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
        T13 = fd.ops.cast(T12, dtype=DataType.Float)
        S14 = fd.define_scalar(0.00000, dtype=DataType.Double)
        S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
        S16 = fd.define_scalar(1, dtype=DataType.Int)
        S17 = fd.define_scalar(2048, dtype=DataType.Int)
        S18 = fd.define_scalar(512, dtype=DataType.Int)
        T20 = fd.ops.uniform(
            S14,
            S15,
            shape=[S16, S17, S18],
            rng_seed=S2,
            rng_offset=S1,
            dtype=DataType.BFloat16,
        )
        S21 = fd.define_scalar(4.00000, dtype=DataType.Double)
        T22 = fd.ops.mul(T13, S21)
        S23 = fd.define_scalar(0.900000, dtype=DataType.Double)
        T24 = fd.ops.lt(T20, S23)
        T25 = fd.ops.cast(T24, dtype=DataType.Float)
        T41 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T42 = fd.ops.mul(T22, T25)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.neg(T43)
        T50 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T66 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T67 = fd.ops.cast(T44, dtype=DataType.BFloat16)
        T73 = fd.ops.broadcast_in_dim(
            T5, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T89 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 28, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        S90 = fd.define_scalar(1.11111, dtype=DataType.Double)
        T91 = fd.ops.mul(T42, S90)
        T97 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T98 = fd.ops.cat([T67, T66], dim=-1, manual_padding=0)
        T104 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T105 = fd.ops.cast(T89, dtype=DataType.Float)
        T106 = fd.ops.cast(T97, dtype=DataType.Float)
        T107 = fd.ops.cast(T98, dtype=DataType.Float)
        T108 = fd.ops.cast(T104, dtype=DataType.Float)
        T109 = fd.ops.cast(T3, dtype=DataType.Float)
        T110 = fd.ops.neg(T105)
        T111 = fd.ops.cast(T7, dtype=DataType.Float)
        T112 = fd.ops.mul(T107, T106)
        T113 = fd.ops.mul(T109, T108)
        T129 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 28, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T130 = fd.ops.cast(T110, dtype=DataType.BFloat16)
        T131 = fd.ops.add(T111, T91)
        T137 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T138 = fd.ops.cat([T130, T129], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
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
        T167 = fd.ops.broadcast_in_dim(
            T159, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T174 = fd.ops.broadcast_in_dim(
            T160, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T181 = fd.ops.broadcast_in_dim(
            T167, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T188 = fd.ops.broadcast_in_dim(
            T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
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

    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device="cuda:0"),
        25546,
        1400552702872758,
        torch.randn(1048576, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 4, 2048, 128), (1048576, 128, 512, 1)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(7340032, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 28, 2048, 128), (7340032, 128, 3584, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 512), dtype=torch.bfloat16, device="cuda:0"
        ),
    ]

    with FusionDefinition() as fd:
        qwen2_cat_fusion_2(fd)

    fd.execute(inputs)
