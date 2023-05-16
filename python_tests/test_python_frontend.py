# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from copy import deepcopy
from functools import partial
import math
import os
import re
from typing import List, Callable
import unittest
import itertools

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA
import torch._refs as refs
import torch._prims as prims

# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser import (
        FusionCache,
        FusionDefinition,
        DataType,
        Tensor,
        version,
        compute_contiguity,
    )
    from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
except ImportError:
    pass

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def serde_check(test_fn: Callable):
    """
    A decorator to verify that serialization works with the given exec_nvfuser function.
    Currently, it uses serialization to rebuild the FusionCache structure.
    """

    def inner(*args, **kwargs):
        self, fusion_func, inputs = args
        # Deep copy inputs because when a fusion output aliases an input, it will change the input value for the
        # subsequent function calls.
        inputs_copy = deepcopy(inputs)

        # For debug purposes, clear FusionCache before running first test
        # if ("new_fusion_expected" not in kwargs) or kwargs["new_fusion_expected"]:
        #    FusionCache.reset()

        # Run test to populate FusionCache
        test_fn(*args, **kwargs)

        # Delete previous file
        if os.path.isfile("foo.bin"):
            os.remove("foo.bin")

        # Serialize FusionCache
        fc = FusionCache.get()
        fc.serialize("foo.bin")

        FusionCache.reset()

        # Get new FusionCache because the previous one was destroyed by the reset call.
        fc = FusionCache.get()
        fc.deserialize("foo.bin")

        # Run test with repopulated FusionCache
        kwargs["new_fusion_expected"] = False
        return test_fn(self, fusion_func, inputs_copy, **kwargs)

    return inner


@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestNvFuserFrontend(TestCase):
    # Helper function to verify the nvfuser output and make sure the string
    # definition based on the FusionDefinition is executable and matches the
    # original definition
    @serde_check
    def exec_nvfuser(self, fusion_func, inputs, *, new_fusion_expected=True):
        inputs_cap = deepcopy(inputs)
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()

        # Execute a fusion function and capture the string python definition
        with FusionDefinition() as fd:
            fusion_func(fd)
        fd_str = fd.__repr__()
        torch.manual_seed(0)
        out = fd.execute(inputs)

        # Execute the python definition that was captured
        func_name = re.findall("(nvfuser_fusion_id\\d+)", fd_str.split("\n")[1])[0]
        exec(fd_str)
        with FusionDefinition() as fd_cap:
            eval(func_name)(fd_cap)
        torch.manual_seed(0)
        out_cap = fd_cap.execute(inputs_cap)

        # Make sure the original and captured definitions match
        for idx in range(len(out)):
            self.assertEqual(out[idx], out_cap[idx])

        self.assertEqual(fc.num_fusions() - before_fusions, int(new_fusion_expected))
        return out, fd

    def test_basic(self):
        inputs = [
            torch.ones(2, 4, 8, device="cuda"),
            torch.ones(2, 4, 8, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            c0 = fd.define_constant(3.0)

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
            c0 = fd.define_constant(3.0)

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
                symbolic_sizes=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
            )
            weights = fd.define_tensor(
                symbolic_sizes=[-1], contiguity=[True], dtype=DataType.Float
            )
            bias = fd.define_tensor(
                symbolic_sizes=[-1], contiguity=[True], dtype=DataType.Float
            )
            sum0 = fd.ops.sum(inputs, axes=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_constant(norm_size)
            mean = fd.ops.div(sum0, norm_const)
            diff = fd.ops.sub(inputs, mean)
            diff_sq = fd.ops.mul(diff, diff)
            sum1 = fd.ops.sum(diff_sq, axes=[normalization_axis], keepdim=keepDim)
            var = fd.ops.div(sum1, norm_const)
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, output_shape=input_shape, broadcast_dims=[2]
            )
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(
                bias, output_shape=input_shape, broadcast_dims=[2]
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
                symbolic_sizes=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
            )
            weights = fd.define_tensor(
                symbolic_sizes=[-1], contiguity=[True], dtype=DataType.Float
            )
            bias = fd.define_tensor(
                symbolic_sizes=[-1], contiguity=[True], dtype=DataType.Float
            )
            var, mean = fd.ops.var_mean(
                inputs, axes=[normalization_axis], correction=0, keepdim=keepDim
            )
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            diff = fd.ops.sub(inputs, mean)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, output_shape=input_shape, broadcast_dims=[2]
            )
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(
                bias, output_shape=input_shape, broadcast_dims=[2]
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
                symbolic_sizes=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
            )
            weights = fd.define_tensor(
                symbolic_sizes=[-1], contiguity=[True], dtype=DataType.Float
            )
            inputs_sq = fd.ops.mul(inputs, inputs)
            sum0 = fd.ops.sum(inputs_sq, axes=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_constant(norm_size)
            var = fd.ops.div(sum0, norm_const)
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale = fd.ops.mul(inputs, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(
                weights, output_shape=input_shape, broadcast_dims=[2]
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

    # Testing a scenario where a broadcast requires a symbolic output shape
    def test_tensor_sizes(self):
        inputs = [
            torch.randn(2, 3, 4, device="cuda"),
            torch.randn(4, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [2])
            t2 = fd.ops.sub(t0, t1_b)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.sub(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where no broadcast is needed
    def test_tensor_sizes_nobcast(self):
        inputs = [
            torch.randn(2, 3, device="cuda"),
            torch.randn(2, 3, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(
            inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1])
        )
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where each arg of a binary op has broadcast.
    def test_tensor_sizes_both_args_bcast(self):
        inputs = [
            torch.randn(1, 3, device="cuda"),
            torch.randn(2, 1, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)
            t1_sizes = fd.ops.tensor_sizes(t1)

            t0_b = fd.ops.broadcast_in_dim(t0, [t1_sizes[0], t0_sizes[1]], [0, 1])
            t1_b = fd.ops.broadcast_in_dim(t1, [t1_sizes[0], t0_sizes[1]], [0, 1])
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
            t0 = fd.define_tensor(
                symbolic_sizes=[-1, -1, -1], contiguity=[True, True, True]
            )
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguity=[True])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_2(fd: FusionDefinition):
            t0 = fd.define_tensor(
                symbolic_sizes=[-1, -1, -1], contiguity=[True, True, True]
            )
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguity=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_1[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_3(fd: FusionDefinition):
            t0 = fd.define_tensor(
                symbolic_sizes=[-1, -1, -1], contiguity=[True, True, True]
            )
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguity=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_2[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        # Func_1 uses tensor_sizes to propagate dynamic size, therefore, it is
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
    def test_tensor_sizes_with_output_bcast(self):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.define_tensor(
                symbolic_sizes=[-1, -1, -1], contiguity=[True, True, True]
            )
            t0_sizes = fd.ops.tensor_sizes(t0)

            t1 = fd.ops.sum(t0, axes=[2])
            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1])

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
    def test_tensor_sizes_expand_bcast(self):
        def fusion_func(fd: FusionDefinition):
            t0 = fd.define_tensor(
                symbolic_sizes=[-1, -1, -1], contiguity=[True, True, True]
            )
            t1 = fd.define_tensor(
                symbolic_sizes=[-1, 1, -1], contiguity=[True, None, True]
            )
            t2 = fd.define_tensor(
                symbolic_sizes=[-1, 1, -1], contiguity=[True, None, True]
            )
            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1, 2])
            t1_b_sizes = fd.ops.tensor_sizes(t1_b)
            t2_b = fd.ops.broadcast_in_dim(t2, t1_b_sizes, [0, 1, 2])

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
        inputs = [
            torch.ones(4, 4, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            s0 = fd.define_constant(1.0)
            s1 = fd.define_constant(2.0)
            s2 = fd.define_constant(3.0)
            t1 = fd.ops.add(t0, s0)
            t2 = fd.ops.add(t0, s1)
            t3 = fd.ops.add(t2, s2)
            fd.add_output(t1)
            fd.add_output(t2, alias_input=t0)
            fd.add_output(t3)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out1 = torch.add(torch.ones(4, 4, device="cuda"), 1.0)
        eager_out2 = torch.add(torch.ones(4, 4, device="cuda"), 2.0)
        eager_out3 = torch.add(eager_out2, 3.0)
        self.assertEqual(eager_out1, nvf_out[0])
        self.assertEqual(eager_out2, inputs[0])
        self.assertEqual(eager_out3, nvf_out[1])

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
            t0 = fd.define_tensor(symbolic_sizes=[-1], contiguity=[True])
            t1 = fd.define_tensor(sizes=t1_sizes, strides=[4, 1, 1])
            t2 = fd.define_tensor(sizes=t2_sizes, strides=[4, 4, 1])
            t3 = fd.ops.squeeze(t1, t1_sizes, [0, -1])
            t4 = fd.ops.squeeze(
                t2,
                t2_sizes,
                [
                    -2,
                ],
            )
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
                symbolic_sizes=[0, 0], contiguity=[True, True], dtype=DataType.Float
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
            s_mean = fd.define_constant(mean)
            s_std = fd.define_constant(std)
            size = fd.ops.tensor_sizes(t0)
            t1 = fd.ops.normal(s_mean, s_std, size, DataType.Double)
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
            s_lo = fd.define_constant(lo)
            s_hi = fd.define_constant(hi)
            size = fd.ops.tensor_sizes(t0)
            t1 = fd.ops.uniform(s_lo, s_hi, size, DataType.Double)
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

            c0 = fd.define_constant(3.0)
            c1 = fd.define_constant(5.0)
            t1 = fd.ops.where(t0, c0, c1)  # DataType.Double
            fd.add_output(t1)

            c0f = fd.define_constant(3.0, DataType.Float)
            c1f = fd.define_constant(5.0, DataType.Float)
            t1f = fd.ops.where(t0, c0f, c1f)  # DataType.Float
            fd.add_output(t1f)

            c0d = fd.define_constant(3.0, DataType.Double)
            c1d = fd.define_constant(5.0, DataType.Double)
            t1d = fd.ops.where(t0, c0d, c1d)  # DataType.Double
            fd.add_output(t1d)

            c0i = fd.define_constant(3, DataType.Int32)
            c1i = fd.define_constant(5, DataType.Int32)
            t1i = fd.ops.where(t0, c0i, c1i)  # DataType.Int32
            fd.add_output(t1i)

            c0l = fd.define_constant(3)
            c1l = fd.define_constant(5)
            t1l = fd.ops.where(t0, c0l, c1l)  # DataType.Int
            fd.add_output(t1l)

            c0c = fd.define_constant(complex(3.0))
            c1c = fd.define_constant(complex(5.0))
            t1c = fd.ops.where(t0, c0c, c1c)  # DataType.ComplexDouble
            fd.add_output(t1c)

            c0cf = fd.define_constant(3.0 + 0j, DataType.ComplexFloat)
            c1cf = fd.define_constant(5.0 + 0j, DataType.ComplexFloat)
            t1cf = fd.ops.where(t0, c0cf, c1cf)  # DataType.ComplexFloat
            fd.add_output(t1cf)

            c0cd = fd.define_constant(3.0 + 0j, DataType.ComplexDouble)
            c1cd = fd.define_constant(5.0 + 0j, DataType.ComplexDouble)
            t1cd = fd.ops.where(t0, c0cd, c1cd)  # DataType.ComplexDouble
            fd.add_output(t1cd)

            c0b = fd.define_constant(True, DataType.Bool)
            c1b = fd.define_constant(False, DataType.Bool)
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
            c0 = fd.define_constant(complex(3.0, 0.5))
            t1 = fd.ops.mul(t0, c0)
            fd.add_output(t1)

        (n,), _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = inputs[0] * (3.0 + 0.5j)

        self.assertEqual(eager_out, n)
        assert n.dtype == torch.complex64

    def test_where_op(self):
        def nvfuser_where(pred, a, b):
            with FusionDefinition() as fd:
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
            return fd.execute((pred, a, b))[0]

        pred = torch.testing.make_tensor((5,), device="cuda", dtype=torch.bool)
        list_of_dtype = [torch.float16, torch.bfloat16, torch.float32]
        for atype in list_of_dtype:
            for btype in list_of_dtype:
                a = torch.randn((5,), device="cuda", dtype=atype)
                b = torch.randn((5,), device="cuda", dtype=btype)
                nv_result = nvfuser_where(pred, a, b)
                torch_result = torch.where(pred, a, b)
                self.assertEqual(nv_result, torch_result)

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
                c0 = fd.define_constant(input[0])
                c1 = None if input[1] is None else fd.define_constant(input[1])
                c2 = None if input[2] is None else fd.define_constant(input[2])
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

    def test_shape(self):
        inputs = [
            # test with both one and multiple dimensions
            torch.randn(3, device="cuda", dtype=torch.float32),
            torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            assert isinstance(t0._get_fusion_definition(), FusionDefinition)

            (B,) = t0.shape
            t2 = fd.ops.mul(t0, B)

            assert len(t1.shape) == t1.ndim

            B, C, W = t1.shape
            t3 = fd.ops.div(t1, C)

            fd.add_output(t2)
            fd.add_output(t3)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        at_out1 = inputs[0] * inputs[0].shape[0]
        at_out2 = inputs[1] / inputs[1].shape[1]

        self.assertEqual(at_out1, nvf_out[0])
        self.assertEqual(at_out2, nvf_out[1])

    def test_arithmetic_ops(self):
        inputs = [
            torch.randn(3, 4, 5, device="cuda", dtype=torch.float32),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])

            c0 = fd.define_constant(1.0)

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

    def test_all_dim_var_mean(self):
        inputs = [torch.randn(2, 2, 2, device="cuda")]

        def fuser_function(correction):
            with FusionDefinition() as fd:
                t0 = fd.from_pytorch(inputs[0])
                t1, t2 = fd.ops.var_mean(t0, [0, 1, 2], correction)
                fd.add_output(t1)
                fd.add_output(t2)
            return fd.execute(inputs)

        list_of_test_cases = [0, 1]
        for correction in list_of_test_cases:
            fuser_result = fuser_function(correction)
            torch_result = torch.var_mean(inputs[0], [0, 1, 2], bool(correction))
            self.assertEqual(fuser_result, torch_result)

    def test_scalar_only_inputs(self):
        # We don't allow scalar outputs, currently,
        # so a tensor has to be returned
        def fusion_func(fd: FusionDefinition):
            s0 = fd.define_scalar()
            s1 = fd.define_scalar()
            s2 = fd.ops.add(s0, s1)
            c0 = fd.define_constant(1.0, DataType.Float)
            t3 = fd.ops.full(size=[2, 2], arg=c0, dtype=DataType.Float)
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
            c0 = fd.define_constant(0.1)

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

            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                c0 = fd.define_constant(3.0)
                t1 = fd.ops.add(t0, c0)
                fd.add_output(t1, perm)

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
            self.assertEqual(eager_out, nvf_out[0])

            nvf_stride = nvf_out[0].stride()
            sorted_stride = list(nvf_stride)
            for idx, axis in enumerate(perm):
                sorted_stride[axis] = nvf_stride[idx]
            self.assertTrue(sorted(sorted_stride, reverse=True) == sorted_stride)

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
            T2 = fd.ops.broadcast_in_dim(T0, output_shape=[4, 4], broadcast_dims=[0, 1])
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
            t1 = ctx_seg_fusion.ops.sum(t0, axis=0)
            t2 = ctx_seg_fusion.ops.sum(t0, axis=-1)
            ctx_seg_fusion.add_output(t1)
            ctx_seg_fusion.add_output(t2)

        test_defs = [DefFuncFusion(), UserSchedFusion(), ctx_fusion, ctx_seg_fusion]

        for fd in test_defs:
            # Attempting to get the cuda code for an un-executed FusionDefinition
            # should trigger a RuntimeError and not a segfault
            with self.assertRaisesRegex(RuntimeError, "Invalid fusion definition!"):
                _ = fd.last_cuda_code()
            with self.assertRaisesRegex(RuntimeError, "Invalid fusion definition!"):
                _ = fd.last_scheduled_fusion_ir()
            # Only make this check for function based definitions
            if hasattr(super(type(self), self), "definition"):
                with self.assertRaisesRegex(RuntimeError, "Invalid fusion definition!"):
                    _ = fd.fusion_ir()

            _ = fd.execute(inputs)

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

            # Attemp to get strings for inputs that do not heuristically match
            # and a new fusion has not been compiled
            with self.assertRaisesRegex(RuntimeError, "Fusion is not compiled!"):
                _ = fd.cuda_code_for(big_inputs)
            with self.assertRaisesRegex(RuntimeError, "Fusion is not compiled!"):
                _ = fd.scheduled_fusion_ir_for(big_inputs)

    def test_pad(self):
        inputs = [
            torch.testing.make_tensor((2, 3), dtype=torch.float32, device="cuda"),
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
            fill_val = fd.define_constant(2.0)
            t5 = fd.ops.pad(t0, [2, 3], fill_val)
            fd.add_output(t5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        self.assertEqual(F.pad(inputs[0], [1, 1, 1, 1]), nvf_out[0])
        self.assertEqual(F.pad(inputs[0], [0, 0, 2, 3]), nvf_out[1])
        self.assertEqual(F.pad(inputs[0], [0, 0, 0, 0]), nvf_out[2])
        self.assertEqual(F.pad(inputs[0], [2, 3]), nvf_out[3])
        self.assertEqual(F.pad(inputs[0], [2, 3], "constant", 2.0), nvf_out[4])

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
            fill_val = fd.define_constant(2.0)
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

    def test_nextafter(self):
        inputs = [
            # torch.nextafter is only defined for float{32,64} tensor inputs
            torch.testing.make_tensor(4, device="cuda", dtype=torch.float32),
            torch.testing.make_tensor(4, device="cuda", dtype=torch.float64),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            s0 = fd.define_constant(1.0, dtype=DataType.Float)
            s1 = fd.define_constant(-1.0, dtype=DataType.Double)

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
                symbolic_sizes=[-1, -1, -1, -1],
                contiguity=[True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            T1 = fd.define_tensor(
                symbolic_sizes=[1, 1, -1, -1],
                contiguity=[None, None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )
            S2 = fd.define_constant(0.125000, dtype=DataType.Double)
            T3 = fd.ops.mul(T0, S2)
            T4 = fd.ops.slice(
                T1,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 1, 128, 128],
                strides=[1, 1, 1, 1],
            )
            S5 = fd.define_constant(0.00000, dtype=DataType.Double)
            T6 = fd.ops.eq(S5, T4)
            T7 = fd.ops.broadcast_in_dim(
                T6, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            S8 = fd.define_constant(float("-inf"), dtype=DataType.Double)
            T9 = fd.ops.where(T7, S8, T3)
            S10 = fd.define_constant(-1, dtype=DataType.Int)
            S11 = fd.define_constant(4, dtype=DataType.Int)
            S12 = fd.ops.add(S10, S11)
            T13 = fd.ops.max(T9, axes=[3], keepdim=False, dtype=DataType.Null)
            T14 = fd.ops.broadcast_in_dim(
                T13, output_shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
            )
            T15 = fd.ops.broadcast_in_dim(
                T14, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T16 = fd.ops.sub(T9, T15)
            T17 = fd.ops.exp(T16)
            S18 = fd.define_constant(-1, dtype=DataType.Int)
            S19 = fd.define_constant(4, dtype=DataType.Int)
            S20 = fd.ops.add(S18, S19)
            T21 = fd.ops.sum(T17, axes=[3], keepdim=False, dtype=DataType.Null)
            T22 = fd.ops.broadcast_in_dim(
                T21, output_shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
            )
            T23 = fd.ops.broadcast_in_dim(
                T22, output_shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T24 = fd.ops.div(T17, T23)
            S25 = fd.define_constant(16, dtype=DataType.Int)
            S26 = fd.define_constant(16, dtype=DataType.Int)
            S27 = fd.define_constant(128, dtype=DataType.Int)
            S28 = fd.define_constant(128, dtype=DataType.Int)
            S29 = fd.define_constant(0.00000, dtype=DataType.Double)
            S30 = fd.define_constant(1.00000, dtype=DataType.Double)
            T31 = fd.ops.uniform(
                S29, S30, shape=[S25, S26, S27, S28], dtype=DataType.Float
            )
            S32 = fd.define_constant(1.0 - prob, dtype=DataType.Double)
            T33 = fd.ops.lt(T31, S32)
            T34 = fd.ops.cast(T33, dtype=DataType.Float)
            T35 = fd.ops.mul(T24, T34)
            S36 = fd.define_constant(1.0 / (1.0 - prob), dtype=DataType.Double)
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
            T1_slice1 = fd.ops.reshape(T0_slice1, [16, 128, 1024], [16, 128, 16, 64])
            T1_slice2 = fd.ops.reshape(T0_slice2, [16, 128, 1024], [16, 128, 16, 64])
            T1_slice3 = fd.ops.reshape(T0_slice3, [16, 128, 1024], [16, 128, 16, 64])
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
                symbolic_sizes=[-1, -1, -1],
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
                "Slice operation start_indices must be greater-than-or-equal-to 0. .*",
            ),
            (
                check_end_indices,
                "Slice operation end_indices must be greater-than-or-equal-to start_indices. .*",
            ),
            (
                check_strides,
                "nvFuser Limitation: All slice operation strides must be of size 1. .*",
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
                    # First check is here on legel fusions since the second time
                    # through they should already be cached
                    out = self.exec_nvfuser(
                        partial(check, acts=inp), inp, new_fusion_expected=first_check
                    )
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        error,
                        self.exec_nvfuser,
                        partial(check, acts=inp),
                        inp,
                    )
            first_check = False

    def test_constant_nans(self):
        inputs = [
            torch.randn(4, 4, device="cuda"),
        ]

        def fusion_func(fd: FusionDefinition) -> None:
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_constant(float("nan"))
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

    def test_matmuls(self):
        # Matmul Constraints:
        # 1. Inputs shapes need to be a multiple of 8
        # 2. Inputs need to be contiguous as the nvFuser matmul does
        #    not see non-contiguous inputs.
        nvf_inputs_nn = [
            torch.randn(8, 24, device="cuda", dtype=torch.float16),
            torch.randn(16, 8, device="cuda", dtype=torch.float16),
        ]
        eager_inputs_nn = [
            nvf_inputs_nn[0].clone().transpose(0, 1),
            nvf_inputs_nn[1].clone().transpose(0, 1),
        ]
        nvf_inputs_nt = [
            torch.randn(8, 24, device="cuda", dtype=torch.float16),
            torch.randn(8, 16, device="cuda", dtype=torch.float16),
        ]
        eager_inputs_nt = [
            nvf_inputs_nt[0].clone().transpose(0, 1),
            nvf_inputs_nt[1].clone(),
        ]
        nvf_inputs_tn = [
            torch.randn(24, 8, device="cuda", dtype=torch.float16),
            torch.randn(16, 8, device="cuda", dtype=torch.float16),
        ]
        eager_inputs_tn = [
            nvf_inputs_tn[0].clone(),
            nvf_inputs_tn[1].clone().transpose(0, 1),
        ]
        nvf_inputs_tt = [
            torch.randn(24, 8, device="cuda", dtype=torch.float16),
            torch.randn(8, 16, device="cuda", dtype=torch.float16),
        ]

        def fusion_func(fd: FusionDefinition, inps, matmul_fn) -> None:
            t0 = fd.from_pytorch(inps[0])
            t1 = fd.from_pytorch(inps[1])
            t2 = eval(matmul_fn)(t0, t1)
            fd.add_output(t2)

        tests = [
            ("fd.ops._matmul_nn", nvf_inputs_nn, eager_inputs_nn),
            ("fd.ops._matmul_nt", nvf_inputs_nt, eager_inputs_nt),
            ("fd.ops._matmul_tn", nvf_inputs_tn, eager_inputs_tn),
            ("fd.ops._matmul_tt", nvf_inputs_tt, nvf_inputs_tt),
        ]

        prop = torch.cuda.get_device_properties(torch.cuda.current_device())

        for mm_str, nvf_test_inputs, eager_test_inputs in tests:
            if prop.major == 8:
                nvf_out, _ = self.exec_nvfuser(
                    partial(fusion_func, inps=nvf_test_inputs, matmul_fn=mm_str),
                    nvf_test_inputs,
                )
                eager_out = torch.matmul(eager_test_inputs[0], eager_test_inputs[1])

                fp16_nvf_out = nvf_out[0].to(dtype=torch.float16)
                self.assertEqual(eager_out, fp16_nvf_out)
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Only the Ampere MMA Op is currently supported!"
                ):
                    with FusionDefinition() as fd:
                        partial(fusion_func, inps=nvf_test_inputs, matmul_fn=mm_str)(fd)
                    nvf_out = fd.execute(nvf_test_inputs)
                # It is necessary to reset the Fusion Cache so
                # serialization/deserialization does not exhibit the same error
                # across tests
                fc = FusionCache.get()
                fc.reset()


if __name__ == "__main__":
    run_tests()
