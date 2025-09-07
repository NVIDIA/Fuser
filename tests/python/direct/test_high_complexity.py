# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import torch._refs as refs
import torch._prims as prims

import io
from contextlib import redirect_stdout, redirect_stderr
import pytest

from nvfuser_direct import FusionDefinition, DataType
import itertools
from functools import partial


def test_broadcast_in_dim_with_dynamic_shapes(nvfuser_direct_test):
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
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_1, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Test 2
    inputs = inputs_2
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_1, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Func_2 and Func_3 are nearly identical except that have a different
    # concrete output shape for their broadcast_in_dim.  Therefore, test 4
    # should not be cached.
    # Note: It is assumed that definition will change with Tensor Size with
    # concrete shapes.

    # Test 3
    inputs = inputs_1
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_2, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Test 4
    inputs = inputs_2
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_3, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


# Test that symbolic IterDomains can be concatenated
# https://github.com/NVIDIA/Fuser/issues/1554
def test_cat_symbolic(nvfuser_direct_test):
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
        T20 = fd.ops.reshape(T15, new_shape=[S16, S17, S18])
        T21 = fd.ops.permute(T13, dims=[0, 2, 1, 3])
        S22 = fd.define_scalar(4, dtype=DataType.Int)
        S23 = fd.define_scalar(64, dtype=DataType.Int)
        S24 = fd.define_scalar(768, dtype=DataType.Int)
        T26 = fd.ops.reshape(T21, new_shape=[S22, S23, S24])
        T27 = fd.ops.cat([T20, T26, T11], dim=2)
        T28 = fd.ops.sum(T27, [0, 1], keepdim=False, dtype=DataType.Null)
        fd.add_output(T27)
        fd.add_output(T28)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    t12 = inputs[1] * inputs[-2]
    t13 = torch.permute(t12, [0, 1, 3, 2])
    t14 = inputs[0] * inputs[-3]
    t15 = torch.permute(t14, [0, 2, 1, 3])
    t20 = torch.reshape(t15, [4, 64, 768])
    t21 = torch.permute(t13, [0, 2, 1, 3])
    t26 = torch.reshape(t21, [4, 64, 768])
    t27 = torch.cat([t20, t26, inputs[-1]], dim=2)
    t28 = t27.sum([0, 1])

    nvfuser_direct_test.assertEqual(nvf_out[0], t27)
    nvfuser_direct_test.assertEqual(nvf_out[1], t28)


def test_slice_error_checks(nvfuser_direct_test):
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
        T1 = fd.ops.slice(T0, start_indices=[3, 4], end_indices=[1, 2], strides=[1, 1])
        fd.add_output(T1)

    def check_strides(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[0, 0], end_indices=[5, 5], strides=[5, 5])
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

    def legal(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[6, 6], end_indices=[8, 8], strides=[1, 1])
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
        (legal, None),
    ]

    # Redirect stdout and stderr messages to log to avoid false positives in
    # the CI.
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    for combination in itertools.product(inputs, checks):
        inputs, (check, error_msg) = combination
        if error_msg is None:
            out = nvfuser_direct_test.exec_nvfuser(
                partial(check, acts=inputs),
                inputs,
            )
        else:
            with pytest.raises(RuntimeError, match=error_msg), redirect_stdout(
                stdout_capture
            ), redirect_stderr(stderr_capture):
                nvfuser_direct_test.exec_nvfuser(
                    partial(check, acts=inputs),
                    inputs,
                )
