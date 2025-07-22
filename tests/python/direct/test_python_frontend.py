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
)
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

import pytest
from python.direct_utils import is_pre_hopper, is_pre_blackwell


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

    # t0 and t1 are ones(2, 4, 8) tensors.
    # t2 = t0 + t1 = twos(2, 4, 8)
    # t3 = t2 * 3.0 = sixes(2,4,8)
    # t4 = sum(t3, dim=-1) = forty-eights(2, 4)
    # The expected output is a tensor of 48's.
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
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
