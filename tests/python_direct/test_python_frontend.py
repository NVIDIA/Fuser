# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)

from utils import (
    is_pre_volta,
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
