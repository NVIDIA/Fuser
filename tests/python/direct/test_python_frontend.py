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
