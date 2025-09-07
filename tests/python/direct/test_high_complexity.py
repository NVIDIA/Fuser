# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import torch._refs as refs
import torch._prims as prims

from nvfuser_direct import FusionDefinition


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
