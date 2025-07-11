# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition
from python.utils import NVFuserTest


# I tried to merge the tests to opinfo and [failed](https://github.com/NVIDIA/Fuser/issues/3225).
class TestDefine(NVFuserTest):
    def test_contiguous(self):
        def fusion_func(fd: FusionDefinition):
            inp = fd.define_tensor([2, 3], contiguity=True)
            out = fd.ops.add(inp, inp)
            fd.add_output(out)

        in_tensor = torch.randn(2, 3, device="cuda")
        out_tensors, _ = self.exec_nvfuser(fusion_func, [in_tensor])
        torch.testing.assert_close(out_tensors[0], in_tensor * 2)

    def test_noncontiguous(self):
        def fusion_func(fd: FusionDefinition):
            inp = fd.define_tensor([2, 3])
            out = fd.ops.add(inp, inp)
            fd.add_output(out)

        in_tensor = torch.randn(8, device="cuda").as_strided([2, 3], [4, 1])
        out_tensors, _ = self.exec_nvfuser(fusion_func, [in_tensor])
        torch.testing.assert_close(out_tensors[0], in_tensor * 2)

    def test_broadcast(self):
        def fusion_func(fd: FusionDefinition):
            inp = fd.define_tensor([1, 2, 1], contiguity=True)
            out = fd.ops.add(inp, inp)
            fd.add_output(out)

        in_tensor = torch.randn(1, 2, 1, device="cuda").as_strided([1, 2, 1], [0, 1, 0])
        out_tensors, _ = self.exec_nvfuser(fusion_func, [in_tensor])
        torch.testing.assert_close(out_tensors[0], in_tensor * 2)

    def test_contiguity_with_stride_order(self):
        def fusion_func(fd: FusionDefinition):
            inp = fd.define_tensor([1, 2], contiguity=True, stride_order=[0, 1])
            out = fd.ops.add(inp, inp)
            fd.add_output(out)

        in_tensor = torch.randn(1, 2, device="cuda")
        out_tensors, _ = self.exec_nvfuser(fusion_func, [in_tensor])
        torch.testing.assert_close(out_tensors[0], in_tensor * 2)
