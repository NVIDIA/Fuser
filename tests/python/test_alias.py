# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition
from nvfuser.testing.utils import NVFuserTest


class TestAlias(NVFuserTest):
    def test_squeeze_issue_3192(self):
        def fusion_func(fd: FusionDefinition):
            inp = fd.define_tensor([1, 1, 2], contiguity=True)
            out = fd.ops.squeeze(inp, [0, 1])
            out = fd.ops.slice(out, [1], [2])
            fd.add_output(out)

        in_tensor = torch.randn(1, 1, 2, device="cuda")
        out_tensors, _ = self.exec_nvfuser(fusion_func, [in_tensor])
        torch.testing.assert_close(
            out_tensors[0], in_tensor.squeeze([0, 1])[1:2], rtol=0, atol=0
        )
