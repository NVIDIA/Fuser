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

from nvfuser_direct.testing.utils import is_pre_volta
from utils import NVFuserTest


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

        # t0 and t1 are ones(2, 4, 8) tensors.
        # t2 = t0 + t1 = twos(2, 4, 8)
        # t3 = t2 * 3.0 = sixes(2,4,8)
        # t4 = sum(t3, dim=-1) = forty-eights(2, 4)
        # The expected output is a tensor of 48's.
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])
