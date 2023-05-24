# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from typing import Callable
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA

# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser import (
        FusionDefinition,
    )
except ImportError:
    pass

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestScheduleOps(TestCase):
    def definition_op_in_schedule_error(self, def_op_fn: Callable):
        """
        Common function to test for an error when a schedule op is used in a definition
        """
        inputs = [
            torch.randn(8, 8, 8, device="cuda"),
        ]

        class SchedError(FusionDefinition):
            def definition(self):
                self.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
                self.t1 = fd.ops.tanh(fd.t0)
                self.add_output(fd.t1)

            def schedule(self):
                def_op_fn(self)

        with self.assertRaisesRegex(
            RuntimeError, "Attempting to add to a completed definition!"
        ):
            fd = SchedError()
            _ = fd.execute(inputs)




if __name__ == "__main__":
    run_tests()
