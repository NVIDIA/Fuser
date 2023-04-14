# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from copy import deepcopy
from functools import partial
import math
import re
from typing import List, Callable
import unittest
import itertools

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA

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

@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestUserSchedules(TestCase):
    def sched_op_in_definition_error(self, sched_op_fn: Callable):
        """
        Common function to test for an error when a schedule op is used in a definition
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
        ]
        def fusion_fn(fd: FusionDefinition):
            fd.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            fd.t1 = fd.ops.tanh(fd.t0)
            fd.add_output(fd.t1)
        
        class DefError(FusionDefinition):
            def definition(self):
                fusion_fn(self)
                sched_op_fn(self)
        
        with self.assertRaisesRegex(RuntimeError, "Attempting to use a SchedOperators Op prior to definition!"):
            fd = DefError()
            _ = fd.execute(inputs)
    
    def valid_use(self, sched_op_fn: Callable):
        """
        Common function to test op works in a common case
        """
        inputs = [
            torch.randn(4, 4, 4, device="cuda"),
        ]
        def fusion_fn(fd: FusionDefinition):
            fd.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            fd.t1 = fd.ops.sum(fd.t0, axis=-1)
            fd.add_output(fd.t1)
        
        class BasicValid(FusionDefinition):
            def definition(self):
                fusion_fn(self)

            def schedule(self):
                sched_op_fn(self)

        fd = BasicValid()
        # TODO: This can cause warnings from the FusionCache. It would be good
        # to capture them, instead.
        # Warning: You are overwriting the current user schedule for a definition!
        nvf_user_out = fd.execute(inputs)
        nvf_out = fd.execute(inputs, override_user_schedule=True)
        self.assertEqual(nvf_user_out, nvf_out)

    def test_merge_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.merge(fd.t1, 1))
        self.valid_use(lambda fd: fd.sched.merge(fd.t1, 0))
    
    def test_reduction_factor_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.reduction_factor(fd.t1, [-1]))

        def sched_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [2])
        self.valid_use(sched_fn)
    
    def test_reorder_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.reorder(fd.t1, {0: 1, 1: 0}))
        self.valid_use(lambda fd: fd.sched.reorder(fd.t1, {0: 1, 1: 0}))
    
    def test_split_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.split(fd.t1, 1, 2))
        self.valid_use(lambda fd: fd.sched.split(fd.t1, 1, 2))
       
if __name__ == "__main__":
    run_tests()
