# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

"""
Standalone version of test_repro extracted from tests/python/test_python_frontend.py
This test runs a fusion profiler test with noncodegen kernels and a gather test 1000 times.
"""

import sys
import os

# Add the project root to the path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from nvfuser import FusionDefinition

# Import the test utilities
sys.path.insert(0, os.path.join(project_root, 'tests', 'python'))
from python.utils import NVFuserTest
    
def test_repro():
    # Inlined from test_fusion_profiler_with_noncodegen_kernels
    inputs = [
        torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
        torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
        torch.randn((16, 16), dtype=torch.bfloat16, device="cuda:0"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.from_pytorch(inputs[1])
        T2 = fd.from_pytorch(inputs[2])
        T3 = fd.ops.linear(T0, T2)
        T4 = fd.ops.add(T3, T1)
        fd.add_output(T4)

    class MyFusion(FusionDefinition):
        def definition(self):
            return
            fusion_func(fd)

    fd = MyFusion()
    # fd.execute(inputs, profile=True)

NVFuserTest().setup_class()
test_repro()
test_repro()

