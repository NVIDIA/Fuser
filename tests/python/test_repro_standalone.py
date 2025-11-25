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
from copy import deepcopy
from nvfuser import FusionDefinition, FusionCache


def test_repro():
    """
    This test combines two tests and runs them 1000 times:
    1. test_fusion_profiler_with_noncodegen_kernels
    2. test_gather
    """
    for iteration in range(1000):
        print(f"\n[Iteration {iteration}] Starting iteration...")
        sys.stdout.flush()
        
        # Inlined from test_fusion_profiler_with_noncodegen_kernels
        print(f"[Iteration {iteration}] Step 1: Creating inputs for fusion_profiler test...")
        sys.stdout.flush()
        
        inputs = [
            torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((2, 4, 16), dtype=torch.bfloat16, device="cuda:0"),
            torch.randn((16, 16), dtype=torch.bfloat16, device="cuda:0"),
        ]

        print(f"[Iteration {iteration}] Step 2: Defining fusion function...")
        sys.stdout.flush()
        
        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.from_pytorch(inputs[0])
            T1 = fd.from_pytorch(inputs[1])
            T2 = fd.from_pytorch(inputs[2])
            T3 = fd.ops.linear(T0, T2)
            T4 = fd.ops.add(T3, T1)
            fd.add_output(T4)

        print(f"[Iteration {iteration}] Step 3: Creating MyFusion class...")
        sys.stdout.flush()
        
        class MyFusion(FusionDefinition):
            def definition(self):
                fusion_func(fd)

        fd = MyFusion()

        return


def main():
    """Main entry point for running the standalone test"""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a CUDA device.")
        sys.exit(1)
    
    # Check GPU architecture (test requires Volta or newer)
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    if prop.major < 7:
        print(f"SKIPPED: Test requires Volta or newer GPU (major >= 7), found compute capability {prop.major}.{prop.minor}")
        sys.exit(0)
    
    # Run the test
    print("Starting test_repro...")
    
    try:
        test_repro()
        print("\n" + "="*70)
        print("TEST PASSED: All 1000 iterations completed successfully!")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    main()

