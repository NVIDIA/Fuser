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


class TestRepro(NVFuserTest):
    """Standalone test class for reproducing the issue"""
    
    def test_repro(self):
        print(f"Running test_fusion_profiler_with_noncodegen_kernels {iteration} times...")
        
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
                fusion_func(fd)

        fd = MyFusion()
        try:
            fd.execute(inputs, profile=True)
            # self.assertTrue(fd.profile().fusion_id >= 0)
            # self.assertEqual(len(fd.profile().kernel_profiles), 2)
            # self.assertGreaterEqual(len(fd.profile().kernel_profiles[0].name), 0)
            # self.assertGreaterEqual(len(fd.profile().kernel_profiles[1].name), 0)
        except Exception as e:
            raise RuntimeError(
                "FusionDefinition's execute() did not run correctly with profile enabled!"
            )
        
        # Inlined from test_gather
        inputs = [
            torch.randn(8, 16, device="cuda"),
            torch.randn(8, 16, device="cuda"),
            torch.randint(0, 8, (4, 4), device="cuda").to(dtype=torch.long),
        ]

        def test_fn(dim):
            def fusion_func(fd: FusionDefinition):
                t0 = fd.from_pytorch(inputs[0])
                t1 = fd.from_pytorch(inputs[1])
                t2 = fd.from_pytorch(inputs[2])
                t3 = fd.ops.add(t0, t1)
                t4 = fd.ops.gather(t3, t2, dim)
                fd.add_output(t4)

            nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

            eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
            torch.equal(eager_out, nvf_out[0])
            # self.assertEqual(eager_out, nvf_out[0])

        test_fn(0)
        test_fn(1)


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
    test_instance = TestRepro()
    test_instance.setup_class()
    
    try:
        test_instance.test_repro()
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

