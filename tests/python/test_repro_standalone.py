import sys
import os

# Add the project root to the path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from nvfuser import FusionDefinition, enable_automatic_serialization

# Import the test utilities
sys.path.insert(0, os.path.join(project_root, 'tests', 'python'))
    
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
            fusion_func(fd)

    fd = MyFusion()
    fd.execute(inputs, profile=True)

enable_automatic_serialization()
test_repro()
