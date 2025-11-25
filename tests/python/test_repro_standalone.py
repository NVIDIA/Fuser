import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from nvfuser import FusionDefinition

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

    with FusionDefinition() as fd:
        fusion_func(fd)
    nvf_out = fd.execute(
        inputs,
        # _enable_options=["id_model_extra_validation"],
    )

    eager_out = torch.gather(inputs[0] + inputs[1], dim, inputs[2])
    torch.equal(eager_out, nvf_out[0])

test_fn(0)
test_fn(1)