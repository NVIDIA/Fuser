import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    T2 = fd.ops.sum(T0, dims=[0], keepdim=False, dtype=DataType.Null)
    T3 = fd.ops.add(T2, T1)
    fd.add_output(T3)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((765952,), dtype=torch.float32, device='cuda:0').as_strided((16, 47872), (47872, 1)),
    torch.randn((47872,), dtype=torch.float32, device='cuda:0').as_strided((47872,), (1,)),
    0,
]
fd.execute(inputs)
