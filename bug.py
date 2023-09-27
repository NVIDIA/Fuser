import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id5271(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False)
    S1 = fd.define_scalar(None, dtype=DataType.Float)
    T2 = fd.ops.pad(T0, [-3, 0, 0, 0], S1)
    fd.add_output(T2)

with FusionDefinition() as fd:
    nvfuser_fusion_id5271(fd)

inputs = [
    torch.randint(0, 10, (4,), dtype=torch.float32, device='cuda:0').as_strided((2, 2), (2, 1)),
    (-3.0035226810462152),
]
fd.execute(inputs)
