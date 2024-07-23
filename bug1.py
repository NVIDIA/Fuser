import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    S2 = fd.define_scalar(32, dtype=DataType.Int)
    S3 = fd.ops.size(T0, dim=0)
    S4 = fd.ops.div(S3, S2)
    V5 = fd.define_vector([S2, S4], dtype=DataType.Int)
    T6 = fd.ops.reshape(T0, new_shape=V5)
    T7 = fd.ops.cast(T6, dtype=DataType.Float)
    T7 = fd.ops.segment_set(T7)
    T9 = fd.ops.reshape(T1, new_shape=V5)
    T10 = fd.ops.cast(T9, dtype=DataType.Float)
    T11 = fd.ops.mul(T7, T10)
    fd.add_output(T11)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
    torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
]
fd.execute(inputs)