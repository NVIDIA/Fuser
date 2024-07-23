import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    S4 = fd.define_scalar(32, dtype=DataType.Int)
    S5 = fd.ops.size(T0, dim=1)
    S6 = fd.ops.div(S5, S4)
    S7 = fd.ops.size(T0, dim=0)
    S8 = fd.ops.size(T0, dim=2)
    S9 = fd.ops.size(T0, dim=3)
    S10 = fd.define_scalar(32, dtype=DataType.Int)
    V11 = fd.define_vector([S7, S10, S6, S8, S9], dtype=DataType.Int)
    T12 = fd.ops.reshape(T0, new_shape=V11)
    T13 = fd.ops.cast(T12, dtype=DataType.Float)
    T14 = fd.ops.cast(T1, dtype=DataType.Float)

    V29 = fd.ops.shape(T13)
    T33 = fd.ops.segment_set(T13)
    S34 = fd.define_scalar(1, dtype=DataType.Int)
    S35 = fd.define_scalar(32, dtype=DataType.Int)
    S36 = fd.define_scalar(1, dtype=DataType.Int)
    S37 = fd.define_scalar(1, dtype=DataType.Int)
    V38 = fd.define_vector([S34, S35, S6, S36, S37], dtype=DataType.Int)
    T39 = fd.ops.reshape(T14, new_shape=V38)
    T41 = fd.ops.broadcast_in_dim(T39, shape=V29, broadcast_dims=[0, 1, 2, 3, 4])
    T42 = fd.ops.mul(T33, T41)
    fd.add_output(T42)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((102760448,), dtype=torch.bfloat16, device='cuda:0').as_strided((32, 256, 112, 112), (3211264, 12544, 112, 1)),
    torch.randn((256,), dtype=torch.bfloat16, device='cuda:0').as_strided((256,), (1,)),
]
fd.execute(inputs)