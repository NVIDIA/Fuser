import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Half, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False)
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False)
    T3 = fd.ops.cast(T0, dtype=DataType.Float)
    S4 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S5 = fd.define_scalar(1.00000, dtype=DataType.Double)
    V6 = fd.ops.shape(T3)
    T7 = fd.ops.uniform(S4, S5, shape=V6, dtype=DataType.Float)
    S8 = fd.define_scalar(0.800000, dtype=DataType.Double)
    T9 = fd.ops.lt(T7, S8)
    T10 = fd.ops.cast(T9, dtype=DataType.Float)
    T11 = fd.ops.mul(T3, T10)
    S12 = fd.define_scalar(1.25000, dtype=DataType.Double)
    T13 = fd.ops.mul(T11, S12)
    T14, T15 = fd.ops.var_mean(T13, dims=[1], correction=0, keepdim=False)
    S16 = fd.ops.size(T3, dim=0)
    S17 = fd.define_scalar(1, dtype=DataType.Int)
    V18 = fd.define_vector([S16, S17], dtype=DataType.Int)
    T19 = fd.ops.broadcast_in_dim(T14, shape=V18, broadcast_dims=[0])
    T20 = fd.ops.broadcast_in_dim(T15, shape=V18, broadcast_dims=[0])
    S21 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T22 = fd.ops.add(T19, S21)
    T23 = fd.ops.rsqrt(T22)
    V24 = fd.ops.shape(T3)
    T25 = fd.ops.broadcast_in_dim(T20, shape=V24, broadcast_dims=[0, 1])
    T26 = fd.ops.sub(T13, T25)
    T27 = fd.ops.broadcast_in_dim(T23, shape=V24, broadcast_dims=[0, 1])
    T28 = fd.ops.mul(T26, T27)
    T29 = fd.ops.broadcast_in_dim(T1, shape=V24, broadcast_dims=[1])
    T30 = fd.ops.cast(T29, dtype=DataType.Float)
    T31 = fd.ops.mul(T28, T30)
    T32 = fd.ops.broadcast_in_dim(T2, shape=V24, broadcast_dims=[1])
    T33 = fd.ops.cast(T32, dtype=DataType.Float)
    T34 = fd.ops.add(T31, T33)
    T35 = fd.ops.cast(T34, dtype=DataType.Half)
    fd.add_output(T15)
    fd.add_output(T23)
    fd.add_output(T35)
    fd.add_output(T9)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((469762048,), dtype=torch.float16, device='cuda:0').as_strided((8192, 57344), (57344, 1)),
    torch.randn((57344,), dtype=torch.float16, device='cuda:0').as_strided((57344,), (1,)),
    torch.randn((57344,), dtype=torch.float16, device='cuda:0').as_strided((57344,), (1,)),
]
fd.execute(inputs)