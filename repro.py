import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id9(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, None], dtype=DataType.Bool, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
    T2 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[1, 0])
    S3 = fd.define_scalar(16, dtype=DataType.Int)
    S4 = fd.define_scalar(16, dtype=DataType.Int)
    S5 = fd.define_scalar(1, dtype=DataType.Int)
    V6 = fd.define_vector([S3, S4, S5], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[0, 1])
    S8 = fd.define_scalar(1, dtype=DataType.Int)
    S9 = fd.define_scalar(16, dtype=DataType.Int)
    S10 = fd.define_scalar(1, dtype=DataType.Int)
    S11 = fd.define_scalar(16, dtype=DataType.Int)
    S12 = fd.define_scalar(32, dtype=DataType.Int)
    S13 = fd.define_scalar(1, dtype=DataType.Int)
    V14 = fd.define_vector([S8, S9, S10, S11, S12, S13], dtype=DataType.Int)
    T15 = fd.ops.broadcast_in_dim(T7, shape=V14, broadcast_dims=[1, 3, 5])
    S16 = fd.define_scalar(16, dtype=DataType.Int)
    S17 = fd.define_scalar(16, dtype=DataType.Int)
    S18 = fd.define_scalar(32, dtype=DataType.Int)
    V19 = fd.define_vector([S16, S17, S18], dtype=DataType.Int)
    T20 = fd.ops.reshape(T15, new_shape=V19)
    T21 = fd.ops.slice(T1, start_indices=[0, 0, 16], end_indices=[16, 16, 32], strides=[1, 1, 1])
    S22 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T23 = fd.ops.where(T20, S22, T1)
    S24 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T25 = fd.ops.where(T0, S24, T23)
    fd.add_output(T21)
    fd.add_output(T25)

with FusionDefinition() as fd:
    nvfuser_fusion_id9(fd)

inputs = [
    torch.randint(0, 2, (256,), dtype=torch.bool, device='cuda:1').as_strided((16, 16, 32), (16, 1, 0)),
    torch.randn((8192,), dtype=torch.float32, device='cuda:1').as_strided((16, 16, 32), (512, 32, 1)),
    torch.randint(0, 2, (256,), dtype=torch.bool, device='cuda:1').as_strided((16, 16), (16, 1)),
]
fd.execute(inputs)
