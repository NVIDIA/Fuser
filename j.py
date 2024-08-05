# CUDA devices:
#  0: NVIDIA H100 80GB HBM3
# torch version: 2.5.0a0+git8927fc2
# nvfuser version: 0.2.8+gitdd6886f
import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    S0 = fd.define_scalar(None, dtype=DataType.Int)
    T1 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False, stride_order=[0])
    T3 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False, stride_order=[0])
    T4 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T5 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, True, None, None], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T6 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, True, None, None], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T7 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[3, 2, 1, 0])
    T8 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[3, 2, 1, 0])
    T9 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[3, 2, 1, 0])
    T10 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Bool, is_cpu=False, stride_order=[3, 2, 1, 0])
    T11 = fd.define_tensor(shape=[-1, -1, -1, -1, 1], contiguity=[True, True, True, True, None], dtype=DataType.Float, is_cpu=False, stride_order=[4, 3, 2, 1, 0])
    T12 = fd.ops.sum(T11, dims=[4], keepdim=False, dtype=DataType.Null)
    T13 = fd.ops.set(T12)
    T14 = fd.ops.set(T12)
    T15 = fd.ops.sum(T14, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
    S16 = fd.define_scalar(1, dtype=DataType.Int)
    S17 = fd.define_scalar(288, dtype=DataType.Int)
    S18 = fd.define_scalar(1, dtype=DataType.Int)
    S19 = fd.define_scalar(1, dtype=DataType.Int)
    V20 = fd.define_vector([S16, S17, S18, S19], dtype=DataType.Int)
    T21 = fd.ops.broadcast_in_dim(T15, shape=V20, broadcast_dims=[1])
    T22 = fd.ops.set(T12)
    T23 = fd.ops.sum(T22, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
    T24 = fd.ops.broadcast_in_dim(T23, shape=V20, broadcast_dims=[1])
    S25 = fd.define_scalar(288, dtype=DataType.Int)
    V26 = fd.define_vector([S25], dtype=DataType.Int)
    T27 = fd.ops.reshape(T24, new_shape=V26)
    S28 = fd.define_scalar(288, dtype=DataType.Int)
    V29 = fd.define_vector([S28], dtype=DataType.Int)
    T30 = fd.ops.reshape(T21, new_shape=V29)
    S31 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T32 = fd.ops.mul(S31, T30)
    S33 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T34 = fd.ops.pow(T3, S33)
    T35 = fd.ops.mul(T32, T34)
    T36 = fd.ops.broadcast_in_dim(T27, shape=V20, broadcast_dims=[1])
    S37 = fd.define_scalar(2, dtype=DataType.Int)
    S38 = fd.define_scalar(288, dtype=DataType.Int)
    S39 = fd.define_scalar(120, dtype=DataType.Int)
    S40 = fd.define_scalar(160, dtype=DataType.Int)
    V41 = fd.define_vector([S37, S38, S39, S40], dtype=DataType.Int)
    T42 = fd.ops.broadcast_in_dim(T36, shape=V41, broadcast_dims=[0, 1, 2, 3])
    S43 = fd.define_scalar(2.60417e-05, dtype=DataType.Double)
    T44 = fd.ops.mul(S43, T42)
    T45 = fd.ops.broadcast_in_dim(T35, shape=V20, broadcast_dims=[1])
    T46 = fd.ops.broadcast_in_dim(T45, shape=V41, broadcast_dims=[0, 1, 2, 3])
    T47 = fd.ops.broadcast_in_dim(T2, shape=V20, broadcast_dims=[1])
    S48 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T49 = fd.ops.mul(S48, T46)
    T50 = fd.ops.sub(T1, T47)
    T51 = fd.ops.mul(T49, T50)
    S52 = fd.ops.cast(S0, dtype=DataType.Double)
    S53 = fd.ops.reciprocal(S52)
    T54 = fd.ops.mul(T51, S53)
    T55 = fd.ops.add(T44, T54)
    T56 = fd.ops.add(T13, T55)
    T57 = fd.ops.cast(T56, dtype=DataType.Half)
    fd.add_output(T57)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    38400,
    torch.randn((11059200,), dtype=torch.float32, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randn((288,), dtype=torch.float32, device='cuda:0').as_strided((288,), (1,)),
    torch.randn((288,), dtype=torch.float32, device='cuda:0').as_strided((288,), (1,)),
    torch.randn((11059200,), dtype=torch.float32, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randn((288,), dtype=torch.float32, device='cuda:0').as_strided((2, 288, 120, 160), (0, 1, 0, 0)),
    torch.randn((288,), dtype=torch.float32, device='cuda:0').as_strided((2, 288, 120, 160), (0, 1, 0, 0)),
    torch.randint(0, 2, (11059200,), dtype=torch.bool, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randint(0, 2, (11059200,), dtype=torch.bool, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randint(0, 2, (11059200,), dtype=torch.bool, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randint(0, 2, (11059200,), dtype=torch.bool, device='cuda:0').as_strided((2, 288, 120, 160), (5529600, 19200, 160, 1)),
    torch.randn((11059200,), dtype=torch.float32, device='cuda:0').as_strided((2, 288, 120, 160, 1), (5529600, 19200, 160, 1, 1)),
]
fd.execute(inputs)
