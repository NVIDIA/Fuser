import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id1064(fd : FusionDefinition) -> None :
    S0 = fd.define_scalar(None, dtype=DataType.Double)
    S1 = fd.define_scalar(None, dtype=DataType.Double)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    S4 = fd.define_scalar(None, dtype=DataType.Int)
    S5 = fd.define_scalar(None, dtype=DataType.Int)
    S6 = fd.define_scalar(None, dtype=DataType.Int)
    S7 = fd.define_scalar(None, dtype=DataType.Int)
    S8 = fd.define_scalar(None, dtype=DataType.Int)
    S9 = fd.define_scalar(None, dtype=DataType.Int)
    S10 = fd.define_scalar(None, dtype=DataType.Int)
    S11 = fd.define_scalar(None, dtype=DataType.Int)
    S12 = fd.define_scalar(None, dtype=DataType.Int)
    T13 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T14 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T15 = fd.define_tensor(shape=[-1, 1, -1, -1, -1], contiguity=[True, None, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[4, 3, 2, 1, 0])
    T16 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T17 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
    T18 = fd.ops.mul(T14, S1)
    T19 = fd.ops.permute(T18, dims=[0, 1, 3, 2])
    T20 = fd.ops.mul(T13, S0)
    T21 = fd.ops.slice(T19, start_indices=[0, 0, 0, 0], end_indices=[5, 7, 5, 64], strides=[1, 1, 1, 1])
    S22 = fd.define_scalar(0, dtype=DataType.Int)
    S23 = fd.define_scalar(5, dtype=DataType.Int)
    S24 = fd.define_scalar(7, dtype=DataType.Int)
    S25 = fd.define_scalar(5, dtype=DataType.Int)
    S26 = fd.define_scalar(0, dtype=DataType.Int)
    V27 = fd.define_vector([S23, S24, S25, S26], dtype=DataType.Int)
    T28 = fd.ops.full(shape=V27, fill_value=S22, dtype=DataType.Float)
    S29 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T30 = fd.ops.pad(T28, [0, 64, 0, 0, 0, 0, 0, 0], S29)
    T31 = fd.ops.slice(T20, start_indices=[0, 0, 0, 0], end_indices=[5, 7, 5, 64], strides=[1, 1, 1, 1])
    S32 = fd.define_scalar(0, dtype=DataType.Int)
    S33 = fd.define_scalar(5, dtype=DataType.Int)
    S34 = fd.define_scalar(7, dtype=DataType.Int)
    S35 = fd.define_scalar(5, dtype=DataType.Int)
    S36 = fd.define_scalar(0, dtype=DataType.Int)
    V37 = fd.define_vector([S33, S34, S35, S36], dtype=DataType.Int)
    T38 = fd.ops.full(shape=V37, fill_value=S32, dtype=DataType.Float)
    S39 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T40 = fd.ops.pad(T38, [0, 64, 0, 0, 0, 0, 0, 0], S39)
    T41 = fd.ops.mul(T21, T17)
    T42 = fd.ops.mul(T21, T16)
    T43 = fd.ops.slice(T41, start_indices=[0, 0, 0, 0], end_indices=[5, 7, 5, 32], strides=[1, 1, 1, 1])
    T44 = fd.ops.slice(T41, start_indices=[0, 0, 0, 32], end_indices=[5, 7, 5, 64], strides=[1, 1, 1, 1])
    T45 = fd.ops.neg(T43)
    S46 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T47 = fd.ops.pad(T45, [32, 0, 0, 0, 0, 0, 0, 0], S46)
    T48 = fd.ops.add(T42, T47)
    S49 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T50 = fd.ops.pad(T44, [0, 32, 0, 0, 0, 0, 0, 0], S49)
    T51 = fd.ops.add(T48, T50)
    S52 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T53 = fd.ops.pad(T51, [0, 0, 0, 0, 0, 0, 0, 0], S52)
    T54 = fd.ops.add(T30, T53)
    T55 = fd.ops.mul(T31, T17)
    T56 = fd.ops.mul(T31, T16)
    T57 = fd.ops.slice(T55, start_indices=[0, 0, 0, 0], end_indices=[5, 7, 5, 32], strides=[1, 1, 1, 1])
    T58 = fd.ops.slice(T55, start_indices=[0, 0, 0, 32], end_indices=[5, 7, 5, 64], strides=[1, 1, 1, 1])
    T59 = fd.ops.neg(T57)
    S60 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T61 = fd.ops.pad(T59, [32, 0, 0, 0, 0, 0, 0, 0], S60)
    T62 = fd.ops.add(T56, T61)
    S63 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T64 = fd.ops.pad(T58, [0, 32, 0, 0, 0, 0, 0, 0], S63)
    T65 = fd.ops.add(T62, T64)
    S66 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T67 = fd.ops.pad(T65, [0, 0, 0, 0, 0, 0, 0, 0], S66)
    T68 = fd.ops.add(T40, T67)
    S69 = fd.define_scalar(5, dtype=DataType.Int)
    S70 = fd.define_scalar(1, dtype=DataType.Int)
    S71 = fd.define_scalar(7, dtype=DataType.Int)
    S72 = fd.define_scalar(5, dtype=DataType.Int)
    S73 = fd.define_scalar(64, dtype=DataType.Int)
    V74 = fd.define_vector([S69, S70, S71, S72, S73], dtype=DataType.Int)
    T75 = fd.ops.reshape(T54, new_shape=V74)
    S76 = fd.define_scalar(5, dtype=DataType.Int)
    S77 = fd.define_scalar(1, dtype=DataType.Int)
    S78 = fd.define_scalar(7, dtype=DataType.Int)
    S79 = fd.define_scalar(5, dtype=DataType.Int)
    S80 = fd.define_scalar(64, dtype=DataType.Int)
    V81 = fd.define_vector([S76, S77, S78, S79, S80], dtype=DataType.Int)
    T82 = fd.ops.reshape(T68, new_shape=V81)
    T83 = fd.ops.sum(T15, dims=[1, 2], keepdim=False, dtype=DataType.Null)
    S84 = fd.define_scalar(5, dtype=DataType.Int)
    S85 = fd.define_scalar(1, dtype=DataType.Int)
    S86 = fd.define_scalar(1, dtype=DataType.Int)
    S87 = fd.define_scalar(5, dtype=DataType.Int)
    S88 = fd.define_scalar(64, dtype=DataType.Int)
    V89 = fd.define_vector([S84, S85, S86, S87, S88], dtype=DataType.Int)
    T90 = fd.ops.broadcast_in_dim(T83, shape=V89, broadcast_dims=[0, 3, 4])
    T91 = fd.ops.sum(T75, dims=[1, 2], keepdim=False, dtype=DataType.Null)
    S92 = fd.define_scalar(5, dtype=DataType.Int)
    S93 = fd.define_scalar(1, dtype=DataType.Int)
    S94 = fd.define_scalar(1, dtype=DataType.Int)
    S95 = fd.define_scalar(5, dtype=DataType.Int)
    S96 = fd.define_scalar(64, dtype=DataType.Int)
    V97 = fd.define_vector([S92, S93, S94, S95, S96], dtype=DataType.Int)
    T98 = fd.ops.broadcast_in_dim(T91, shape=V97, broadcast_dims=[0, 3, 4])
    T99 = fd.ops.cat([T82, T98, T90], dim=2)
    fd.add_output(T99)

with FusionDefinition() as fd:
    nvfuser_fusion_id1064(fd)

inputs = [
    0.3535533905932738,
    0.3535533905932738,
    2,
    5,
    1,
    7,
    5,
    64,
    5,
    1,
    7,
    5,
    64,
    torch.randn((11200,), dtype=torch.float32, device='cuda:0').as_strided((5, 7, 5, 64), (2240, 320, 64, 1)),
    torch.randn((11200,), dtype=torch.float32, device='cuda:0').as_strided((5, 7, 64, 5), (2240, 320, 5, 1)),
    torch.randn((11200,), dtype=torch.float32, device='cuda:0').as_strided((5, 1, 7, 5, 64), (2240, 2240, 320, 64, 1)),
    torch.randn((320,), dtype=torch.float32, device='cuda:0').as_strided((5, 7, 5, 64), (0, 0, 64, 1)),
    torch.randn((320,), dtype=torch.float32, device='cuda:0').as_strided((5, 7, 5, 64), (0, 0, 64, 1)),
]
fd.execute(inputs)
