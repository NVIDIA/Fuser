import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[None, None, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T3 = fd.define_tensor(
        shape=[-1, -1, 1],
        contiguity=[True, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T5 = fd.ops.cast(T4, dtype=DataType.Float)
    S6 = fd.define_scalar(56, dtype=DataType.Int)
    S7 = fd.define_scalar(1024, dtype=DataType.Int)
    S8 = fd.define_scalar(1, dtype=DataType.Int)
    V9 = fd.define_vector([S6, S7, S8], dtype=DataType.Int)
    T10 = fd.ops.broadcast_in_dim(T2, shape=V9, broadcast_dims=[0, 1])
    S11 = fd.define_scalar(56, dtype=DataType.Int)
    S12 = fd.define_scalar(1024, dtype=DataType.Int)
    S13 = fd.define_scalar(1024, dtype=DataType.Int)
    V14 = fd.define_vector([S11, S12, S13], dtype=DataType.Int)
    T15 = fd.ops.broadcast_in_dim(T10, shape=V14, broadcast_dims=[0, 1, 2])
    T16 = fd.ops.sub(T5, T15)
    S17 = fd.define_scalar(56, dtype=DataType.Int)
    S18 = fd.define_scalar(1024, dtype=DataType.Int)
    S19 = fd.define_scalar(1024, dtype=DataType.Int)
    V20 = fd.define_vector([S17, S18, S19], dtype=DataType.Int)
    T21 = fd.ops.broadcast_in_dim(T3, shape=V20, broadcast_dims=[0, 1, 2])
    T22 = fd.ops.mul(T16, T21)
    T23 = fd.ops.cast(T0, dtype=DataType.Float)
    T24 = fd.ops.cast(T1, dtype=DataType.Float)
    T25 = fd.ops.sum(T24, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T26 = fd.ops.cast(T25, dtype=DataType.BFloat16)
    T27 = fd.ops.mul(T23, T24)
    T28 = fd.ops.mul(T22, T24)
    T29 = fd.ops.sum(T28, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T30 = fd.ops.cast(T29, dtype=DataType.BFloat16)
    T31 = fd.ops.mul(T21, T27)
    T32 = fd.ops.mul(T16, T27)
    T33 = fd.ops.sum(T32, dims=[2], keepdim=False, dtype=DataType.Null)
    S34 = fd.define_scalar(56, dtype=DataType.Int)
    S35 = fd.define_scalar(1024, dtype=DataType.Int)
    S36 = fd.define_scalar(1, dtype=DataType.Int)
    V37 = fd.define_vector([S34, S35, S36], dtype=DataType.Int)
    T38 = fd.ops.broadcast_in_dim(T33, shape=V37, broadcast_dims=[0, 1])
    T39 = fd.ops.neg(T31)
    T40 = fd.ops.sum(T39, dims=[2], keepdim=False, dtype=DataType.Null)
    S41 = fd.define_scalar(56, dtype=DataType.Int)
    S42 = fd.define_scalar(1024, dtype=DataType.Int)
    S43 = fd.define_scalar(1, dtype=DataType.Int)
    V44 = fd.define_vector([S41, S42, S43], dtype=DataType.Int)
    T45 = fd.ops.broadcast_in_dim(T40, shape=V44, broadcast_dims=[0, 1])
    S46 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T47 = fd.ops.mul(S46, T38)
    S48 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T49 = fd.ops.pow(T3, S48)
    T50 = fd.ops.mul(T47, T49)
    T51 = fd.ops.sum(T45, dims=[2], keepdim=False, dtype=DataType.Null)
    T52 = fd.ops.sum(T50, dims=[2], keepdim=False, dtype=DataType.Null)
    S53 = fd.define_scalar(56, dtype=DataType.Int)
    S54 = fd.define_scalar(1024, dtype=DataType.Int)
    S55 = fd.define_scalar(1, dtype=DataType.Int)
    V56 = fd.define_vector([S53, S54, S55], dtype=DataType.Int)
    T57 = fd.ops.broadcast_in_dim(T51, shape=V56, broadcast_dims=[0, 1])
    S58 = fd.define_scalar(56, dtype=DataType.Int)
    S59 = fd.define_scalar(1024, dtype=DataType.Int)
    S60 = fd.define_scalar(1024, dtype=DataType.Int)
    V61 = fd.define_vector([S58, S59, S60], dtype=DataType.Int)
    T62 = fd.ops.broadcast_in_dim(T57, shape=V61, broadcast_dims=[0, 1, 2])
    S63 = fd.define_scalar(0.000976562, dtype=DataType.Double)
    T64 = fd.ops.mul(S63, T62)
    S65 = fd.define_scalar(56, dtype=DataType.Int)
    S66 = fd.define_scalar(1024, dtype=DataType.Int)
    S67 = fd.define_scalar(1, dtype=DataType.Int)
    V68 = fd.define_vector([S65, S66, S67], dtype=DataType.Int)
    T69 = fd.ops.broadcast_in_dim(T52, shape=V68, broadcast_dims=[0, 1])
    S70 = fd.define_scalar(56, dtype=DataType.Int)
    S71 = fd.define_scalar(1024, dtype=DataType.Int)
    S72 = fd.define_scalar(1024, dtype=DataType.Int)
    V73 = fd.define_vector([S70, S71, S72], dtype=DataType.Int)
    T74 = fd.ops.broadcast_in_dim(T69, shape=V73, broadcast_dims=[0, 1, 2])
    S75 = fd.define_scalar(56, dtype=DataType.Int)
    S76 = fd.define_scalar(1024, dtype=DataType.Int)
    S77 = fd.define_scalar(1, dtype=DataType.Int)
    V78 = fd.define_vector([S75, S76, S77], dtype=DataType.Int)
    T79 = fd.ops.broadcast_in_dim(T2, shape=V78, broadcast_dims=[0, 1])
    S80 = fd.define_scalar(56, dtype=DataType.Int)
    S81 = fd.define_scalar(1024, dtype=DataType.Int)
    S82 = fd.define_scalar(1024, dtype=DataType.Int)
    V83 = fd.define_vector([S80, S81, S82], dtype=DataType.Int)
    T84 = fd.ops.broadcast_in_dim(T79, shape=V83, broadcast_dims=[0, 1, 2])
    S85 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T86 = fd.ops.mul(S85, T74)
    T87 = fd.ops.sub(T5, T84)
    T88 = fd.ops.mul(T86, T87)
    S89 = fd.define_scalar(1024.00, dtype=DataType.Double)
    S90 = fd.ops.reciprocal(S89)
    T91 = fd.ops.mul(T88, S90)
    T92 = fd.ops.add(T64, T91)
    T93 = fd.ops.add(T31, T92)
    T94 = fd.ops.cast(T93, dtype=DataType.BFloat16)
    fd.add_output(T94)
    fd.add_output(T26)
    fd.add_output(T30)


with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

inputs = [
    torch.randn((1024,), dtype=torch.bfloat16, device="cuda:0").as_strided(
        (56, 1024, 1024), (0, 0, 1)
    ),
    torch.randn((58720256,), dtype=torch.bfloat16, device="cuda:0").as_strided(
        (56, 1024, 1024), (1048576, 1024, 1)
    ),
    torch.randn((57344,), dtype=torch.float32, device="cuda:0").as_strided(
        (56, 1024), (1024, 1)
    ),
    torch.randn((57344,), dtype=torch.float32, device="cuda:0").as_strided(
        (56, 1024, 1), (1024, 1, 1)
    ),
    torch.randn((58720256,), dtype=torch.bfloat16, device="cuda:0").as_strided(
        (56, 1024, 1024), (1048576, 1024, 1)
    ),
]
fd.execute(inputs)
