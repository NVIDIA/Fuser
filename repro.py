import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Half, is_cpu=False) # [216, 20480]
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Half, is_cpu=False) # [216, 20480]
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False) # [216]
    T3 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False) # [216]
    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False) # [20480]
    T5 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False) # [20480]
    T6 = fd.ops.cast(T0, dtype=DataType.Float) # [216, 20480]
    T7 = fd.ops.cast(T1, dtype=DataType.Float) # [216, 20480]
    T8 = fd.ops.cast(T4, dtype=DataType.Float) # [20480]
    T9 = fd.ops.cast(T5, dtype=DataType.Float) # [20480]
    S10 = fd.define_scalar(20480, dtype=DataType.Int) # []
    T11 = fd.ops.broadcast_in_dim(T2, shape=[216, 20480], broadcast_dims=[0]) # [216, 20480]
    T12 = fd.ops.sub(T7, T11) # [216, 20480]
    T13 = fd.ops.broadcast_in_dim(T3, shape=[216, 20480], broadcast_dims=[0]) # [216, 20480]
    T14 = fd.ops.mul(T12, T13) # [216, 20480]
    T15 = fd.ops.broadcast_in_dim(T8, shape=[216, 20480], broadcast_dims=[1]) # [216, 20480]
    T16 = fd.ops.mul(T6, T15) # [216, 20480]
    T17 = fd.ops.mul(S10, T16) # [216, 20480]
    T18 = fd.ops.sum(T16, axes=[1], keepdim=False, dtype=DataType.Null) #[216]
    T19 = fd.ops.broadcast_in_dim(T18, shape=[216, 20480], broadcast_dims=[0]) # [216, 20480]
    T20 = fd.ops.mul(T16, T14) # [216, 20480]
    T21 = fd.ops.sum(T20, axes=[1], keepdim=False, dtype=DataType.Null) # [216]
    T22 = fd.ops.broadcast_in_dim(T21, shape=[216, 20480], broadcast_dims=[0]) # [216, 20480]
    T23 = fd.ops.mul(T14, T22) # [216, 20480]
    T24 = fd.ops.sub(T17, T19) # [216, 20480]
    T25 = fd.ops.sub(T24, T23) # [216, 20480]
    S26 = fd.ops.reciprocal(S10) # []
    T27 = fd.ops.mul(S26, T13) # [216, 20480]
    T28 = fd.ops.mul(T27, T25) # [216, 20480]
    T29 = fd.ops.mul(T6, T14) # [216, 20480]
    T30 = fd.ops.sum(T29, axes=[0], keepdim=False, dtype=DataType.Null) # [20480]
    T31 = fd.ops.sum(T6, axes=[0], keepdim=False, dtype=DataType.Null) # [20480]
    T32 = fd.ops.cast(T28, dtype=DataType.Half) # [216, 20480]
    T33 = fd.ops.cast(T30, dtype=DataType.Half) # [20480]
    T34 = fd.ops.cast(T31, dtype=DataType.Half) # [20480]
    fd.add_output(T32)
    fd.add_output(T33)
    fd.add_output(T34)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((216, 20480), dtype=torch.float16, device='cuda:0').as_strided((216, 20480), (20480, 1)),
    torch.randn((216, 20480), dtype=torch.float16, device='cuda:0').as_strided((216, 20480), (20480, 1)),
    torch.randn((216,), dtype=torch.float32, device='cuda:0').as_strided((216,), (1,)),
    torch.randn((216,), dtype=torch.float32, device='cuda:0').as_strided((216,), (1,)),
    torch.randn((20480,), dtype=torch.float16, device='cuda:0').as_strided((20480,), (1,)),
    torch.randn((20480,), dtype=torch.float16, device='cuda:0').as_strided((20480,), (1,)),
]
outputs = fd.execute(inputs)
for output in outputs:
    print(output.shape)
