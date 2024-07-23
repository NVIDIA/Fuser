import torch
from nvfuser import FusionDefinition, DataType
# def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
#     T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
#     T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
#     S2 = fd.define_scalar(32, dtype=DataType.Int)
#     S5 = fd.ops.size(T0, dim=0)
#     S4 = fd.ops.div(S5, S2)
#     V9 = fd.define_vector([S2, S4], dtype=DataType.Int)
#     T10 = fd.ops.reshape(T0, new_shape=V9)
#     T11 = fd.ops.cast(T10, dtype=DataType.Float)
#     # no err if remove this segment_set
#     T11 = fd.ops.segment_set(T11)
#     T13 = fd.ops.reshape(T1, new_shape=V9)
#     T14 = fd.ops.cast(T13, dtype=DataType.Float)
#     T15 = fd.ops.mul(T11, T14)
#     fd.add_output(T15)

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    S2 = fd.define_scalar(32, dtype=DataType.Int)
    S3 = fd.ops.size(T0, dim=0)
    S4 = fd.ops.div(S3, S2)
    V5 = fd.define_vector([S2, S4], dtype=DataType.Int)
    T6 = fd.ops.reshape(T0, new_shape=V5)
    T7 = fd.ops.cast(T6, dtype=DataType.Float)
    T8 = fd.ops.segment_set(T7)
    T9 = fd.ops.reshape(T1, new_shape=V5)
    T10 = fd.ops.cast(T9, dtype=DataType.Float)
    T11 = fd.ops.mul(T8, T10)
    fd.add_output(T11)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

# inputs = [
#     torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
#     torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
# ]
inputs = [
    torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
    torch.randn((128,), dtype=torch.bfloat16, device='cuda:0').as_strided((128,), (1,)),
]
fd.execute(inputs)

# def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
#     T0 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False)
#     T1 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False)
#     S2 = fd.define_scalar(32, dtype=DataType.Int)
#     S3 = fd.ops.size(T0, dim=1)
#     S4 = fd.ops.div(S3, S2)
#     S5 = fd.ops.size(T0, dim=0)
#     S6 = fd.ops.size(T0, dim=2)
#     S7 = fd.ops.size(T0, dim=3)
#     S8 = fd.define_scalar(32, dtype=DataType.Int)
#     V9 = fd.define_vector([S5, S8, S4, S6, S7], dtype=DataType.Int)
#     T10 = fd.ops.reshape(T0, new_shape=V9)
#     T11 = fd.ops.cast(T10, dtype=DataType.Float)
#     # no err if remove this segment_set
#     T11 = fd.ops.segment_set(T11)
#     T13 = fd.ops.reshape(T1, new_shape=V9)
#     T14 = fd.ops.cast(T13, dtype=DataType.Float)
#     T15 = fd.ops.mul(T11, T14)
#     fd.add_output(T15)

# with FusionDefinition() as fd:
#     nvfuser_fusion_id0(fd)

# inputs = [
#     torch.randn((102760448,), dtype=torch.bfloat16, device='cuda:0').as_strided((32, 256, 112, 112), (3211264, 12544, 112, 1)),
#     torch.randn((102760448,), dtype=torch.bfloat16, device='cuda:0').as_strided((32, 256, 112, 112), (3211264, 12544, 112, 1)),
# ]
# fd.execute(inputs)

# replaceSymbolicSizes after extent_simplification_map: 
# replaceSymbolicSizes: i6 -> ( (( (( getMetaData(T1) )).logical_size ))[1] )
# replaceSymbolicSizes: i3 -> ( (( (( getMetaData(T0) )).logical_size ))[3] )
# replaceSymbolicSizes: i2 -> ( (( (( getMetaData(T0) )).logical_size ))[2] )
# replaceSymbolicSizes: i1 -> ( (( (( getMetaData(T0) )).logical_size ))[1] )
# replaceSymbolicSizes: i0 -> ( (( (( getMetaData(T0) )).logical_size ))[0] )


# replaceSymbolicSizes: id: iS84{i0} -> ( (( (( getMetaData(T1) )).logical_size ))[0] )
# replaceSymbolicSizes: id: iS5{i6} -> ( (( (( getMetaData(T1) )).logical_size ))[1] )
# replaceSymbolicSizes: id: iS85{i2} -> ( (( (( getMetaData(T1) )).logical_size ))[2] )
# replaceSymbolicSizes: id: iS86{i3} -> ( (( (( getMetaData(T1) )).logical_size ))[3] )
# replaceSymbolicSizes: id: iS69{( ceilDiv(i6, 32) )} -> ( (( (( getMetaData(T4) )).logical_size ))[2] )

# replaceSymbolicSizes after extent_simplification_map: 
# replaceSymbolicSizes: i82 -> ( (( (( getMetaData(T4) )).logical_size ))[2] )
# replaceSymbolicSizes: i3 -> ( (( (( getMetaData(T1) )).logical_size ))[3] )
# replaceSymbolicSizes: i2 -> ( (( (( getMetaData(T1) )).logical_size ))[2] )
# replaceSymbolicSizes: i6 -> ( (( (( getMetaData(T1) )).logical_size ))[1] )
# replaceSymbolicSizes: i0 -> ( (( (( getMetaData(T1) )).logical_size ))[0] )

