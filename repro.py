import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, None], dtype=DataType.BFloat16, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[None, True], dtype=DataType.BFloat16, is_cpu=False)
    T2 = fd.ops.cast(T1, dtype=DataType.Float)
    S3 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T4 = fd.ops.full(fill_value=S3, shape=[5, 0], dtype=DataType.BFloat16)
    T5 = fd.ops.cast(T4, dtype=DataType.Float)
    T6 = fd.ops.mul(T2, T5)
    T7 = fd.ops.cast(T0, dtype=DataType.Float)
    T8 = fd.ops.mul(T7, T5)
    T24 = fd.ops.sum(T6, axes=[1], keepdim=False, dtype=DataType.Null)
    T11 = fd.ops.sum(T8, axes=[0], keepdim=False, dtype=DataType.Null)
    fd.add_output(T24)
    fd.add_output(T11)

inputs = [
         torch.randn(5, 0, device='cuda', dtype=torch.bfloat16),
         torch.randn(5, 0, device='cuda', dtype=torch.bfloat16),
        ]

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

out = fd.execute(inputs)
