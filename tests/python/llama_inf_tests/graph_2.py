import torch
from nvfuser import FusionDefinition, DataType
import time

def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 32, 6, 64], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
    T1 = fd.define_tensor(shape=[2048, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T2 = fd.define_tensor(shape=[1, 6, 2048], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T3 = fd.define_tensor(shape=[2048], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T4 = fd.define_tensor(shape=[8192, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T5 = fd.define_tensor(shape=[8192, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T6 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T7 = fd.define_tensor(shape=[2048], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T8 = fd.define_tensor(shape=[128256, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T9 = fd.ops.permute(T0, dims=[0, 2, 1, 3])
    T10 = fd.ops.stride_order(T9, stride_order=[3, 2, 1, 0])
    T15 = fd.ops.reshape(T10, new_shape=[1, 6, 2048])
    T16 = fd.ops.stride_order(T15, stride_order=[2, 1, 0])
    T17 = fd.ops.linear(T16, T1)
    T18 = fd.ops.cast(T2, dtype=DataType.Float)
    T19 = fd.ops.cast(T17, dtype=DataType.Float)
    T20 = fd.ops.add(T18, T19)
    S21 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T22 = fd.ops.pow(T20, S21)
    T23 = fd.ops.sum(T22, dims=[2], keepdim=False, dtype=DataType.Null)
    T28 = fd.ops.broadcast_in_dim(T23, shape=[1, 6, 1], broadcast_dims=[0, 1])
    S29 = fd.define_scalar(2048.00, dtype=DataType.Double)
    S30 = fd.ops.reciprocal(S29)
    T31 = fd.ops.mul(T28, S30)
    S32 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T33 = fd.ops.add(T31, S32)
    T34 = fd.ops.rsqrt(T33)
    T39 = fd.ops.broadcast_in_dim(T34, shape=[1, 6, 2048], broadcast_dims=[0, 1, 2])
    T40 = fd.ops.mul(T20, T39)
    T45 = fd.ops.broadcast_in_dim(T3, shape=[1, 6, 2048], broadcast_dims=[2])
    T46 = fd.ops.cast(T45, dtype=DataType.Float)
    T47 = fd.ops.mul(T46, T40)
    T48 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    T49 = fd.ops.linear(T48, T4)
    T50 = fd.ops.cast(T49, dtype=DataType.Float)
    T51 = fd.ops.neg(T50)
    T52 = fd.ops.exp(T51)
    S53 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T54 = fd.ops.add(S53, T52)
    T55 = fd.ops.reciprocal(T54)
    T56 = fd.ops.mul(T50, T55)
    T57 = fd.ops.linear(T48, T5)
    T58 = fd.ops.cast(T57, dtype=DataType.Float)
    T59 = fd.ops.mul(T56, T58)
    T60 = fd.ops.cast(T59, dtype=DataType.BFloat16)
    T61 = fd.ops.linear(T60, T6)
    T62 = fd.ops.cast(T61, dtype=DataType.Float)
    T63 = fd.ops.add(T20, T62)
    S64 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T65 = fd.ops.pow(T63, S64)
    T66 = fd.ops.sum(T65, dims=[2], keepdim=False, dtype=DataType.Null)
    T71 = fd.ops.broadcast_in_dim(T66, shape=[1, 6, 1], broadcast_dims=[0, 1])
    S72 = fd.define_scalar(2048.00, dtype=DataType.Double)
    S73 = fd.ops.reciprocal(S72)
    T74 = fd.ops.mul(T71, S73)
    S75 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T76 = fd.ops.add(T74, S75)
    T77 = fd.ops.rsqrt(T76)
    T82 = fd.ops.broadcast_in_dim(T77, shape=[1, 6, 2048], broadcast_dims=[0, 1, 2])
    T83 = fd.ops.mul(T63, T82)
    T88 = fd.ops.broadcast_in_dim(T7, shape=[1, 6, 2048], broadcast_dims=[2])
    T89 = fd.ops.cast(T88, dtype=DataType.Float)
    T90 = fd.ops.mul(T89, T83)
    T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
    T92 = fd.ops.linear(T91, T8)
    fd.add_output(T92)

with FusionDefinition() as fd:
    nvfuser_fusion_id2(fd)

inputs = [
    torch.randn(12288, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 32, 6, 64), (12288, 64, 2048, 1)),
    torch.testing.make_tensor((2048, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((1, 6, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048, 8192), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((128256, 2048), dtype=torch.bfloat16, device='cuda:0'),
]

fd.execute(inputs)


# for _ in range(3):
#     fd.execute(inputs)

# torch.cuda.synchronize()
# start = time.time()
# # Mark the profiling region
# torch.cuda.cudart().cudaProfilerStart()

# for _ in range(100):
#     fd.execute(inputs)

# torch.cuda.cudart().cudaProfilerStop()
# torch.cuda.synchronize()
# end = time.time()
# print(end-start)