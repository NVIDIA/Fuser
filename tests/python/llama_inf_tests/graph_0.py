import torch
from nvfuser import FusionDefinition, DataType
import time

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 6], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[128256, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T2 = fd.define_tensor(shape=[1, 6], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    S3 = fd.define_scalar(2.00000, dtype=DataType.Double)
    S4 = fd.define_scalar(False, dtype=DataType.Bool)
    S5 = fd.define_scalar(False, dtype=DataType.Bool)
    T6 = fd.ops.embedding_fwd(T0, T1, None, None, S3, S4, S5)
    S7 = fd.define_scalar(6, dtype=DataType.Int)
    S8 = fd.define_scalar(0, dtype=DataType.Int)
    S9 = fd.define_scalar(1, dtype=DataType.Int)
    T10 = fd.ops.iota(S7, S8, S9, dtype=DataType.Int)
    T14 = fd.ops.broadcast_in_dim(T10, shape=[1, 6], broadcast_dims=[1])
    S15 = fd.define_scalar(-3.38953e+38, dtype=DataType.Double)
    T19 = fd.ops.full(shape=[6, 6], fill_value=S15, dtype=DataType.BFloat16)
    T23 = fd.ops.broadcast_in_dim(T10, shape=[6, 1], broadcast_dims=[0])
    T27 = fd.ops.broadcast_in_dim(T14, shape=[6, 6], broadcast_dims=[0, 1])
    T31 = fd.ops.broadcast_in_dim(T23, shape=[6, 6], broadcast_dims=[0, 1])
    T32 = fd.ops.sub(T27, T31)
    S33 = fd.define_scalar(1, dtype=DataType.Int)
    T34 = fd.ops.ge(T32, S33)
    S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T36 = fd.ops.where(T34, T19, S35)
    T40 = fd.ops.reshape(T10, new_shape=[6, 1])
    T44 = fd.ops.broadcast_in_dim(T10, shape=[6, 6], broadcast_dims=[1])
    T48 = fd.ops.broadcast_in_dim(T40, shape=[6, 6], broadcast_dims=[0, 1])
    T49 = fd.ops.gt(T44, T48)
    T50 = fd.ops.cast(T36, dtype=DataType.Float)
    T51 = fd.ops.cast(T49, dtype=DataType.Float)
    T52 = fd.ops.mul(T50, T51)
    T53 = fd.ops.cast(T52, dtype=DataType.BFloat16)
    T59 = fd.ops.broadcast_in_dim(T53, shape=[1, 1, 6, 6], broadcast_dims=[2, 3])
    T65 = fd.ops.broadcast_in_dim(T59, shape=[1, 1, 6, 6], broadcast_dims=[0, 1, 2, 3])
    T66 = fd.ops.set(T65)
    T72 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 1, 6], broadcast_dims=[0, 3])
    T78 = fd.ops.broadcast_in_dim(T72, shape=[1, 1, 6, 6], broadcast_dims=[0, 1, 2, 3])
    T79 = fd.ops.cast(T66, dtype=DataType.Float)
    T80 = fd.ops.cast(T78, dtype=DataType.Float)
    T81 = fd.ops.add(T79, T80)
    T82 = fd.ops.cast(T81, dtype=DataType.BFloat16)
    S83 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T84 = fd.ops.eq(T82, S83)
    S85 = fd.define_scalar(-3.38953e+38, dtype=DataType.Double)
    T86 = fd.ops.where(T84, S85, T66)
    fd.add_output(T6)
    fd.add_output(T66)
    fd.add_output(T86)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.ones((1, 6), dtype=torch.int64, device='cuda:0'),
    torch.testing.make_tensor((128256, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.ones((1, 6), dtype=torch.int64, device='cuda:0'),
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