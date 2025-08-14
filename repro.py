# CUDA devices:
#  0: NVIDIA RTX 6000 Ada Generation
#  1: NVIDIA RTX 6000 Ada Generation
# torch version: 2.8.0a0+34c6371d24.nvInternal
# cuda version: 13.0
# nvfuser version: 0.2.29+git1fb41ec
import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 1, 16],
        contiguity=[None, None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 16, 48],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.ops.permute(T0, dims=[0, 2, 1])
    T3 = fd.ops.cat([T2, T2], dim=-1, manual_padding=0)
    T4 = fd.ops.cos(T3)
    T5 = fd.ops.sin(T3)
    S6 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T7 = fd.ops.mul(T4, S6)
    S8 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T9 = fd.ops.mul(T5, S8)
    T10 = fd.ops.cast(T7, dtype=DataType.BFloat16)
    T11 = fd.ops.cast(T9, dtype=DataType.BFloat16)
    T17 = fd.ops.broadcast_in_dim(T10, shape=[1, 1, 16, 2], broadcast_dims=[0, 2, 3])
    T23 = fd.ops.broadcast_in_dim(T11, shape=[1, 1, 16, 2], broadcast_dims=[0, 2, 3])
    T29 = fd.ops.broadcast_in_dim(
        T17, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T30 = fd.ops.cast(T29, dtype=DataType.Float)
    T36 = fd.ops.broadcast_in_dim(
        T23, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T37 = fd.ops.cast(T36, dtype=DataType.Float)
    T50 = fd.ops.slice(
        T1,
        start_indices=[0, 0, 0],
        end_indices=[1, 16, 16],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T63 = fd.ops.slice(
        T1,
        start_indices=[0, 0, 16],
        end_indices=[1, 16, 32],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T76 = fd.ops.slice(
        T1,
        start_indices=[0, 0, 32],
        end_indices=[1, 16, 48],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T82 = fd.ops.reshape(T50, new_shape=[1, 16, 16, 1])
    T88 = fd.ops.reshape(T63, new_shape=[1, 16, 16, 1])
    T94 = fd.ops.reshape(T76, new_shape=[1, 16, 16, 1])
    T95 = fd.ops.permute(T82, dims=[0, 2, 1, 3])
    T96 = fd.ops.permute(T88, dims=[0, 2, 1, 3])
    T97 = fd.ops.permute(T94, dims=[0, 2, 1, 3])
    T113 = fd.ops.slice(
        T95,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 16, 16, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T129 = fd.ops.slice(
        T96,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 16, 16, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T135 = fd.ops.broadcast_in_dim(
        T95, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T136 = fd.ops.cast(T135, dtype=DataType.Float)
    T137 = fd.ops.cast(T95, dtype=DataType.Float)
    T138 = fd.ops.neg(T137)
    T139 = fd.ops.cast(T138, dtype=DataType.BFloat16)
    T145 = fd.ops.broadcast_in_dim(
        T96, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T146 = fd.ops.cast(T145, dtype=DataType.Float)
    T147 = fd.ops.cast(T96, dtype=DataType.Float)
    T148 = fd.ops.neg(T147)
    T149 = fd.ops.cast(T148, dtype=DataType.BFloat16)
    T150 = fd.ops.stride_order(T97, stride_order=[3, 2, 1, 0])
    T151 = fd.ops.mul(T136, T30)
    T152 = fd.ops.cat([T139, T113], dim=-1, manual_padding=0)
    T153 = fd.ops.mul(T146, T30)
    T154 = fd.ops.cat([T149, T129], dim=-1, manual_padding=0)
    T160 = fd.ops.broadcast_in_dim(
        T152, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T161 = fd.ops.cast(T160, dtype=DataType.Float)
    T167 = fd.ops.broadcast_in_dim(
        T154, shape=[1, 16, 16, 2], broadcast_dims=[0, 1, 2, 3]
    )
    T168 = fd.ops.cast(T167, dtype=DataType.Float)
    T169 = fd.ops.mul(T161, T37)
    T170 = fd.ops.mul(T168, T37)
    T171 = fd.ops.add(T151, T169)
    T172 = fd.ops.add(T153, T170)
    T173 = fd.ops.cast(T171, dtype=DataType.BFloat16)
    T174 = fd.ops.cast(T172, dtype=DataType.BFloat16)
    T175 = fd.ops.cat([T173, T113], dim=-1, manual_padding=0)
    T176 = fd.ops.cat([T174, T129], dim=-1, manual_padding=0)
    T177 = fd.ops.stride_order(T175, stride_order=[3, 2, 1, 0])
    T178 = fd.ops.stride_order(T176, stride_order=[3, 2, 1, 0])
    T179 = fd.ops.cast(T177, dtype=DataType.Float)
    S180 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T181 = fd.ops.mul(T179, S180)
    T182 = fd.ops.permute(T178, dims=[0, 1, 3, 2])
    T183 = fd.ops.cast(T181, dtype=DataType.BFloat16)
    T184 = fd.ops.cast(T182, dtype=DataType.Float)
    S185 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T186 = fd.ops.mul(T184, S185)
    T187 = fd.ops.cast(T186, dtype=DataType.BFloat16)
    fd.add_output(T97)
    fd.add_output(T150)
    fd.add_output(T176)
    fd.add_output(T183)
    fd.add_output(T187)


with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

inputs = [
    torch.testing.make_tensor((1, 1, 16), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((1, 16, 48), dtype=torch.bfloat16, device="cuda:0"),
]
fd.execute(inputs)
