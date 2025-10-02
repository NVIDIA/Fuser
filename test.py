import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id9(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1, -1, -1],
        contiguity=[True, True, True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T3 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T4 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T5 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    T6 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    T7 = fd.define_tensor(
        shape=[-1, -1, -1, -1, -1],
        contiguity=[True, True, True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    (
        S8,
        S9,
        S10,
        S11,
        S12,
    ) = fd.ops.tensor_sizes(T0)
    S13 = fd.define_scalar(1, dtype=DataType.Int)
    S14 = fd.ops.mul(S13, S10)
    S15 = fd.ops.mul(S14, S11)
    S16 = fd.ops.mul(S15, S12)
    T17 = fd.ops.broadcast(T5, is_broadcast_dim=[False, False, True, True, True])
    S18 = fd.ops.reciprocal(S16)
    T19 = fd.ops.sum(T7, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
    T20 = fd.ops.sub(T0, T17)
    T21 = fd.ops.mul(T7, T20)
    T22 = fd.ops.sum(T21, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
    T23 = fd.ops.mul(T19, S18)
    T24 = fd.ops.broadcast(T23, is_broadcast_dim=[False, False, True, True, True])
    T25 = fd.ops.mul(T22, S18)
    T26 = fd.ops.mul(T6, T6)
    T27 = fd.ops.mul(T25, T26)
    T28 = fd.ops.broadcast(T27, is_broadcast_dim=[False, False, True, True, True])
    T29 = fd.ops.broadcast(T6, is_broadcast_dim=[False, False, True, True, True])
    T30 = fd.ops.broadcast(T1, is_broadcast_dim=[True, False, True, True, True])
    T31 = fd.ops.mul(T29, T30)
    T32 = fd.ops.sub(T0, T17)
    T33 = fd.ops.mul(T32, T28)
    T34 = fd.ops.sub(T7, T33)
    T35 = fd.ops.sub(T34, T24)
    T36 = fd.ops.mul(T35, T31)
    T37 = fd.ops.mul(T22, T6)
    T38 = fd.ops.sum(T37, dims=[0], keepdim=False, dtype=DataType.Null)
    T39 = fd.ops.sum(T19, dims=[0], keepdim=False, dtype=DataType.Null)
    fd.add_output(T36)
    fd.add_output(T38)
    fd.add_output(T39)


with FusionDefinition() as fd:
    nvfuser_fusion_id9(fd)

inputs = [
    torch.testing.make_tensor((5, 7, 3, 3, 3), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((7,), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((7,), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((7,), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((7,), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((5, 7), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((5, 7), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((5, 7, 3, 3, 3), dtype=torch.float32, device="cuda:0"),
]
fd.execute(inputs)
