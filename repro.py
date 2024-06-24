import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id9(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, None],
        dtype=DataType.Bool,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Bool,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T15 = fd.ops.broadcast_in_dim(T2, shape=[16, 16, 32], broadcast_dims=[0, 1])
    T21 = fd.ops.slice(
        T1, start_indices=[0, 0, 16], end_indices=[16, 16, 32], strides=[1, 1, 1]
    )
    S22 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T23 = fd.ops.where(T15, S22, T1)
    S24 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T25 = fd.ops.where(T0, S24, T23)
    fd.add_output(T21)
    fd.add_output(T25)


with FusionDefinition() as fd:
    nvfuser_fusion_id9(fd)

inputs = [
    torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:1").as_strided(
        (16, 16, 32), (16, 1, 0)
    ),
    torch.randn((8192,), dtype=torch.float32, device="cuda:1").as_strided(
        (16, 16, 32), (512, 32, 1)
    ),
    torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:1").as_strided(
        (16, 16), (16, 1)
    ),
]
fd.execute(inputs)
