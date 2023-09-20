import torch
from nvfuser import FusionDefinition, DataType


def clearL2Cache():
    n_elements = 40 * 1024 * 1024 // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


inputs = [
    1.1111111,
    1.0,
    1,
    1,
    1,
    torch.randn(1, 1, 512, 512, device="cuda") > 0.0,
    torch.randn(16, 25, 512, 512, device="cuda"),
    torch.randn(16, 25, 512, 512, device="cuda") > 0.0,
    torch.randn(16, 25, 512, 512, device="cuda"),
]


def nvfuser_fusion_id9(fd: FusionDefinition) -> None:
    S0 = fd.define_scalar(None, dtype=DataType.Double)
    S1 = fd.define_scalar(None, dtype=DataType.Double)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    S4 = fd.define_scalar(None, dtype=DataType.Int)
    T5 = fd.define_tensor(
        shape=[1, 1, -1, -1],
        contiguity=[None, None, True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )
    T6 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    T7 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )
    T8 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    T9 = fd.ops.cast(T7, dtype=DataType.Float)
    T10 = fd.ops.mul(T8, S1)
    T11 = fd.ops.mul(T10, T9)
    T12 = fd.ops.mul(T6, T11)
    T13 = fd.ops.sum(T12, axes=[3], keepdim=False, dtype=DataType.Null)
    S14 = fd.define_scalar(16, dtype=DataType.Int)
    S15 = fd.define_scalar(25, dtype=DataType.Int)
    S16 = fd.define_scalar(512, dtype=DataType.Int)
    S17 = fd.define_scalar(1, dtype=DataType.Int)
    V18 = fd.define_vector([S14, S15, S16, S17], dtype=DataType.Int)
    T19 = fd.ops.broadcast_in_dim(T13, shape=V18, broadcast_dims=[0, 1, 2])
    S20 = fd.define_scalar(16, dtype=DataType.Int)
    S21 = fd.define_scalar(25, dtype=DataType.Int)
    S22 = fd.define_scalar(512, dtype=DataType.Int)
    S23 = fd.define_scalar(512, dtype=DataType.Int)
    V24 = fd.define_vector([S20, S21, S22, S23], dtype=DataType.Int)
    T25 = fd.ops.broadcast_in_dim(T19, shape=V24, broadcast_dims=[0, 1, 2, 3])
    T26 = fd.ops.sub(T11, T25)
    T27 = fd.ops.mul(T6, T26)
    S28 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T29 = fd.ops.where(T5, S28, T27)
    T30 = fd.ops.mul(T29, S0)
    fd.add_output(T30)


with FusionDefinition() as fd:
    nvfuser_fusion_id9(fd)

for _ in range(5):
    clearL2Cache()
    out = fd.execute(inputs)
