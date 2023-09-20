import torch
from nvfuser import FusionDefinition, DataType


def clearL2Cache():
    n_elements = 40 * 1024 * 1024 // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


inputs = [
    torch.randn(16, 25, 512, 512, device="cuda"),
    torch.randn(1, 1, 512, 512, device="cuda"),
]


def nvfuser_fusion_id1(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[1, 1, -1, -1],
        contiguity=[None, None, False, True],
        dtype=DataType.Float,
        is_cpu=False,
    )
    S2 = fd.define_scalar(0.125000, dtype=DataType.Float)
    T3 = fd.ops.mul(T0, S2)
    S4 = fd.define_scalar(0.00000, dtype=DataType.Float)
    T5 = fd.ops.eq(T1, S4)
    S6 = fd.define_scalar(16, dtype=DataType.Int)
    S7 = fd.define_scalar(25, dtype=DataType.Int)
    S8 = fd.define_scalar(512, dtype=DataType.Int)
    S9 = fd.define_scalar(512, dtype=DataType.Int)
    V10 = fd.define_vector([S6, S7, S8, S9], dtype=DataType.Int)
    T11 = fd.ops.broadcast_in_dim(T5, shape=V10, broadcast_dims=[0, 1, 2, 3])
    S12 = fd.define_scalar(float("-inf"), dtype=DataType.Float)
    T13 = fd.ops.where(T11, S12, T3)
    T14 = fd.ops.max(T13, axes=[3], keepdim=False, dtype=DataType.Null)
    S15 = fd.define_scalar(16, dtype=DataType.Int)
    S16 = fd.define_scalar(25, dtype=DataType.Int)
    S17 = fd.define_scalar(512, dtype=DataType.Int)
    S18 = fd.define_scalar(1, dtype=DataType.Int)
    V19 = fd.define_vector([S15, S16, S17, S18], dtype=DataType.Int)
    T20 = fd.ops.broadcast_in_dim(T14, shape=V19, broadcast_dims=[0, 1, 2])
    S21 = fd.define_scalar(16, dtype=DataType.Int)
    S22 = fd.define_scalar(25, dtype=DataType.Int)
    S23 = fd.define_scalar(512, dtype=DataType.Int)
    S24 = fd.define_scalar(512, dtype=DataType.Int)
    V25 = fd.define_vector([S21, S22, S23, S24], dtype=DataType.Int)
    T26 = fd.ops.broadcast_in_dim(T20, shape=V25, broadcast_dims=[0, 1, 2, 3])
    T27 = fd.ops.sub(T13, T26)
    T28 = fd.ops.exp(T27)
    T29 = fd.ops.sum(T28, axes=[3], keepdim=False, dtype=DataType.Null)
    S30 = fd.define_scalar(16, dtype=DataType.Int)
    S31 = fd.define_scalar(25, dtype=DataType.Int)
    S32 = fd.define_scalar(512, dtype=DataType.Int)
    S33 = fd.define_scalar(1, dtype=DataType.Int)
    V34 = fd.define_vector([S30, S31, S32, S33], dtype=DataType.Int)
    T35 = fd.ops.broadcast_in_dim(T29, shape=V34, broadcast_dims=[0, 1, 2])
    S36 = fd.define_scalar(16, dtype=DataType.Int)
    S37 = fd.define_scalar(25, dtype=DataType.Int)
    S38 = fd.define_scalar(512, dtype=DataType.Int)
    S39 = fd.define_scalar(512, dtype=DataType.Int)
    V40 = fd.define_vector([S36, S37, S38, S39], dtype=DataType.Int)
    T41 = fd.ops.broadcast_in_dim(T35, shape=V40, broadcast_dims=[0, 1, 2, 3])
    T42 = fd.ops.reciprocal(T41)
    T43 = fd.ops.mul(T28, T42)
    S44 = fd.define_scalar(0.00000, dtype=DataType.Float)
    S45 = fd.define_scalar(1.00000, dtype=DataType.Float)
    S46 = fd.define_scalar(16, dtype=DataType.Int)
    S47 = fd.define_scalar(25, dtype=DataType.Int)
    S48 = fd.define_scalar(512, dtype=DataType.Int)
    S49 = fd.define_scalar(512, dtype=DataType.Int)
    T50 = fd.ops.uniform(S44, S45, shape=[S46, S47, S48, S49], dtype=DataType.Float)
    S51 = fd.define_scalar(0.900000, dtype=DataType.Float)
    T52 = fd.ops.lt(T50, S51)
    T53 = fd.ops.cast(T52, dtype=DataType.Float)
    T54 = fd.ops.mul(T43, T53)
    S55 = fd.define_scalar(1.11111, dtype=DataType.Float)
    T56 = fd.ops.mul(T54, S55)
    fd.add_output(T11)
    fd.add_output(T43)
    fd.add_output(T52)
    fd.add_output(T56)


with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

for _ in range(5):
    clearL2Cache()
    out = fd.execute(inputs)
