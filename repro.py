from nvfuser import FusionDefinition, DataType
import torch

inputs = [
    1e-6,
    10,
    4096,
    4096,
    torch.randn(
        (
            1,
            4096,
            4096,
        ),
        dtype=torch.bfloat16,
        device="cuda:0",
    ),
    torch.randn((10, 32), dtype=torch.bfloat16, device="cuda:0"),
    torch.randn(
        (
            1,
            4096,
            4096,
        ),
        dtype=torch.bfloat16,
        device="cuda:0",
    ),
    torch.randn(
        (
            1,
            4096,
            1,
        ),
        dtype=torch.bfloat16,
        device="cuda:0",
    ),
    torch.randn(
        (
            1,
            1,
            4096,
        ),
        dtype=torch.bfloat16,
        device="cuda:0",
    ).expand(1, 4096, 4096),
]


def fusion_func(fd: FusionDefinition) -> None:
    S0 = fd.define_scalar(None, dtype=DataType.Double)
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    T4 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T5 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T6 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T7 = fd.define_tensor(
        shape=[1, -1, 1],
        contiguity=[None, True, None],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T8 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, None, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T9 = fd.ops.cast(T6, dtype=DataType.Float)
    T10 = fd.ops.cast(T6, dtype=DataType.Float)
    T11 = fd.ops.cast(T7, dtype=DataType.Float)
    T12 = fd.ops.rsqrt(T11)
    T13 = fd.ops.cast(T12, dtype=DataType.BFloat16)
    S14 = fd.define_scalar(1, dtype=DataType.Int)
    S15 = fd.define_scalar(4096, dtype=DataType.Int)
    S16 = fd.define_scalar(4096, dtype=DataType.Int)
    V17 = fd.define_vector([S14, S15, S16], dtype=DataType.Int)
    T18 = fd.ops.broadcast_in_dim(T13, shape=V17, broadcast_dims=[0, 1, 2])
    T19 = fd.ops.cast(T6, dtype=DataType.Float)
    T20 = fd.ops.cast(T18, dtype=DataType.Float)
    T21 = fd.ops.mul(T19, T20)
    T22 = fd.ops.cast(T21, dtype=DataType.BFloat16)
    T23 = fd.ops.cast(T8, dtype=DataType.Float)
    T24 = fd.ops.cast(T22, dtype=DataType.Float)
    T25 = fd.ops.cast(T4, dtype=DataType.Float)
    T26 = fd.ops.mul(T25, T24)
    T27 = fd.ops.mul(T25, T23)
    T28 = fd.ops.cast(T27, dtype=DataType.BFloat16)
    T29 = fd.ops.cast(T26, dtype=DataType.BFloat16)
    T30 = fd.ops.cast(T29, dtype=DataType.Float)
    T31 = fd.ops.sum(T30, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T32 = fd.ops.cast(T31, dtype=DataType.BFloat16)
    T33 = fd.ops.cast(T32, dtype=DataType.Float)
    S34 = fd.define_scalar(2.00000, dtype=DataType.Double)
    S35 = fd.ops.reciprocal(S34)
    T36 = fd.ops.mul(T33, S35)
    T37 = fd.ops.cast(T36, dtype=DataType.BFloat16)
    T38 = fd.ops.cast(T28, dtype=DataType.Float)
    T39 = fd.ops.mul(T38, T20)
    T40 = fd.ops.mul(T38, T19)
    T41 = fd.ops.cast(T40, dtype=DataType.BFloat16)
    T42 = fd.ops.cast(T39, dtype=DataType.BFloat16)
    T43 = fd.ops.cast(T41, dtype=DataType.Float)
    T44 = fd.ops.sum(T43, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T45 = fd.ops.cast(T44, dtype=DataType.BFloat16)
    S46 = fd.define_scalar(1, dtype=DataType.Int)
    S47 = fd.define_scalar(4096, dtype=DataType.Int)
    S48 = fd.define_scalar(1, dtype=DataType.Int)
    V49 = fd.define_vector([S46, S47, S48], dtype=DataType.Int)
    T50 = fd.ops.broadcast_in_dim(T45, shape=V49, broadcast_dims=[1])
    T51 = fd.ops.cast(T50, dtype=DataType.Float)
    S52 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T53 = fd.ops.mul(S52, T51)
    S54 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T55 = fd.ops.pow(T12, S54)
    T56 = fd.ops.mul(T53, T55)
    T57 = fd.ops.cast(T56, dtype=DataType.BFloat16)
    T58 = fd.ops.cast(T57, dtype=DataType.Float)
    T59 = fd.ops.cast(T58, dtype=DataType.BFloat16)
    T60 = fd.ops.cast(T59, dtype=DataType.Float)
    S61 = fd.ops.reciprocal(S0)
    T62 = fd.ops.mul(T60, S61)
    T63 = fd.ops.sum(T62, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S64 = fd.define_scalar(1, dtype=DataType.Int)
    S65 = fd.define_scalar(4096, dtype=DataType.Int)
    V66 = fd.define_vector([S64, S65], dtype=DataType.Int)
    T67 = fd.ops.broadcast_in_dim(T63, shape=V66, broadcast_dims=[1])
    S68 = fd.define_scalar(1, dtype=DataType.Int)
    S69 = fd.define_scalar(4096, dtype=DataType.Int)
    S70 = fd.define_scalar(1, dtype=DataType.Int)
    V71 = fd.define_vector([S68, S69, S70], dtype=DataType.Int)
    T72 = fd.ops.broadcast_in_dim(T67, shape=V71, broadcast_dims=[0, 1])
    S73 = fd.define_scalar(1, dtype=DataType.Int)
    S74 = fd.define_scalar(4096, dtype=DataType.Int)
    S75 = fd.define_scalar(4096, dtype=DataType.Int)
    V76 = fd.define_vector([S73, S74, S75], dtype=DataType.Int)
    T77 = fd.ops.broadcast_in_dim(T72, shape=V76, broadcast_dims=[0, 1, 2])
    T78 = fd.ops.cast(T77, dtype=DataType.BFloat16)
    T79 = fd.ops.cast(T78, dtype=DataType.Float)
    T80 = fd.ops.mul(T79, T10)
    T81 = fd.ops.mul(T79, T9)
    T82 = fd.ops.cast(T81, dtype=DataType.BFloat16)
    T83 = fd.ops.cast(T80, dtype=DataType.BFloat16)
    T84 = fd.ops.cast(T42, dtype=DataType.Float)
    T85 = fd.ops.cast(T83, dtype=DataType.Float)
    T86 = fd.ops.add(T84, T85)
    T87 = fd.ops.cast(T86, dtype=DataType.BFloat16)
    T88 = fd.ops.cast(T87, dtype=DataType.Float)
    T89 = fd.ops.cast(T82, dtype=DataType.Float)
    T90 = fd.ops.add(T88, T89)
    T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
    T92 = fd.ops.cast(T91, dtype=DataType.Float)
    T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
    T94 = fd.ops.cast(T92, dtype=DataType.BFloat16)
    T95 = fd.ops.cast(T93, dtype=DataType.Float)
    T96 = fd.ops.cast(T5, dtype=DataType.Float)
    S97 = fd.define_scalar(2.00000, dtype=DataType.Double)
    S98 = fd.ops.reciprocal(S97)
    T99 = fd.ops.mul(T96, S98)
    T100 = fd.ops.cast(T99, dtype=DataType.BFloat16)
    fd.add_output(T100)
    fd.add_output(T37)
    fd.add_output(T94)
    fd.add_output(T95)


torch.manual_seed(0)
with FusionDefinition() as fd:
    fusion_func(fd)

first_time_outputs = fd.execute(inputs)
for _ in range(5):
    outputs = fd.execute(inputs)
    torch.testing.assert_close(first_time_outputs, outputs)
