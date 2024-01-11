import torch
from nvfuser import FusionDefinition, DataType
from python_benchmarks.core import run_benchmark, clear_cuda_cache


def rope_fusion(fd: FusionDefinition) -> None:
    cos = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    q = fd.define_tensor(
        shape=[32, 4096, 32, 128],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    sin = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )

    q = fd.ops.permute(q, dims=[0, 2, 1, 3])
    q_left = fd.ops.slice(q, start_indices=[0, 0, 0, 0], end_indices=[32, 32, 4096, 64], strides=[1, 1, 1, 1])
    q_right = fd.ops.slice(q, start_indices=[0, 0, 0, 64], end_indices=[32, 32, 4096, 128], strides=[1, 1, 1, 1])

    q_right = fd.ops.cast(q_right, dtype=DataType.Float)
    q_right = -q_right
    q_right = fd.ops.cast(q_right, dtype=DataType.BFloat16)

    q_rotated = fd.ops.cat([q_right, q_left], dim=-1)

    cos = fd.ops.broadcast_in_dim(cos, shape=[1, 1, 4096, 128], broadcast_dims=[2, 3])
    sin = fd.ops.broadcast_in_dim(sin, shape=[1, 1, 4096, 128], broadcast_dims=[2, 3])
    fd.add_output(q * cos + q_rotated * sin)


def qkv_split_rope_fusion(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    S3 = fd.define_scalar(32, dtype=DataType.Int)
    S4 = fd.define_scalar(4096, dtype=DataType.Int)
    S5 = fd.define_scalar(32, dtype=DataType.Int)
    S6 = fd.define_scalar(3, dtype=DataType.Int)
    S7 = fd.define_scalar(128, dtype=DataType.Int)
    V8 = fd.define_vector([S3, S4, S5, S6, S7], dtype=DataType.Int)
    T9 = fd.ops.reshape(T1, new_shape=V8)
    T10 = fd.ops.permute(T9, dims=[0, 2, 3, 1, 4])
    T11 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 0, 0, 0],
        end_indices=[32, 32, 1, 4096, 128],
        strides=[1, 1, 1, 1, 1],
    )
    T12 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 1, 0, 0],
        end_indices=[32, 32, 2, 4096, 128],
        strides=[1, 1, 1, 1, 1],
    )
    T13 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 2, 0, 0],
        end_indices=[32, 32, 3, 4096, 128],
        strides=[1, 1, 1, 1, 1],
    )
    S14 = fd.define_scalar(32, dtype=DataType.Int)
    S15 = fd.define_scalar(32, dtype=DataType.Int)
    S16 = fd.define_scalar(4096, dtype=DataType.Int)
    S17 = fd.define_scalar(128, dtype=DataType.Int)
    V18 = fd.define_vector([S14, S15, S16, S17], dtype=DataType.Int)
    T19 = fd.ops.reshape(T11, new_shape=V18)
    S20 = fd.define_scalar(32, dtype=DataType.Int)
    S21 = fd.define_scalar(32, dtype=DataType.Int)
    S22 = fd.define_scalar(4096, dtype=DataType.Int)
    S23 = fd.define_scalar(128, dtype=DataType.Int)
    V24 = fd.define_vector([S20, S21, S22, S23], dtype=DataType.Int)
    T25 = fd.ops.reshape(T12, new_shape=V24)
    S26 = fd.define_scalar(32, dtype=DataType.Int)
    S27 = fd.define_scalar(32, dtype=DataType.Int)
    S28 = fd.define_scalar(4096, dtype=DataType.Int)
    S29 = fd.define_scalar(128, dtype=DataType.Int)
    V30 = fd.define_vector([S26, S27, S28, S29], dtype=DataType.Int)
    T31 = fd.ops.reshape(T13, new_shape=V30)
    T32 = fd.ops.slice(
        T19,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 128],
        strides=[1, 1, 1, 1],
    )
    T33 = fd.ops.slice(
        T32,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 64],
        strides=[1, 1, 1, 1],
    )
    T34 = fd.ops.slice(
        T32,
        start_indices=[0, 0, 0, 64],
        end_indices=[32, 32, 4096, 128],
        strides=[1, 1, 1, 1],
    )
    T35 = fd.ops.cast(T34, dtype=DataType.Float)
    T36 = fd.ops.neg(T35)
    T37 = fd.ops.cast(T36, dtype=DataType.BFloat16)
    T38 = fd.ops.cat([T37, T33], dim=-1)
    S39 = fd.define_scalar(32, dtype=DataType.Int)
    S40 = fd.define_scalar(32, dtype=DataType.Int)
    S41 = fd.define_scalar(4096, dtype=DataType.Int)
    S42 = fd.define_scalar(128, dtype=DataType.Int)
    V43 = fd.define_vector([S39, S40, S41, S42], dtype=DataType.Int)
    T44 = fd.ops.broadcast_in_dim(T0, shape=V43, broadcast_dims=[2, 3])
    T45 = fd.ops.cast(T32, dtype=DataType.Float)
    T46 = fd.ops.cast(T44, dtype=DataType.Float)
    T47 = fd.ops.mul(T45, T46)
    S48 = fd.define_scalar(32, dtype=DataType.Int)
    S49 = fd.define_scalar(32, dtype=DataType.Int)
    S50 = fd.define_scalar(4096, dtype=DataType.Int)
    S51 = fd.define_scalar(128, dtype=DataType.Int)
    V52 = fd.define_vector([S48, S49, S50, S51], dtype=DataType.Int)
    T53 = fd.ops.broadcast_in_dim(T2, shape=V52, broadcast_dims=[2, 3])
    T54 = fd.ops.cast(T38, dtype=DataType.Float)
    T55 = fd.ops.cast(T53, dtype=DataType.Float)
    T56 = fd.ops.mul(T54, T55)
    T57 = fd.ops.add(T47, T56)
    T58 = fd.ops.cast(T57, dtype=DataType.BFloat16)
    T59 = fd.ops.slice(
        T25,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 128],
        strides=[1, 1, 1, 1],
    )
    T60 = fd.ops.slice(
        T59,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 64],
        strides=[1, 1, 1, 1],
    )
    T61 = fd.ops.slice(
        T59,
        start_indices=[0, 0, 0, 64],
        end_indices=[32, 32, 4096, 128],
        strides=[1, 1, 1, 1],
    )
    T62 = fd.ops.cast(T61, dtype=DataType.Float)
    T63 = fd.ops.neg(T62)
    T64 = fd.ops.cast(T63, dtype=DataType.BFloat16)
    T65 = fd.ops.cat([T64, T60], dim=-1)
    S66 = fd.define_scalar(32, dtype=DataType.Int)
    S67 = fd.define_scalar(32, dtype=DataType.Int)
    S68 = fd.define_scalar(4096, dtype=DataType.Int)
    S69 = fd.define_scalar(128, dtype=DataType.Int)
    V70 = fd.define_vector([S66, S67, S68, S69], dtype=DataType.Int)
    T71 = fd.ops.broadcast_in_dim(T0, shape=V70, broadcast_dims=[2, 3])
    T72 = fd.ops.cast(T59, dtype=DataType.Float)
    T73 = fd.ops.cast(T71, dtype=DataType.Float)
    T74 = fd.ops.mul(T72, T73)
    S75 = fd.define_scalar(32, dtype=DataType.Int)
    S76 = fd.define_scalar(32, dtype=DataType.Int)
    S77 = fd.define_scalar(4096, dtype=DataType.Int)
    S78 = fd.define_scalar(128, dtype=DataType.Int)
    V79 = fd.define_vector([S75, S76, S77, S78], dtype=DataType.Int)
    T80 = fd.ops.broadcast_in_dim(T2, shape=V79, broadcast_dims=[2, 3])
    T81 = fd.ops.cast(T65, dtype=DataType.Float)
    T82 = fd.ops.cast(T80, dtype=DataType.Float)
    T83 = fd.ops.mul(T81, T82)
    T84 = fd.ops.add(T74, T83)
    T85 = fd.ops.cast(T84, dtype=DataType.BFloat16)
    T86 = fd.ops.slice(
        T19,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 0],
        strides=[1, 1, 1, 1],
    )
    T87 = fd.ops.cat([T58, T86], dim=-1)
    T88 = fd.ops.slice(
        T25,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 0],
        strides=[1, 1, 1, 1],
    )
    T89 = fd.ops.cat([T85, T88], dim=-1)
    fd.add_output(T87)
    fd.add_output(T89)
    fd.add_output(T31)


def test_qkv_split_rope(benchmark):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        qkv_split_rope_fusion(fd)

    inputs = [
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (4096, 128), (128, 1)
        ),
        torch.randn((1610612736,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (32, 4096, 12288), (50331648, 12288, 1)
        ),
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (4096, 128), (128, 1)
        ),
    ]

    run_benchmark(benchmark, fd.execute, inputs)


def test_rope(benchmark):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        rope_fusion(fd)

    inputs = [
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (4096, 128), (128, 1)
        ),
        torch.randn(32, 4096, 32, 128, dtype=torch.bfloat16, device="cuda:0"),
        torch.randn((524288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (4096, 128), (128, 1)
        ),
    ]

    run_benchmark(benchmark, fd.execute, inputs)
