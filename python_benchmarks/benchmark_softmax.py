import pytest
import pytest_benchmark
from nvfuser import FusionDefinition, DataType
from .core import NVFBenchmark, clearL2Cache
import torch

def softmax_fwd_fusion(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T1 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.max(T1, axes=[2], keepdim=False, dtype=DataType.Null)
    V6 = fd.define_vector([T0.size(0), T0.size(1), 1], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[0, 1])
    V11 = T0.shape()
    T12 = fd.ops.broadcast_in_dim(T7, shape=V11, broadcast_dims=[0, 1, 2])
    T13 = fd.ops.sub(T1, T12)
    T14 = fd.ops.exp(T13)
    T15 = fd.ops.sum(T14, axes=[2], keepdim=False, dtype=DataType.Null)
    T20 = fd.ops.broadcast_in_dim(T15, shape=V6, broadcast_dims=[0, 1])
    T25 = fd.ops.broadcast_in_dim(T20, shape=V11, broadcast_dims=[0, 1, 2])
    T26 = fd.ops.reciprocal(T25)
    T27 = fd.ops.mul(T14, T26)
    T28 = fd.ops.cast(T27, dtype=DataType.BFloat16)
    fd.add_output(T28)


def softmax_bwd_fusion(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
    )
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    T3 = fd.ops.cast(T1, dtype=DataType.Float)
    T4 = fd.ops.mul(T2, T3)
    T5 = fd.ops.sum(T4, axes=[2], keepdim=False, dtype=DataType.Null)
    V9 = fd.define_vector([T0.size(0), T0.size(1), 1], dtype=DataType.Int)
    T10 = fd.ops.broadcast_in_dim(T5, shape=V9, broadcast_dims=[0, 1])
    T11 = fd.ops.cast(T10, dtype=DataType.BFloat16)
    V15 = fd.define_vector([T0.size(0), T0.size(1), T0.size(2)], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T11, shape=V15, broadcast_dims=[0, 1, 2])
    T17 = fd.ops.cast(T16, dtype=DataType.Float)
    T18 = fd.ops.sub(T3, T17)
    T19 = fd.ops.mul(T2, T18)
    T20 = fd.ops.cast(T19, dtype=DataType.BFloat16)
    fd.add_output(T20)


@pytest.mark.parametrize("size", [(64, 128, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_softmax_fwd_benchmark(benchmark, size, dtype):
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]
    with FusionDefinition() as fd:
        softmax_fwd_fusion(fd)

    clearL2cache()

    nvf_bench = NVFBenchmark(benchmark)
    output = nvf_bench(fd.execute, inputs)
    nvf_bench.cleanup()

@pytest.mark.parametrize("size", [(64, 128, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_softmax_bwd_benchmark(benchmark, size, dtype):
    inputs = [
        torch.randn(*size, device="cuda", dtype=dtype),
        torch.randn(*size, device="cuda", dtype=dtype),
    ]
    with FusionDefinition() as fd:
        softmax_bwd_fusion(fd)

    clearL2cache()

    nvf_bench = NVFBenchmark(benchmark)
    output = nvf_bench(fd.execute, inputs)
    nvf_bench.cleanup()