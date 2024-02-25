import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def softmax_fwd_fusion(
    fd: FusionDefinition, dtype: DataType, reduction_axis: int
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.max(T0, dims=[reduction_axis], keepdim=False, dtype=DataType.Null)

    if reduction_axis:
        V6 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    else:
        V6 = fd.define_vector([1, T0.size(1)], dtype=DataType.Int)
    bcast_dim = 1 - reduction_axis

    T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[bcast_dim])

    V11 = T0.shape()
    T12 = fd.ops.broadcast_in_dim(T7, shape=V11, broadcast_dims=[0, 1])
    T13 = fd.ops.sub(T0, T12)
    T14 = fd.ops.exp(T13)
    T15 = fd.ops.sum(T14, dims=[reduction_axis], keepdim=False, dtype=DataType.Null)

    T20 = fd.ops.broadcast_in_dim(T15, shape=V6, broadcast_dims=[bcast_dim])
    T25 = fd.ops.broadcast_in_dim(T20, shape=V11, broadcast_dims=[0, 1])

    T26 = fd.ops.reciprocal(T25)
    T27 = fd.ops.mul(T14, T26)

    if dtype in PROMOTE_DTYPES:
        T27 = fd.ops.cast(T27, dtype=dtype)
    fd.add_output(T27)


def softmax_fwd_fn(inputs: list):  # [in_tensor, reduction_axis]
    return torch.nn.functional.softmax(inputs[0], inputs[1])


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
def test_softmax_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    inputs = [torch.randn(size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        softmax_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = softmax_fwd_fn([inputs[0], reduction_axis])
        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
def test_softmax_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    compile: bool,
):
    clear_cuda_cache()
    input = torch.randn(size, device="cuda", dtype=dtype)
    run_benchmark(
        benchmark,
        torch.compile(softmax_fwd_fn) if compile else softmax_fwd_fn,
        [input, reduction_axis],
    )
