import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
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
    T2 = fd.ops.max(T0, axes=[reduction_axis], keepdim=False, dtype=DataType.Null)

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
    T15 = fd.ops.sum(T14, axes=[reduction_axis], keepdim=False, dtype=DataType.Null)

    T20 = fd.ops.broadcast_in_dim(T15, shape=V6, broadcast_dims=[bcast_dim])
    T25 = fd.ops.broadcast_in_dim(T20, shape=V11, broadcast_dims=[0, 1])

    T26 = fd.ops.reciprocal(T25)
    T27 = fd.ops.mul(T14, T26)

    if dtype in PROMOTE_DTYPES:
        T27 = fd.ops.cast(T27, dtype=dtype)
    fd.add_output(T27)


def softmax_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, reduction_axis: int
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype is not DataType.Float:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T4 = fd.ops.mul(T0, T1)
    T5 = fd.ops.sum(T4, axes=[reduction_axis], keepdim=False, dtype=DataType.Null)

    if reduction_axis:
        V9 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    else:
        V9 = fd.define_vector([1, T0.size(1)], dtype=DataType.Int)
    bcast_dim = 1 - reduction_axis

    T10 = fd.ops.broadcast_in_dim(T5, shape=V9, broadcast_dims=[bcast_dim])

    if dtype is not DataType.Float:
        T10 = fd.ops.cast(T10, dtype=dtype)

    V15 = fd.define_vector([T0.size(0), T0.size(1)], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T10, shape=V15, broadcast_dims=[0, 1])

    if dtype is not DataType.Float:
        T16 = fd.ops.cast(T16, dtype=DataType.Float)

    T18 = fd.ops.sub(T1, T16)
    T19 = fd.ops.mul(T0, T18)

    if dtype is not DataType.Float:
        T19 = fd.ops.cast(T19, dtype=dtype)
    fd.add_output(T19)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
def test_softmax_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        softmax_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        nvf_output = fd.execute(inputs)
        eager_output = torch.nn.functional.softmax(inputs[0], dim=reduction_axis)
        assert torch.allclose(
            nvf_output[0], eager_output, rtol=1e-3, atol=1e-3
        ), f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)

    torch.cuda.empty_cache()


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
def test_softmax_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [
        torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(*size, device="cuda", dtype=dtype),
    ]

    with FusionDefinition() as fd:
        softmax_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = torch.nn.functional.softmax(inputs[0], dim=reduction_axis)
        eager_output.backward(inputs[1])

        nvf_output = fd.execute([eager_output, inputs[1]])
        assert torch.allclose(
            nvf_output[0], inputs[0].grad, rtol=1e-3, atol=1e-3
        ), f"{torch.max(nvf_output[0] - inputs[0].grad)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)

    torch.cuda.empty_cache()
