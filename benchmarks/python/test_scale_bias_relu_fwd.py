import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def sbr_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    V7 = T2.shape()
    T8 = fd.ops.broadcast_in_dim(T1, shape=V7, broadcast_dims=[1])
    T11 = fd.ops.mul(T2, T8)

    T18 = fd.ops.broadcast_in_dim(T0, shape=V7, broadcast_dims=[1])
    T20 = fd.ops.add(T11, T18)

    if dtype in PROMOTE_DTYPES:
        T20 = fd.ops.cast(T20, dtype=dtype)

    S22 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T23 = fd.ops.gt(T20, S22)
    T25 = fd.ops.where(T23, T20, S22)

    fd.add_output(T23)
    fd.add_output(T25)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sbr_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.ones(size[-1], device="cuda", dtype=dtype)
    scale = torch.ones(size[-1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        sbr_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    if not disable_validation:
        eager_output = torch.nn.functional.relu(inputs * scale + bias)
        bool_mask = torch.gt(inputs * scale + bias, 0.0)
        fd.validate([bias, scale, inputs], [bool_mask, eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [bias, scale, inputs])
