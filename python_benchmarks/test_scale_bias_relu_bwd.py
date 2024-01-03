import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def sbr_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    V7 = T2.shape()
    T8 = fd.ops.broadcast_in_dim(T0, shape=V7, broadcast_dims=[1])

    S10 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T11 = fd.ops.where(T1, T2, S10)
    T15 = fd.ops.mul(T11, T8)

    if dtype in PROMOTE_DTYPES:
        T15 = fd.ops.cast(T15, dtype=dtype)
    fd.add_output(T15)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sbr_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    scale = torch.ones(size[-1], device="cuda", dtype=dtype)
    bias = torch.ones(size[-1], device="cuda", dtype=dtype)
    bool_mask = torch.gt(inputs * scale + bias, 0.0)

    with FusionDefinition() as fd:
        sbr_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.relu(inputs * scale + bias)
        eager_output.backward(grads)
        fd.validate([scale, bool_mask, grads], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [scale, bool_mask, grads])
