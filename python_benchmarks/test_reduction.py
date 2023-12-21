import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def reduction_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    reduction_axis: int,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.sum(T0, axes=[reduction_axis], keepdim=False)
    if dtype in PROMOTE_DTYPES:
        T2 = fd.ops.cast(T2, dtype=dtype)
    fd.add_output(T2)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
def test_reduction_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        reduction_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = torch.sum(inputs[0], dim=reduction_axis)
        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
