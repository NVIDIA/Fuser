import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def silu_mul_fwd_fusion(fd: FusionDefinition, dtype: DataType):
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
    T2 = fd.ops.neg(T0)
    T3 = fd.ops.exp(T2)
    S4 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T5 = fd.ops.add(S4, T3)
    T6 = fd.ops.reciprocal(T5)
    T7 = fd.ops.mul(T0, T6)
    T8 = fd.ops.mul(T7, T1)
    if dtype in PROMOTE_DTYPES:
        T8 = fd.ops.cast(T8, dtype=dtype)
    fd.add_output(T8)


def silu_mul_fwd_fn(inputs: list):  # [in_tensor1, in_tensor_2]
    return torch.nn.functional.silu(inputs[0]) * inputs[1]


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_silu_mul_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    inputs = [torch.randn(*size, device="cuda", dtype=dtype) for _ in range(2)]

    with FusionDefinition() as fd:
        silu_mul_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    if not disable_validation:
        eager_output = silu_mul_fwd_fn(inputs)
        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_silu_mul_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    compile: bool,
):
    clear_cuda_cache()
    inputs = [torch.randn(*size, device="cuda", dtype=dtype) for _ in range(2)]

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        torch.compile(silu_mul_fwd_fn) if compile else silu_mul_fwd_fn,
        inputs,
    )
