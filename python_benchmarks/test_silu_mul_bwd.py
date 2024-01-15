import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def silu_mul_bwd_fusion(fd: FusionDefinition, dtype: DataType):
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T2 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    T3 = fd.ops.neg(T1)
    T4 = fd.ops.exp(T3)
    S5 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T6 = fd.ops.add(S5, T4)
    T7 = fd.ops.reciprocal(T6)
    T8 = fd.ops.mul(T1, T7)
    T9 = fd.ops.mul(T0, T2)
    T10 = fd.ops.mul(T0, T8)
    T11 = fd.ops.mul(T9, T7)
    T12 = fd.ops.mul(T9, T1)
    T13 = fd.ops.neg(T12)
    T14 = fd.ops.mul(T13, T7)
    T15 = fd.ops.mul(T14, T7)
    S16 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T17 = fd.ops.mul(S16, T15)
    T18 = fd.ops.mul(T17, T4)
    T19 = fd.ops.neg(T18)
    T20 = fd.ops.add(T11, T19)
    if dtype in PROMOTE_DTYPES:
        T10 = fd.ops.cast(T10, dtype=dtype)
        T20 = fd.ops.cast(T20, dtype=dtype)
    fd.add_output(T10)
    fd.add_output(T20)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_silu_mul_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    x = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    with FusionDefinition() as fd:
        silu_mul_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    if not disable_validation:
        eager_output = torch.nn.functional.silu(x) * y
        eager_output.backward(grads)
        fd.validate([grads, x, y], [y.grad, x.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [grads, x, y])
