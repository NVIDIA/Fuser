import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def gelu_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    input = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    grad = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        grad = fd.ops.cast(grad, dtype=DataType.Float)
        bias = fd.ops.cast(bias, dtype=DataType.Float)
    S_079 = fd.define_scalar(0.79788456)
    S_004 = fd.define_scalar(0.044715)
    S_010 = fd.define_scalar(0.1070322243)
    V1 = fd.define_vector([1, input.size(-1)], dtype=DataType.Int)
    bias = fd.ops.broadcast_in_dim(bias, shape=V1, broadcast_dims=[1])
    T1 = fd.ops.add(input, bias)
    T2 = fd.ops.mul(T1, S_079)
    T3 = fd.ops.mul(T1, S_004)
    T4 = fd.ops.mul(T3, T1)
    S1 = fd.define_scalar(1.0)
    T5 = fd.ops.add(S1, T4)
    T6 = fd.ops.mul(T2, T5)
    T7 = fd.ops.tanh(T6)
    S2 = fd.define_scalar(0.50)
    T8 = fd.ops.mul(T1, S2)
    T9 = fd.ops.mul(T7, T7)
    T10 = fd.ops.neg(T9)
    T11 = fd.ops.add(T10, S1)
    T12 = fd.ops.mul(T1, S_010)
    T13 = fd.ops.mul(T12, T1)
    T14 = fd.ops.add(T13, S_079)
    T15 = fd.ops.mul(T11, T14)
    T16 = fd.ops.mul(T8, T15)
    T17 = fd.ops.add(T7, S1)
    T18 = fd.ops.mul(T17, S2)
    T19 = fd.ops.add(T16, T18)
    T20 = fd.ops.mul(grad, T19)
    if dtype in PROMOTE_DTYPES:
        T20 = fd.ops.cast(T20, dtype=dtype)
    fd.add_output(T20)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gelu_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    bias = torch.ones(size[-1], device="cuda", dtype=dtype)
    with FusionDefinition() as fd:
        gelu_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.gelu(inputs + bias, approximate="tanh")
        eager_output.backward(grads)
        fd.validate([inputs, grads, bias], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, bias])

    torch.cuda.empty_cache()
