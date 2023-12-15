import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def layernorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    T3, T4 = fd.ops.var_mean(T0, axes=[1], correction=0, keepdim=False)

    V6 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T3, shape=V6, broadcast_dims=[0])
    T11 = fd.ops.broadcast_in_dim(T4, shape=V6, broadcast_dims=[0])

    S12 = fd.define_scalar(eps, dtype=DataType.Double)
    T13 = fd.ops.add(T7, S12)
    T14 = fd.ops.rsqrt(T13)

    V17 = T0.shape()
    T18 = fd.ops.broadcast_in_dim(T11, shape=V17, broadcast_dims=[0, 1])
    T19 = fd.ops.sub(T0, T18)
    T23 = fd.ops.broadcast_in_dim(T14, shape=V17, broadcast_dims=[0, 1])
    T24 = fd.ops.mul(T19, T23)

    T25 = fd.ops.broadcast_in_dim(T1, shape=V17, broadcast_dims=[1])
    T26 = fd.ops.mul(T24, T25)
    T27 = fd.ops.broadcast_in_dim(T2, shape=V17, broadcast_dims=[1])
    T28 = fd.ops.add(T26, T27)

    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)

    fd.add_output(T28)
    fd.add_output(T4)
    fd.add_output(T14)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layernorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = [
        torch.randn(*size, device="cuda", dtype=dtype),
        torch.randn(size[1], device="cuda", dtype=dtype),
        torch.randn(size[1], device="cuda", dtype=dtype),
    ]

    with FusionDefinition() as fd:
        layernorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.layer_norm(
            inputs[0], inputs[0].shape[1:], weight=inputs[1], bias=inputs[2]
        )

        mean = inputs[0].to(torch.float).mean(dim=-1)
        variance = inputs[0].to(torch.float).var(dim=-1, unbiased=False)
        invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

        fd.validate(inputs, [eager_output, mean, invstd])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)

    torch.cuda.empty_cache()
