import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def instancenorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:

    input = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    gamma = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    beta = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        gamma = fd.ops.cast(gamma, dtype=DataType.Float)
        beta = fd.ops.cast(beta, dtype=DataType.Float)
    
    var, mean = fd.ops.var_mean(input, axes=[2, 3], correction=0, keepdim=True)
    
    V1 = fd.define_vector([input.size(0), input.size(1), 1, 1], dtype=DataType.Int)
    S1 = fd.define_scalar(eps, dtype=DataType.Double)
    T1 = fd.ops.sub(input, mean)
    T2 = fd.ops.add(var, S1)
    T3 = fd.ops.rsqrt(T2)
    T4 = fd.ops.mul(T1, T3)

    gamma = fd.ops.broadcast_in_dim(gamma, shape=V1, broadcast_dims=[1])
    T5 = fd.ops.mul(T4, gamma)
    beta = fd.ops.broadcast_in_dim(beta, shape=V1, broadcast_dims=[1])
    output = fd.ops.add(T5, beta)

    if dtype in PROMOTE_DTYPES:
        output = fd.ops.cast(output, dtype=dtype)
    
    fd.add_output(output)

@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_instancenorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        instancenorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output = fd.execute([inputs, weight, bias])
        eager_output = torch.nn.functional.instance_norm(
            inputs, weight=weight, bias=bias
        )
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weight, bias])
