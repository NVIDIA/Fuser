import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def batchnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
    momentum: float = 0.01,
) -> None:

    input = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    running_mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    running_var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    gamma = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    beta = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        gamma = fd.ops.cast(gamma, dtype=DataType.Float)
        beta = fd.ops.cast(beta, dtype=DataType.Float)
        running_mean = fd.ops.cast(running_mean, dtype=DataType.Float)
        running_var = fd.ops.cast(running_var, dtype=DataType.Float)
    
    var, mean = fd.ops.var_mean(input, axes=[0, 2, 3], correction=0, keepdim=False)
    
    V1 = fd.define_vector([1, input.size(1), 1, 1], dtype=DataType.Int)

    var_bcast = fd.ops.broadcast_in_dim(var, shape=V1, broadcast_dims=[1])
    mean_bcast = fd.ops.broadcast_in_dim(mean, shape=V1, broadcast_dims=[1])

    S1 = fd.define_scalar(eps, dtype=DataType.Double)
    T1 = fd.ops.sub(input, mean_bcast)
    T2 = fd.ops.add(var_bcast, S1)
    T3 = fd.ops.rsqrt(T2)
    T4 = fd.ops.mul(T1, T3)

    gamma = fd.ops.broadcast_in_dim(gamma, shape=V1, broadcast_dims=[1])
    T5 = fd.ops.mul(T4, gamma)
    beta = fd.ops.broadcast_in_dim(beta, shape=V1, broadcast_dims=[1])
    output = fd.ops.add(T5, beta)

    S2 = fd.define_scalar(momentum, dtype=DataType.Double)
    S3 = fd.define_scalar(1-momentum, dtype=DataType.Double)

    T6 = fd.ops.mul(S3, running_mean)
    T7 = fd.ops.mul(S2, mean)
    T8 = fd.ops.add(T6, T7)

    T9 = fd.ops.mul(S3, running_var)
    T10 = fd.ops.mul(S2, var)
    T11 = fd.ops.add(T9, T10)

    if dtype in PROMOTE_DTYPES:
        output = fd.ops.cast(output, dtype=dtype)
        T8 = fd.ops.cast(T8, dtype=dtype)
        T11 = fd.ops.cast(T11, dtype=dtype)
    
    fd.add_output(output)
    fd.add_output(T8)
    fd.add_output(T11)

def batchnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:
    
    input = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    grad = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    gamma = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    beta = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        grad = fd.ops.cast(grad, dtype=DataType.Float)
        mean = fd.ops.cast(mean, dtype=DataType.Float)
        var = fd.ops.cast(var, dtype=DataType.Float)
        gamma = fd.ops.cast(gamma, dtype=DataType.Float)
        beta = fd.ops.cast(beta, dtype=DataType.Float)

    dbeta = fd.ops.sum(grad, axes=[0, 2, 3], keepdim = False)
    
    V1 = [1, input.size(1), 1, 1]

    S1 = fd.define_scalar(eps, dtype=DataType.Double)
    T1 = fd.ops.add(var, S1)
    invstd = fd.ops.rsqrt(T1)
    invstd = fd.ops.broadcast_in_dim(invstd, shape=V1, broadcast_dims=[1])
    mean = fd.ops.broadcast_in_dim(mean, shape=V1, broadcast_dims=[1])
    
    T3 = fd.ops.sub(input, mean)
    input_norm = fd.ops.mul(T3, invstd)

    T4 = fd.ops.mul(grad, input_norm)
    dgamma = fd.ops.sum(T4, axes=[0, 2, 3], keepdim=False)

    dgamma_bcast = fd.ops.broadcast_in_dim(dgamma, shape=V1, broadcast_dims=[1])
    num_features = input.size(0) * input.size(2) * input.size(3)

    gamma_bcast = fd.ops.broadcast_in_dim(gamma, shape=V1, broadcast_dims=[1])
    T5 = fd.ops.mul(gamma_bcast, invstd)
    T6 = fd.ops.div(T5, num_features)
    
    T7 = fd.ops.mul(dgamma_bcast, input_norm)
    T8 = fd.ops.mul(grad, num_features)
    T9 = fd.ops.sub(T8, T7)

    dbeta_bcast = fd.ops.broadcast_in_dim(dbeta, shape=V1, broadcast_dims=[1])
    T10 = fd.ops.sub(T9, dbeta_bcast)

    dinput = fd.ops.mul(T6, T10)

    if dtype in PROMOTE_DTYPES:
        dinput = fd.ops.cast(dinput, dtype=dtype)
        dbeta = fd.ops.cast(dbeta, dtype=dtype)
        dgamma = fd.ops.cast(dgamma, dtype=dtype)
    
    fd.add_output(dinput)
    fd.add_output(dgamma)
    fd.add_output(dbeta)

    
@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_batchnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    running_mean = torch.zeros(size[1], device="cuda", dtype=dtype)
    running_var = torch.ones(size[1], device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        batchnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output = fd.execute([inputs, running_mean, running_var, weight, bias])
        eager_output = torch.nn.functional.batch_norm(
            inputs, running_mean, running_var, weight=weight, bias=bias, training=True,
        )
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, running_mean, running_var, weight, bias])

@pytest.mark.parametrize("size", [(16, 64, 32, 8)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_batchnorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    running_mean = torch.zeros(size[1], device="cuda", dtype=dtype)
    running_var = torch.ones(size[1], device="cuda", dtype=dtype)

    current_mean = inputs.mean(dim=(0, 2, 3))
    current_var = inputs.var(dim=(0, 2, 3), unbiased=False)

    with FusionDefinition() as fd:
        batchnorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output = fd.execute([inputs, grads, current_mean, current_var, weight, bias])
        eager_output = torch.nn.functional.batch_norm(
            inputs, running_mean, running_var, weight=weight, bias=bias, training=True,
        )
        eager_output.backward(grads)
        assert torch.allclose(nvf_output[0], inputs.grad, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - inputs.grad)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, current_mean, current_var, weight, bias])