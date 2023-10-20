import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES, RESNET_BATCHNORM_SIZES

def batchnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
    momentum: float = 0.01,
) -> None:

    input = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    running_mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    running_var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)
        bias = fd.ops.cast(bias, dtype=DataType.Float)
        
    var, mean = fd.ops.var_mean(input, axes=[0, 2, 3], correction=0, keepdim=False)
    
    bcast_shape = fd.define_vector([1, input.size(1), 1, 1], dtype=DataType.Int)

    var_bcast = fd.ops.broadcast_in_dim(var, shape=bcast_shape, broadcast_dims=[1])
    mean_bcast = fd.ops.broadcast_in_dim(mean, shape=bcast_shape, broadcast_dims=[1])

    eps = fd.define_scalar(eps, dtype=DataType.Double)
    x_sub_mean = fd.ops.sub(input, mean_bcast)
    var_eps = fd.ops.add(var_bcast, eps)
    invstd = fd.ops.rsqrt(var_eps)
    x_norm = fd.ops.mul(x_sub_mean, invstd)

    weight = fd.ops.broadcast_in_dim(weight, shape=bcast_shape, broadcast_dims=[1])
    x_scaled = fd.ops.mul(x_norm, weight)
    bias = fd.ops.broadcast_in_dim(bias, shape=bcast_shape, broadcast_dims=[1])
    output = fd.ops.add(x_scaled, bias)

    rev_momentum = fd.define_scalar(1-momentum, dtype=DataType.Double)
    momentum = fd.define_scalar(momentum, dtype=DataType.Double)
    
    running_mean = fd.ops.add(fd.ops.mul(momentum, mean), fd.ops.mul(rev_momentum, running_mean))
    running_var = fd.ops.add(fd.ops.mul(momentum, var), fd.ops.mul(rev_momentum, running_var))

    if dtype in PROMOTE_DTYPES:
        output = fd.ops.cast(output, dtype=dtype)
    
    fd.add_output(output)
    fd.add_output(mean)
    fd.add_output(invstd)

def batchnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:
    
    input = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    grad = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    
    running_mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    running_var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    invstd = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        grad = fd.ops.cast(grad, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)

    reduction_axes = [0, 2, 3]
    num_features = input.size(0) * input.size(2) * input.size(3)
    norm = fd.ops.reciprocal(num_features)

    bcast_shape = [1, input.size(1), 1, 1]
    mean = fd.ops.broadcast_in_dim(mean, shape=bcast_shape, broadcast_dims=[1])
    invstd = fd.ops.broadcast_in_dim(invstd, shape=bcast_shape, broadcast_dims=[1])
    
    grad_sum = fd.ops.sum(grad, axes=reduction_axes, keepdim = True)
    
    x_sub_mean = fd.ops.sub(input, mean)
    dot_p = fd.ops.sum(fd.ops.mul(grad, x_sub_mean), axes=reduction_axes, keepdim=True)
    
    grad_mean = fd.ops.mul(grad_sum, norm)
    proj_scale = fd.ops.mul(fd.ops.mul(dot_p, norm), fd.ops.mul(invstd, invstd))
    
    weight = fd.ops.broadcast_in_dim(weight, shape=bcast_shape, broadcast_dims=[1])
    grad_scale = fd.ops.mul(weight, invstd)
    proj = fd.ops.mul(proj_scale, x_sub_mean)
    
    grad_input = fd.ops.mul(fd.ops.sub(fd.ops.sub(grad, proj), grad_mean), grad_scale)
    grad_weight = fd.ops.mul(dot_p, invstd)
    grad_bias = grad_sum

    if dtype in PROMOTE_DTYPES:
        grad_input = fd.ops.cast(grad_input, dtype=dtype)
        grad_weight = fd.ops.cast(grad_weight, dtype=dtype)
        grad_bias = fd.ops.cast(grad_bias, dtype=dtype)
    
    fd.add_output(grad_input)
    fd.add_output(grad_weight)
    fd.add_output(grad_bias)
    
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
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    with FusionDefinition() as fd:
        batchnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output = fd.execute([inputs, weight, bias, running_mean, running_var])
        # PyTorch expects running mean and variance to be of same type as input.
        eager_output = torch.nn.functional.batch_norm(
            inputs, running_mean.to(dtype), running_var.to(dtype), weight=weight, bias=bias, training=True,
        )
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weight, bias, running_mean, running_var])

@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_batchnorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    mean = inputs.to(torch.float).mean(dim=(0, 2, 3))
    var = inputs.to(torch.float).var(dim=(0, 2, 3), unbiased=False)
    invstd = (1.0 / torch.sqrt(var + eps))

    with FusionDefinition() as fd:
        batchnorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output = fd.execute([inputs, grads, weight, running_mean, running_var, mean, invstd])

        # PyTorch expects running mean and variance to be of same type as input.
        eager_output = torch.nn.functional.batch_norm(
            inputs, running_mean.to(dtype), running_var.to(dtype), weight=weight, bias=bias, training=True,
        )
        eager_output.backward(grads)
        assert torch.allclose(nvf_output[0], inputs.grad, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - inputs.grad)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, weight, running_mean, running_var, mean, invstd])