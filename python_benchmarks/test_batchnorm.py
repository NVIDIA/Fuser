import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES, RESNET_BATCHNORM_SIZES
from .normalization import norm_fwd_fusion, norm_bwd_fusion

@pytest.mark.parametrize("size",  generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [False])
def test_batchnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    # Size is assumed to be in the order N, C, ...
    input_size = size
    if channels_last:
        input_size = (size[0], *size[2:], size[1])
    num_dims = len(size)

    inputs = torch.randn(*input_size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    with FusionDefinition() as fd:
        norm_fwd_fusion(
            fd = fd, 
            dtype = torch_dtype_to_nvfuser_dtype(dtype), 
            norm = "batch_norm", 
            num_dims=num_dims, 
            channels_last=channels_last)

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
@pytest.mark.parametrize("channels_last", [False])
def test_batchnorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):  
    # Size is assumed to be in the order N, C, ...
    input_size = size
    if channels_last:
        input_size = (size[0], *size[2:], size[1])
    num_dims = len(size)

    inputs = torch.randn(*input_size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*input_size, device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    channel_dim = 1 if not channels_last else len(size) - 1
    reduction_axes = [i for i in range(len(size)) if i != channel_dim]
    mean = inputs.to(torch.float).mean(dim = reduction_axes)
    var = inputs.to(torch.float).var(dim = reduction_axes, unbiased=False)
    invstd = (1.0 / torch.sqrt(var + eps))

    with FusionDefinition() as fd:
        norm_bwd_fusion(fd = fd, 
            dtype = torch_dtype_to_nvfuser_dtype(dtype), 
            norm = "batch_norm", 
            num_dims=num_dims, 
            channels_last=channels_last)

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