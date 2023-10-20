import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
from .normalization import norm_fwd_fusion, norm_bwd_fusion

@pytest.mark.parametrize("size",  generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_batchnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    # Size is assumed to be in the order N, C, ...
    num_dims = len(size)
    
    at_inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)
    
    if channels_last:
        at_inputs = at_inputs.to(memory_format=torch.channels_last)
        inputs = at_inputs.clone().detach().permute((0, *range(2, num_dims), 1))
    else:
        inputs = at_inputs

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
            at_inputs, running_mean.to(dtype), running_var.to(dtype), weight=weight, bias=bias, training=True,
        )

        if channels_last:
            nvf_output[0] = nvf_output[0].permute((0, -1, *range(1, num_dims-1)))

        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weight, bias, running_mean, running_var])

@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [True, False])
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

    num_dims = len(size)

    at_inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    at_grads = torch.randn(*size, device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    if channels_last:
        at_inputs = at_inputs.to(memory_format=torch.channels_last)
        at_inputs.retain_grad()
        at_grads = at_grads.to(memory_format=torch.channels_last)
        
        inputs = at_inputs.clone().detach().permute((0, *range(2, num_dims), 1))
        grads = at_grads.clone().detach().permute((0, *range(2, num_dims), 1))

    else:
        inputs = at_inputs
        grads = at_grads

    channel_dim = 1 if not channels_last else num_dims - 1
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
            at_inputs, running_mean.to(dtype), running_var.to(dtype), weight=weight, bias=bias, training=True,
        )
        eager_output.backward(at_grads)
        if channels_last:
            nvf_output[0] = nvf_output[0].permute((0, -1, *range(1, num_dims-1)))

        assert torch.allclose(nvf_output[0], at_inputs.grad, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - at_inputs.grad)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, weight, running_mean, running_var, mean, invstd])