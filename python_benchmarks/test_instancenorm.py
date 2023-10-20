import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
from .normalization import norm_fwd_fusion, norm_bwd_fusion

def instancenorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
    momentum: float = 1e-5,
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
    
    reduction_axes = [2, 3]
    var, mean = fd.ops.var_mean(input, axes=reduction_axes, correction=0, keepdim=False)
    bcast_shape = fd.define_vector([input.size(0), input.size(1), 1, 1], dtype=DataType.Int)
    var_bcast = fd.ops.broadcast_in_dim(var, shape=bcast_shape, broadcast_dims=[0, 1])
    mean_bcast = fd.ops.broadcast_in_dim(mean, shape=bcast_shape, broadcast_dims=[0, 1])

    eps = fd.define_scalar(eps, dtype=DataType.Double)
    x_sub_mean = fd.ops.sub(input, mean_bcast)
    var_eps = fd.ops.add(var_bcast, eps)
    invstd = fd.ops.rsqrt(var_eps)
    x_norm = fd.ops.mul(x_sub_mean, invstd)

    weight = fd.ops.broadcast_in_dim(weight, shape=bcast_shape, broadcast_dims=[0, 1])
    x_scaled = fd.ops.mul(x_norm, weight)
    bias = fd.ops.broadcast_in_dim(bias, shape=bcast_shape, broadcast_dims=[0, 1])
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

@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [False])
def test_instancenorm_fwd_benchmark(
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
            norm = "instance_norm", 
            num_dims = num_dims, 
            channels_last = channels_last)

    if not disable_validation:
        nvf_output = fd.execute([inputs, weight, bias, running_mean, running_var])
        # PyTorch expects running mean and variance to be of same type as input.
        eager_output = torch.nn.functional.instance_norm(
            inputs, running_mean.to(dtype), running_var.to(dtype), weight=weight, bias=bias,
        )
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3),\
                              f"{torch.max(nvf_output[0] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weight, bias, running_mean, running_var])
