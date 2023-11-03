import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def rmsnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps:float = 1e-5,
):  
    inputs = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype)
    weights = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype)

    if dtype in PROMOTE_DTYPES:
        inputs = fd.ops.cast(inputs, dtype=DataType.Float)
        weights = fd.ops.cast(weights, dtype=DataType.Float)
        
    inputs_sq = fd.ops.mul(inputs, inputs)
    squared_sum = fd.ops.sum(inputs_sq, axes=[1], keepdim=True)
    inverse_norm_size = fd.ops.reciprocal(inputs.size(1))
    var = fd.ops.mul(squared_sum, inverse_norm_size)
    eps_const = fd.define_scalar(eps)
    var_eps = fd.ops.add(var, eps_const)
    invstd = fd.ops.rsqrt(var_eps)
    pre_scale = fd.ops.mul(inputs, invstd)
    weights_bcast = fd.ops.broadcast_in_dim(weights, shape=inputs.shape(), broadcast_dims=[1])
    out = fd.ops.mul(pre_scale, weights_bcast)
    if dtype in PROMOTE_DTYPES:
        out = fd.ops.cast(out, dtype=dtype)
    fd.add_output(out)
    fd.add_output(invstd)



@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps:float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        rmsnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    
    if not disable_validation:
        nvf_output = fd.execute([inputs, weights])
        ms = (inputs**2).mean(-1, keepdim=True)
        eager_output = weights * (inputs / torch.sqrt(ms + eps))
        
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3), \
            f"Max error: {torch.max(torch.abs(nvf_output[0] - eager_output))}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weights])

    torch.cuda.empty_cache()