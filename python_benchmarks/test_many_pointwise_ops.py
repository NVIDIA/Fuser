import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def pointwise_ops_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    num_iters: int
):
    x = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False)
    if dtype in PROMOTE_DTYPES:
        x = fd.ops.cast(x, dtype=DataType.Float)
    for _ in range(num_iters):
        x = fd.ops.add(x, x)
    if dtype in PROMOTE_DTYPES:
        x = fd.ops.cast(x, dtype=dtype)
    fd.add_output(x)

@pytest.mark.parametrize("size", [(2, 4)])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_iters", [32])
def test_pointwise_ops_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    num_iters: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype)
    with FusionDefinition() as fd:
        pointwise_ops_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), num_iters)
    
    if not disable_validation:
        nvf_output = fd.execute([inputs])
        eager_output = inputs

        for _ in range(num_iters):
            eager_output += eager_output

        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs])

    torch.cuda.empty_cache()

    