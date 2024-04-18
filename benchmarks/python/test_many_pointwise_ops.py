import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import PROMOTE_DTYPES

def pointwise_ops_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    num_iters: int
):
    x = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    y = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        x = fd.ops.cast(x, dtype=DataType.Float)
        y = fd.ops.cast(y, dtype=DataType.Float)

    a = fd.ops.add(x, y)
    for _ in range(num_iters):
        x = fd.ops.cos(a)
        y = fd.ops.sin(a)
        a = fd.ops.add(x, y)

    if dtype in PROMOTE_DTYPES:
        a = fd.ops.cast(a, dtype=dtype)
    fd.add_output(a)

#NOTE: num_iters restricted due to issue #1234.
@pytest.mark.parametrize("num_iters", [2, 4, 8, 16])
def test_pointwise_ops_benchmark(
    benchmark,
    num_iters: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):  
    clear_cuda_cache()
    inputs = [torch.randn(13, device="cuda", dtype=torch.float16) for _ in range(2)]
    with FusionDefinition() as fd:
        pointwise_ops_fusion(fd, torch_dtype_to_nvfuser_dtype(torch.float16), num_iters)

    if not disable_validation:
        eager_output = inputs[0] + inputs[1]
        for _ in range(num_iters):
            x = torch.cos(eager_output)
            y = torch.sin(eager_output)
            eager_output = x + y

        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs, device="cpu")