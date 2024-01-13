from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from core import clear_cuda_cache
import torch

def pointwise_ops_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    num_iters: int
):
    x = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    y = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    x = fd.ops.cast(x, dtype=DataType.Float)
    y = fd.ops.cast(y, dtype=DataType.Float)

    a = fd.ops.add(x, y)
    for _ in range(num_iters):
        x = fd.ops.cos(a)
        y = fd.ops.sin(a)
        a = fd.ops.add(x, y)

    a = fd.ops.cast(a, dtype=dtype)
    fd.add_output(a)

def test_pointwise_ops_benchmark(
    num_iters: int
):  
    clear_cuda_cache()
    inputs = [torch.randn(13, device="cuda", dtype=torch.float16) for _ in range(2)]
    with FusionDefinition() as fd:
        pointwise_ops_fusion(fd, torch_dtype_to_nvfuser_dtype(torch.float16), num_iters)

    print(f'num_iters: {num_iters}')
    fd.execute(inputs)

for num_iters in [2, 4, 8, 16]:
    test_pointwise_ops_benchmark(num_iters)