import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def transpose_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    config: dict,
    axes: list,
):
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
    
    if config['transpose_input1']:
        T0 = fd.ops.permute(T0, dims=axes)
    
    if config['transpose_input2']:
        T1 = fd.ops.permute(T1, dims=axes)
        
    T4 = fd.ops.add(T0, T1)

    if config['transpose_intermediate']:
        T4 = fd.ops.permute(T4, dims=axes)
    
    if dtype in PROMOTE_DTYPES:
        T4 = fd.ops.cast(T4, dtype=dtype)

    S6 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T7 = fd.ops.gt(T4, S6)
    T9 = fd.ops.where(T7, T4, S6)

    if config['transpose_output']:
        T7 = fd.ops.permute(T7, dims=axes)
        T9 = fd.ops.permute(T9, dims=axes)

    fd.add_output(T7)
    fd.add_output(T9)

@pytest.mark.parametrize("size", [(128, 64)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# @pytest.mark.parametrize("axes", [()])
def test_transpose_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    input1 = torch.randn(*size, device="cuda", dtype=dtype)
    input2 = torch.randn(*size, device="cuda", dtype=dtype)
    config = {'transpose_input1': False,
              'transpose_input2': False,
              'transpose_intermediate': False,
              'transpose_output': True}
    
    with FusionDefinition() as fd:
        transpose_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), config, (0, 1))
    
    if not disable_validation:
        nvf_output = fd.execute([input1, input2])
        eager_output = torch.nn.functional.relu(input1 + input2)
        print(eager_output.shape)
        print(nvf_output[1].shape)

        assert torch.allclose(
            nvf_output[1], eager_output, rtol=1e-3, atol=1e-3
        ), f"{torch.max(nvf_output[1] - eager_output)}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [input1, input2])

