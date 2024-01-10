import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def dropout_rmsnorm_fwd_fusion(
        fd : FusionDefinition,
        dtype: DataType,
        dropout_p: float,
        eps: float = 1e-5,
    ) -> None :
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    S2 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S3 = fd.define_scalar(1.00000, dtype=DataType.Double)

    V6 = T0.shape()
    T7 = fd.ops.uniform(S2, S3, shape=V6, dtype=dtype)
    S8 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    T9 = fd.ops.lt(T7, S8)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T9 = fd.ops.cast(T9, dtype=DataType.Float)

    T12 = fd.ops.mul(T0, T9)
    S13 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T14 = fd.ops.mul(T12, S13)
    T15 = fd.ops.add(T0, T14)
    S16 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T17 = fd.ops.pow(T15, S16)
    T18 = fd.ops.sum(T17, axes=[1], keepdim=False, dtype=DataType.Null)

    V21 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T22 = fd.ops.broadcast_in_dim(T18, shape=V21, broadcast_dims=[0])

    S24 = fd.ops.reciprocal(T0.size(1))
    T25 = fd.ops.mul(T22, S24)
    S26 = fd.define_scalar(eps, dtype=DataType.Double)
    T27 = fd.ops.add(T25, S26)
    T28 = fd.ops.sqrt(T27)
    
    T33 = fd.ops.broadcast_in_dim(T28, shape=V6, broadcast_dims=[0, 1])

    T35 = fd.ops.reciprocal(T33)
    T36 = fd.ops.mul(T15, T35)
    T40 = fd.ops.broadcast_in_dim(T1, shape=V6, broadcast_dims=[1])
    T42 = fd.ops.mul(T40, T36)

    if dtype in PROMOTE_DTYPES:
        T42 = fd.ops.cast(T42, dtype=dtype)
    
    fd.add_output(T42)
    fd.add_output(T9)
    fd.add_output(T28)

@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    clear_cuda_cache()

    inputs = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype)

    # dropout_p = 0.0 in fwd benchmark for validating the dropout mask
    dropout_p = 0.0
    dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)

    with FusionDefinition() as fd:
        dropout_rmsnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p, eps)

    if not disable_validation:
        x = inputs + 1 / (1 - dropout_p) * dropout_mask * inputs
        squared_mean = (x.to(torch.float) ** 2).mean(1, keepdim=True)
        rms_eps = torch.sqrt(squared_mean + eps)
        eager_output = weights * (x / rms_eps)
        fd.validate([inputs, weights], [eager_output.to(dtype), dropout_mask, rms_eps])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weights])
