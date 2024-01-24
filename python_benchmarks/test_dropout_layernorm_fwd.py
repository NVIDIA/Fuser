import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def dropout_layernorm_fwd_fusion(
    fd: FusionDefinition, dtype: DataType, dropout_p: float, eps: float = 1e-5
) -> None:
    """
    Forward pass fusion definition for computing:
        output = layernorm (input + dropout (input, p=dropout_p))

    Fusion inputs: input, weights, bias
    Fusion outputs: output, mean, invstd, dropout_mask
    """
    T2 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    S3 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S4 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T8 = fd.ops.uniform(S3, S4, shape=T2.shape(), dtype=DataType.Float)
    S9 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    T10 = fd.ops.lt(T8, S9)
    T11 = fd.ops.cast(T10, dtype=DataType.Float)
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    # Dropout + Add
    T13 = fd.ops.mul(T2, T11)
    S14 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T15 = fd.ops.mul(T13, S14)
    T16 = fd.ops.add(T2, T15)
    # Layernorm
    T17, T18 = fd.ops.var_mean(T16, axes=[1], correction=0, keepdim=False)
    V21 = fd.define_vector([T2.size(0), 1], dtype=DataType.Int)
    T22 = fd.ops.broadcast_in_dim(T17, shape=V21, broadcast_dims=[0])
    T26 = fd.ops.broadcast_in_dim(T18, shape=V21, broadcast_dims=[0])
    S27 = fd.define_scalar(eps, dtype=DataType.Double)
    T28 = fd.ops.add(T22, S27)
    T29 = fd.ops.rsqrt(T28)
    T33 = fd.ops.broadcast_in_dim(T26, shape=T2.shape(), broadcast_dims=[0, 1])
    T34 = fd.ops.sub(T16, T33)
    T38 = fd.ops.broadcast_in_dim(T29, shape=T2.shape(), broadcast_dims=[0, 1])
    T39 = fd.ops.mul(T34, T38)
    T43 = fd.ops.broadcast_in_dim(T1, shape=T2.shape(), broadcast_dims=[1])
    T45 = fd.ops.mul(T39, T43)
    T49 = fd.ops.broadcast_in_dim(T0, shape=T2.shape(), broadcast_dims=[1])
    T51 = fd.ops.add(T45, T49)
    if dtype in PROMOTE_DTYPES:
        T51 = fd.ops.cast(T51, dtype=dtype)

    fd.add_output(T51)
    fd.add_output(T18)
    fd.add_output(T29)
    fd.add_output(T10)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dropout_layernorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    clear_cuda_cache()
    inputs = [
        torch.randn(*size, device="cuda", dtype=dtype),
        torch.ones(size[1], device="cuda", dtype=dtype),
        torch.zeros(size[1], device="cuda", dtype=dtype),
    ]
    # dropout_p = 0.0 in fwd benchmark for validating the dropout mask
    dropout_p = 0.0
    dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)
    with FusionDefinition() as fd:
        dropout_layernorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p)
    if not disable_validation:
        # dropout + add
        x = inputs[0] + 1 / (1 - dropout_p) * dropout_mask * inputs[0]
        # layernorm
        eager_output = torch.nn.functional.layer_norm(
            x.to(torch.float),
            inputs[0].shape[1:],
            weight=inputs[1].to(torch.float),
            bias=inputs[2].to(torch.float),
        )
        # mean and invstd are computed for the output of dropout + add
        mean = x.to(torch.float).mean(dim=-1)
        variance = x.to(torch.float).var(dim=-1, unbiased=False)
        invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)
        fd.validate(inputs, [eager_output.to(dtype), mean, invstd, dropout_mask])
    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
