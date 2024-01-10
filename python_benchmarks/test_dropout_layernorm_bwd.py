import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def dropout_layernorm_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, dropout_p: float
) -> None:
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Bool, is_cpu=False) # mask
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False) # mean
    T3 = fd.define_tensor(shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False) # invstd
    T4 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False) # grads
    T5 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False) # weights
    T6 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False) # inputs
    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
        T5 = fd.ops.cast(T5, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        
    T9 = fd.ops.mul(T6, T1)
    S10 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T11 = fd.ops.mul(T9, S10)
    T12 = fd.ops.add(T6, T11)
    
    V15 = fd.define_vector([T6.size(0), 1], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T2, shape=V15, broadcast_dims=[0])
    V19 = T6.shape()
    T20 = fd.ops.broadcast_in_dim(T16, shape=V19, broadcast_dims=[0, 1])
    T21 = fd.ops.sub(T12, T20)
    T25 = fd.ops.broadcast_in_dim(T3, shape=V19, broadcast_dims=[0, 1])
    T26 = fd.ops.mul(T21, T25)
    T30 = fd.ops.broadcast_in_dim(T5, shape=V19, broadcast_dims=[1])
    T35 = fd.ops.sum(T4, axes=[0], keepdim=False, dtype=DataType.Null)
    
    T37 = fd.ops.mul(T4, T30)
    T38 = fd.ops.mul(T4, T26)
    T39 = fd.ops.sum(T38, axes=[0], keepdim=False, dtype=DataType.Null)
    
    T41 = fd.ops.mul(T37, T25)
    T42 = fd.ops.mul(T37, T21)
    T43 = fd.ops.sum(T42, axes=[1], keepdim=False, dtype=DataType.Null)
    T47 = fd.ops.broadcast_in_dim(T43, shape=V15, broadcast_dims=[0])
    T48 = fd.ops.neg(T41)
    T49 = fd.ops.sum(T48, axes=[1], keepdim=False, dtype=DataType.Null)
    T53 = fd.ops.broadcast_in_dim(T49, shape=V15, broadcast_dims=[0])
    S54 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T55 = fd.ops.mul(S54, T47)
    S56 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T57 = fd.ops.pow(T3, S56)
    T58 = fd.ops.mul(T55, T57)
    T61 = fd.ops.sum(T53, axes=[1], keepdim=False, dtype=DataType.Null)
    T62 = fd.ops.sum(T58, axes=[1], keepdim=False, dtype=DataType.Null)
    T66 = fd.ops.broadcast_in_dim(T62, shape=V15, broadcast_dims=[0])
    T70 = fd.ops.broadcast_in_dim(T66, shape=V19, broadcast_dims=[0, 1])
    T74 = fd.ops.broadcast_in_dim(T2, shape=V15, broadcast_dims=[0])
    T78 = fd.ops.broadcast_in_dim(T74, shape=V19, broadcast_dims=[0, 1])
    S79 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T80 = fd.ops.mul(S79, T70)
    T81 = fd.ops.sub(T12, T78)
    T82 = fd.ops.mul(T80, T81)
    S84 = fd.ops.reciprocal(T6.size(1))
    T85 = fd.ops.mul(T82, S84)
    T89 = fd.ops.broadcast_in_dim(T61, shape=V15, broadcast_dims=[0])
    T93 = fd.ops.broadcast_in_dim(T89, shape=V19, broadcast_dims=[0, 1])
    T95 = fd.ops.mul(S84, T93)
    T96 = fd.ops.add(T85, T95)
    T97 = fd.ops.add(T41, T96)

    T100 = fd.ops.mul(T97, S10)
    T101 = fd.ops.mul(T100, T1)
    T102 = fd.ops.add(T97, T101)
    if dtype in PROMOTE_DTYPES:
        T35 = fd.ops.cast(T35, dtype=dtype)
        T39 = fd.ops.cast(T39, dtype=dtype)
        T102 = fd.ops.cast(T102, dtype=dtype)
    fd.add_output(T102)
    fd.add_output(T39)
    fd.add_output(T35)
    
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layernorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    clear_cuda_cache()
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    dropout_p = 0.1
    dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)
    x = inputs + 1 / (1 - dropout_p) * dropout_mask * inputs
    mean = x.to(torch.float).mean(dim=-1)
    variance = x.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)
    
    with FusionDefinition() as fd:
        dropout_layernorm_bwd_fusion(
            fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p=dropout_p
        )
    if not disable_validation:
        eager_output = torch.nn.functional.layer_norm(
            x.to(torch.double),
            inputs.shape[1:],
            weight=weights.to(torch.double),
            bias=bias.to(torch.double),
        )
        
        eager_output.backward(grads.to(torch.double))
        fd.validate(
            [dropout_mask, mean, invstd, grads, weights, inputs],
            [inputs.grad, weights.grad, bias.grad]
        )
    if not disable_benchmarking:
        run_benchmark(
            benchmark, fd.execute, [dropout_mask, mean, invstd, grads, weights, inputs]
        )