import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import LLM_CONFIGS, FLOAT_DTYPES, PROMOTE_DTYPES
 
def nanogpt_attn_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    head_size: int,
    dropout_p: float      
):
    T0 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=[1, 1, -1, -1], contiguity=[None, None, False, True], dtype=DataType.Float, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)

    S2 = fd.define_scalar( 1 / head_size**0.5, dtype=DataType.Double)
    T3 = fd.ops.mul(T0, S2)
    S4 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T5 = fd.ops.eq(T1, S4)
    V10 = T0.shape()
    T11 = fd.ops.broadcast_in_dim(T5, shape=V10, broadcast_dims=[0, 1, 2, 3])
    S12 = fd.define_scalar(float("-inf"), dtype=DataType.Double)
    T13 = fd.ops.where(T11, S12, T3)
    T14 = fd.ops.max(T13, axes=[3], keepdim=False, dtype=DataType.Null)
    V19 = fd.define_vector([T0.size(0), T0.size(1), T0.size(2), 1], dtype=DataType.Int)
    T20 = fd.ops.broadcast_in_dim(T14, shape=V19, broadcast_dims=[0, 1, 2])
    T26 = fd.ops.broadcast_in_dim(T20, shape=V10, broadcast_dims=[0, 1, 2, 3])
    T27 = fd.ops.sub(T13, T26)
    T28 = fd.ops.exp(T27)
    T29 = fd.ops.sum(T28, axes=[3], keepdim=False, dtype=DataType.Null)
    T35 = fd.ops.broadcast_in_dim(T29, shape=V19, broadcast_dims=[0, 1, 2])
    T41 = fd.ops.broadcast_in_dim(T35, shape=V10, broadcast_dims=[0, 1, 2, 3])
    T42 = fd.ops.reciprocal(T41)
    T43 = fd.ops.mul(T28, T42)
    S44 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S45 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T51 = fd.ops.uniform(S44, S45, shape=V10, dtype=DataType.Float)
    S52 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    T53 = fd.ops.lt(T51, S52)
    T55 = fd.ops.mul(T43, T53)
    S56 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T57 = fd.ops.mul(T55, S56)

    if dtype in PROMOTE_DTYPES:
        T57 = fd.ops.cast(T57, dtype=dtype)
        T43 = fd.ops.cast(T43, dtype=dtype)

    fd.add_output(T57)
    fd.add_output(T53)
    fd.add_output(T43)
    fd.add_output(T11)

@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
@pytest.mark.parametrize("llm_config", LLM_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanogpt_attn_fwd_benchmark(
    benchmark,
    batch_size: int,
    seq_len: int,
    llm_config: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    (nh, n_embd) = llm_config
    hs = n_embd // nh
    dropout_p = 0.0
    inputs = torch.randn(batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype)
    bias = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).view(1, 1, seq_len, seq_len)
    dropout_mask = torch.lt(torch.rand(batch_size, nh, seq_len, seq_len, device="cuda"),  1 - dropout_p)
    bias_mask = bias[:, :, :seq_len, :seq_len] == 0
    bias_mask = bias_mask.broadcast_to(batch_size, nh, seq_len, seq_len)
    
    with FusionDefinition() as fd:
        nanogpt_attn_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), hs, dropout_p)

    if not disable_validation:
        attn = inputs / (hs ** 0.5)
        attn = attn.masked_fill(bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.nn.functional.dropout(attn, p = dropout_p) 
        fd.validate([inputs, bias], [out, dropout_mask, attn, bias_mask])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, bias])