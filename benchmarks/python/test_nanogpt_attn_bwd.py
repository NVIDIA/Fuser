import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_attn_inputs, FLOAT_DTYPES, PROMOTE_DTYPES


# Fusion from nanogpt attention module
# The nvFuser defintion only includes the non-matmul computation (masked_fill + softmax + dropout)
def nanogpt_attn_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, head_size: int, dropout_p: float
):
    S0 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    S1 = fd.define_scalar(1 / head_size**0.5, dtype=DataType.Double)

    T2 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=dtype,
        is_cpu=False,
    )
    T3 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=dtype,
        is_cpu=False,
    )
    T4 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )
    T5 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[None, None, True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T2 = fd.ops.cast(T2, dtype=DataType.Float)
        T3 = fd.ops.cast(T3, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)

    T7 = fd.ops.mul(T2, S0)
    T8 = fd.ops.mul(T7, T4)
    T9 = fd.ops.mul(T3, T8)
    T10 = fd.ops.sum(T9, dims=[3], keepdim=False, dtype=DataType.Null)

    V15 = fd.define_vector([T2.size(0), T2.size(1), T2.size(2), 1], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T10, shape=V15, broadcast_dims=[0, 1, 2])

    V21 = T2.shape()
    T22 = fd.ops.broadcast_in_dim(T16, shape=V21, broadcast_dims=[0, 1, 2, 3])

    T23 = fd.ops.sub(T8, T22)
    T24 = fd.ops.mul(T3, T23)
    S25 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T26 = fd.ops.where(T5, S25, T24)
    T27 = fd.ops.mul(T26, S1)

    if dtype in PROMOTE_DTYPES:
        T27 = fd.ops.cast(T27, dtype=dtype)

    fd.add_output(T27)


@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanogpt_attn_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    batch_size, seq_len, nh, n_embd = size
    hs = n_embd // nh
    dropout_p = 0.0
    inputs = torch.randn(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype, requires_grad=True
    )
    bias = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).view(
        1, 1, seq_len, seq_len
    )
    dropout_mask = torch.lt(
        torch.rand(batch_size, nh, seq_len, seq_len, device="cuda"), 1 - dropout_p
    )
    bias_mask = bias[:, :, :seq_len, :seq_len] == 0
    bias_mask = bias_mask.broadcast_to(batch_size, nh, seq_len, seq_len)
    grads = torch.randn(batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype)

    attn = inputs / (hs**0.5)
    attn = attn.masked_fill(bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
    attn = torch.nn.functional.softmax(attn, dim=-1)

    with FusionDefinition() as fd:
        nanogpt_attn_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), hs, dropout_p)

    if not disable_validation:
        out = torch.nn.functional.dropout(attn, p=dropout_p)
        out.backward(grads)
        fd.validate([grads, attn, dropout_mask, bias_mask], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [grads, attn, dropout_mask, bias_mask])
