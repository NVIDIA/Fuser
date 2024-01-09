import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_attn_inputs, FLOAT_DTYPES, PROMOTE_DTYPES


# Fusion from huggingface attention implementation
# https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/tests/hf_bart_self_attn.py#L73-L83
def huggingface_attn_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    batch_size: int,
    num_heads: int,
    dropout_p: float,
):
    S0 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T5 = fd.define_tensor(
        shape=[-1, -1, -1], contiguity=[True, True, True], dtype=dtype, is_cpu=False
    )
    T6 = fd.define_tensor(
        shape=[-1, -1, -1], contiguity=[True, True, True], dtype=dtype, is_cpu=False
    )
    T7 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T5 = fd.ops.cast(T5, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        T7 = fd.ops.cast(T7, dtype=DataType.Float)

    T10 = fd.ops.mul(T5, S0)
    T11 = fd.ops.mul(T10, T7)
    T13 = fd.ops.mul(T6, T11)
    T14 = fd.ops.sum(T13, axes=[2], keepdim=False, dtype=DataType.Null)

    V18 = fd.define_vector([T5.size(0), T5.size(1), 1], dtype=DataType.Int)
    T19 = fd.ops.broadcast_in_dim(T14, shape=V18, broadcast_dims=[0, 1])
    V24 = T5.shape()
    T25 = fd.ops.broadcast_in_dim(T19, shape=V24, broadcast_dims=[0, 1, 2])
    T27 = fd.ops.sub(T11, T25)
    T28 = fd.ops.mul(T6, T27)
    V34 = fd.define_vector(
        [batch_size, num_heads, T5.size(1), T5.size(2)], dtype=DataType.Int
    )
    T35 = fd.ops.reshape(T28, new_shape=V34)

    if dtype in PROMOTE_DTYPES:
        T35 = fd.ops.cast(T35, dtype=dtype)
    fd.add_output(T35)


@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_huggingface_attn_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    batch_size, seq_len, nh, n_embd = size

    dropout_p = 0.0
    inputs = torch.randn(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype, requires_grad=True
    )
    dropout_mask = torch.lt(
        torch.rand(batch_size * nh, seq_len, seq_len, device="cuda"), 1 - dropout_p
    )
    attention_mask = torch.zeros(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype
    )
    grads = torch.randn(batch_size * nh, seq_len, seq_len, device="cuda", dtype=dtype)

    attn = (inputs + attention_mask).view(batch_size * nh, seq_len, seq_len)
    attn = torch.nn.functional.softmax(attn, dim=-1)

    with FusionDefinition() as fd:
        huggingface_attn_bwd_fusion(
            fd, torch_dtype_to_nvfuser_dtype(dtype), batch_size, nh, dropout_p
        )

    if not disable_validation:
        out = torch.nn.functional.dropout(attn, p=dropout_p)
        out.backward(grads)
        fd.validate([grads, attn, dropout_mask], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [grads, attn, dropout_mask])
