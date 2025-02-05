# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import (
    run_benchmark,
    clear_dynamo_cache,
    unary_bwd_torch,
    with_executor,
    DEFAULT_EXECUTORS,
)
import torch
from .global_params import generate_attn_inputs, FLOAT_DTYPES, PROMOTE_DTYPES
from .torch_ops import huggingface_attn


# Fusion from huggingface attention implementation
# The nvFuser defintion only includes the non-matmul computation (add + reshape + softmax + dropout)
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
    T14 = fd.ops.sum(T13, dims=[2], keepdim=False, dtype=DataType.Null)

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


def huggingface_attn_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser input/outputs (grad_out, attn, dropout_mask, grad_input]) differ from baseline input/outputs (output, grad_output).

    # Total IO bytes = grad_output ([bs, nh, seq_len, seq_len], dtype) + attn ([bs*nh, seq_len, seq_len], dtype) + dropout_mask ([bs*nh, seq_len, seq_len], bool) + grad_input ([bs, nh, seq_len, seq_len], dtype)
    bs, seq_len, nh, n_embd = size
    return int(bs * nh * seq_len * seq_len * (dtype.itemsize * 3 + torch.bool.itemsize))


@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_huggingface_attn_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    batch_size, seq_len, nh, n_embd = size

    dropout_p = 0.2
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
        # Use dropout_mask instead of torch.nn.functional.dropout for validating results.
        out = attn * dropout_mask * 1 / (1 - dropout_p)
        out.backward(grads)
        fd.validate([grads, attn, dropout_mask], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [grads, attn, dropout_mask])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_huggingface_attn_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    batch_size, seq_len, nh, n_embd = size
    dropout_p = 0.2
    inputs = torch.randn(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype, requires_grad=True
    )
    attention_mask = torch.zeros(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype
    )

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, huggingface_attn)
    fwd_inputs = [inputs, attention_mask, size, dropout_p]
    outputs = fwd_fn(fwd_inputs)
    grads = torch.randn(batch_size * nh, seq_len, seq_len, device="cuda", dtype=dtype)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=huggingface_attn_bwd_iobytes(size, dtype),
    )
