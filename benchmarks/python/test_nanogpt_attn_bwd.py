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
from .torch_ops import nanogpt_attn


# Fusion from nanogpt attention module
# The nvFuser defintion only includes the non-matmul computation (masked_fill + softmax + dropout)
def nanogpt_attn_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, head_size: int, dropout_p: float
):
    S0 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
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


def nanogpt_attn_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser input/outputs (grad_out, attn, dropout_mask, bias_mask, grad_input]) differ from baseline input/outputs (output, grad_output).

    # Total IO bytes = grad_out ([bs, nh, seq_len, seq_len], dtype) + attn ([bs, nh, seq_len, seq_len], dtype) +
    #       dropout_mask ([bs, nh, seq_len, seq_len], bool) + bias_mask ([bs, nh, seq_len, seq_len], bool) + grad_in ([bs, nh, seq_len, seq_len], dtype)
    bs, seq_len, nh, n_embd = size

    return int(
        bs * nh * seq_len * seq_len * (3 * dtype.itemsize + 2 * torch.bool.itemsize)
    )


@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
def test_nanogpt_attn_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    batch_size, seq_len, nh, n_embd = size
    hs = n_embd // nh
    dropout_p = 0.2
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
        # Use dropout_mask instead of torch.nn.functional.dropout for validating results.
        out = attn * dropout_mask * 1 / (1 - dropout_p)
        out.backward(grads)
        fd.validate([grads, attn, dropout_mask, bias_mask], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [grads, attn, dropout_mask, bias_mask])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
def test_nanogpt_attn_bwd_baseline_benchmark(
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
    bias = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).view(
        1, 1, seq_len, seq_len
    )

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, nanogpt_attn)
    fwd_inputs = [inputs, bias, size, dropout_p]
    outputs = fwd_fn(fwd_inputs)

    grads = torch.randn(batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=nanogpt_attn_bwd_iobytes(size, dtype),
    )
