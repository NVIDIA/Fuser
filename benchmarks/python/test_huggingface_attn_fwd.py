# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_attn_inputs, FLOAT_DTYPES, PROMOTE_DTYPES
from .torch_ops import huggingface_attn


# Fusion from huggingface attention implementation.
# The nvFuser defintion only includes the non-matmul computation (add + reshape + softmax + dropout)
def huggingface_attn_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    dropout_p: float,
):
    T0 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=dtype,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1, -1],
        contiguity=[True, True, True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T4 = fd.ops.add(T1, T0)

    V9 = fd.define_vector(
        [T0.size(0) * T0.size(1), T0.size(2), T0.size(3)], dtype=DataType.Int
    )
    T10 = fd.ops.reshape(T4, new_shape=V9)
    T12 = fd.ops.max(T10, dims=[2], keepdim=False, dtype=DataType.Null)

    V16 = fd.define_vector([T0.size(0) * T0.size(1), T0.size(2), 1], dtype=DataType.Int)
    T17 = fd.ops.broadcast_in_dim(T12, shape=V16, broadcast_dims=[0, 1])
    T22 = fd.ops.broadcast_in_dim(T17, shape=V9, broadcast_dims=[0, 1, 2])
    T23 = fd.ops.sub(T10, T22)
    T24 = fd.ops.exp(T23)
    T25 = fd.ops.sum(T24, dims=[2], keepdim=False, dtype=DataType.Null)

    T30 = fd.ops.broadcast_in_dim(T25, shape=V16, broadcast_dims=[0, 1])
    T35 = fd.ops.broadcast_in_dim(T30, shape=V9, broadcast_dims=[0, 1, 2])

    T36 = fd.ops.reciprocal(T35)
    T37 = fd.ops.mul(T24, T36)

    S39 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S40 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T45 = fd.ops.uniform(S39, S40, shape=V9, dtype=DataType.Float)
    S46 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    T47 = fd.ops.lt(T45, S46)

    T48 = fd.ops.cast(T47, dtype=DataType.Float)
    T49 = fd.ops.mul(T37, T48)
    S50 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T51 = fd.ops.mul(T49, S50)

    if dtype in PROMOTE_DTYPES:
        T37 = fd.ops.cast(T37, dtype=dtype)
        T51 = fd.ops.cast(T51, dtype=dtype)

    fd.add_output(T51)
    fd.add_output(T37)
    fd.add_output(T47)


def huggingface_attn_fwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser outputs (output, attn, dropout_mask) differ from baseline outputs (output).

    # Total IO bytes = attention_mask ([bs, nh, seq_len, seq_len], dtype) + in_tensor ([bs, nh, seq_len, seq_len], dtype) +
    #   output ([bs, nh, seq_len, seq_len], dtype) + attn ([bs * nh, seq_len, seq_len], dtype) + dropout_mask ([bs * nh, seq_len, seq_len], bool)

    bs, seq_len, nh, n_embd = size
    return int(bs * nh * seq_len * seq_len * (dtype.itemsize * 4 + torch.bool.itemsize))


@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_huggingface_attn_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    batch_size, seq_len, nh, n_embd = size
    dropout_p = 0.2
    inputs = torch.randn(batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype)

    attention_mask = torch.zeros(
        batch_size, nh, seq_len, seq_len, device="cuda", dtype=dtype
    )

    if not disable_validation:
        # For validating use a fusion definition with dropout_p=0.0
        with FusionDefinition() as val_fd:
            huggingface_attn_fwd_fusion(
                val_fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p=0.0
            )

        dropout_mask = torch.ones(
            batch_size * nh, seq_len, seq_len, dtype=torch.bool, device="cuda"
        )
        attn = (inputs + attention_mask).view(batch_size * nh, seq_len, seq_len)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.nn.functional.dropout(attn, p=0.0)
        val_fd.validate([attention_mask, inputs], [out, attn, dropout_mask])

    if not disable_benchmarking:
        with FusionDefinition() as fd:
            huggingface_attn_fwd_fusion(
                fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p
            )
        run_benchmark(benchmark, fd.execute, [attention_mask, inputs])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_attn_inputs())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_huggingface_attn_fwd_baseline_benchmark(
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

    benchmark_fn = with_executor(executor, huggingface_attn)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, attention_mask, size, dropout_p],
        iobytes=huggingface_attn_fwd_iobytes(size, dtype),
    )
