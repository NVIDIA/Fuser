# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import (
    run_benchmark,
    clear_dynamo_cache,
    compute_total_iobytes,
    with_executor,
    DEFAULT_EXECUTORS,
)
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
from .torch_ops import dropout_rmsnorm


def dropout_rmsnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    dropout_p: float,
    eps: float = 1e-5,
) -> None:
    """
    Forward pass fusion definition for computing:
        output = rmsnorm (input2 + dropout (input1, p=dropout_p))

    Fusion inputs: input, weights
    Fusion outputs: output, dropout_mask, rms
    """
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T2 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    S2 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S3 = fd.define_scalar(1.00000, dtype=DataType.Double)

    V6 = T0.shape()
    T7 = fd.ops.uniform(S2, S3, shape=V6, dtype=DataType.Float)
    S8 = fd.define_scalar(1 - dropout_p, dtype=DataType.Double)
    T9 = fd.ops.lt(T7, S8)
    T10 = fd.ops.cast(T9, dtype=DataType.Float)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    T12 = fd.ops.mul(T0, T10)
    S13 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T14 = fd.ops.mul(T12, S13)
    T15 = fd.ops.add(T2, T14)
    S16 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T17 = fd.ops.pow(T15, S16)
    T18 = fd.ops.sum(T17, dims=[1], keepdim=False, dtype=DataType.Null)

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


def dropout_rmsnorm_fwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser input/outputs differ from baseline outputs (output).
    nvf_inp_out = {
        # Inputs
        "input1": (size, dtype),
        "input2": (size, dtype),
        "weights": (size[1], dtype),
        # Outputs
        "rms": (size[0], torch.float),
        "output": (size, dtype),
        "dropout_mask": (size, torch.bool),
    }
    return compute_total_iobytes(nvf_inp_out)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_dropout_rmsnorm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    input2 = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype)

    dropout_p = 0.2

    with FusionDefinition() as fd:
        dropout_rmsnorm_fwd_fusion(
            fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p, eps
        )

    if not disable_validation:
        # For validating use a fusion definition with dropout_p=0.0
        with FusionDefinition() as val_fd:
            dropout_rmsnorm_fwd_fusion(
                val_fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p=0.0, eps=eps
            )

        dropout_mask = torch.ones(size, dtype=torch.bool, device="cuda")

        x = input2.to(torch.double) + torch.nn.functional.dropout(
            input1.to(torch.double), p=0.0
        )
        rms_eps = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        eager_output = weights.to(torch.double) * (x / rms_eps)
        val_fd.validate(
            [input1, input2, weights],
            [eager_output.to(dtype), dropout_mask, rms_eps.to(torch.float)],
        )

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [input1, input2, weights])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_dropout_rmsnorm_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    dropout_p = 0.2

    inputs = [
        torch.randn(size, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(size, device="cuda", dtype=dtype, requires_grad=True),
        torch.ones(size[1], device="cuda", dtype=dtype, requires_grad=True),
        dropout_p,
    ]

    benchmark_fn = with_executor(executor, dropout_rmsnorm)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        benchmark_fn,
        inputs,
        iobytes=dropout_rmsnorm_fwd_iobytes(size, dtype),
    )
