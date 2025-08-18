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
from .torch_ops import dropout_layernorm


def dropout_layernorm_fwd_fusion(
    fd: FusionDefinition, dtype: DataType, dropout_p: float, eps: float = 1e-5
) -> None:
    """
    Forward pass fusion definition for computing:
        output = layernorm (input2 + dropout (input1, p=dropout_p))

    Fusion inputs: input1, input2, weights, bias
    Fusion outputs: output, mean, invstd, dropout_mask
    """
    T2 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T3 = fd.define_tensor(
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
        T3 = fd.ops.cast(T3, dtype=DataType.Float)

    # Dropout + Add
    T13 = fd.ops.mul(T2, T11)
    S14 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T15 = fd.ops.mul(T13, S14)
    T16 = fd.ops.add(T3, T15)
    # Layernorm
    T17, T18 = fd.ops.var_mean(T16, dims=[1], correction=0, keepdim=False)
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


def dropout_layernorm_fwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser outputs differ from baseline outputs (output).
    nvf_inp_out = {
        # Inputs
        "input1": (size, dtype),
        "input2": (size, dtype),
        "weights": (size[1], dtype),
        "bias": (size[1], dtype),
        # Outputs
        "mean": (size[0], torch.float),
        "invstd": (size[0], torch.float),
        "outputs": (size, dtype),
        "dropout_mask": (size, torch.bool),
    }
    return compute_total_iobytes(nvf_inp_out)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_dropout_layernorm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = [
        torch.randn(size, device="cuda", dtype=dtype),
        torch.randn(size, device="cuda", dtype=dtype),
        torch.ones(size[1], device="cuda", dtype=dtype),
        torch.zeros(size[1], device="cuda", dtype=dtype),
    ]

    dropout_p = 0.2
    with FusionDefinition() as fd:
        dropout_layernorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p)

    if not disable_validation:
        # For validating use a fusion definition with dropout_p=0.0
        with FusionDefinition() as val_fd:
            dropout_layernorm_fwd_fusion(
                val_fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p=0.0, eps=eps
            )

        dropout_mask = torch.ones(size, dtype=torch.bool, device="cuda")

        # dropout + add
        x = inputs[1] + torch.nn.functional.dropout(inputs[0], p=0.0)
        # layernorm
        eager_output = torch.nn.functional.layer_norm(
            x.to(torch.float),
            inputs[0].shape[1:],
            weight=inputs[2].to(torch.float),
            bias=inputs[3].to(torch.float),
        )

        # mean and invstd are computed for the output of dropout + add
        mean = x.to(torch.double).mean(dim=-1)
        variance = x.to(torch.double).var(dim=-1, unbiased=False)
        invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

        val_fd.validate(
            inputs,
            [
                eager_output.to(dtype),
                mean.to(torch.float),
                invstd.to(torch.float),
                dropout_mask,
            ],
        )

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_dropout_layernorm_fwd_baseline_benchmark(
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
        torch.zeros(size[1], device="cuda", dtype=dtype, requires_grad=True),
        dropout_p,
    ]

    benchmark_fn = with_executor(executor, dropout_layernorm)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        benchmark_fn,
        inputs,
        iobytes=dropout_layernorm_fwd_iobytes(size, dtype),
    )
