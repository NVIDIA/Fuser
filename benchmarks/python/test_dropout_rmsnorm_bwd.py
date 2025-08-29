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
    compute_total_iobytes,
    with_executor,
    DEFAULT_EXECUTORS,
)
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
from .torch_ops import dropout_rmsnorm


def dropout_rmsnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    dropout_p: float,
) -> None:
    """
    Backward pass fusion definition for computing:
        output = rmsnorm (input2 + dropout (input1, p=dropout_p))

    Fusion inputs: input, dropout_mask, rms, grad_output, weights
    Fusion outputs: grad_input, grad_weights
    """
    T5 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # input1
    T4 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # input2
    T6 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Bool, is_cpu=False
    )  # dropout_mask
    T7 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )  # rms_eps
    T8 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # grads
    T9 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False
    )  # weights

    if dtype in PROMOTE_DTYPES:
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
        T5 = fd.ops.cast(T5, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        T8 = fd.ops.cast(T8, dtype=DataType.Float)
        T9 = fd.ops.cast(T9, dtype=DataType.Float)

    T12 = fd.ops.mul(T5, T6)
    S13 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T14 = fd.ops.mul(T12, S13)
    T15 = fd.ops.add(T4, T14)

    V19 = T5.shape()
    T20 = fd.ops.broadcast_in_dim(T7, shape=V19, broadcast_dims=[0, 1])
    T22 = fd.ops.reciprocal(T20)
    T23 = fd.ops.mul(T15, T22)

    T27 = fd.ops.broadcast_in_dim(T9, shape=V19, broadcast_dims=[1])

    T30 = fd.ops.mul(T8, T23)
    T31 = fd.ops.mul(T8, T27)
    T32 = fd.ops.sum(T30, dims=[0], keepdim=False, dtype=DataType.Null)

    T35 = fd.ops.mul(T31, T22)
    T36 = fd.ops.neg(T31)
    T37 = fd.ops.mul(T36, T15)
    S38 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T39 = fd.ops.pow(T20, S38)
    T40 = fd.ops.reciprocal(T39)
    T41 = fd.ops.mul(T37, T40)
    T42 = fd.ops.sum(T41, dims=[1], keepdim=False, dtype=DataType.Null)

    V60 = fd.define_vector([T5.size(0), 1], dtype=DataType.Int)
    T47 = fd.ops.broadcast_in_dim(T42, shape=V60, broadcast_dims=[0])

    T50 = fd.ops.mul(S38, T7)
    T51 = fd.ops.reciprocal(T50)
    T52 = fd.ops.mul(T47, T51)
    S55 = fd.ops.reciprocal(T5.size(1))
    T56 = fd.ops.mul(T52, S55)
    T57 = fd.ops.sum(T56, dims=[1], keepdim=False, dtype=DataType.Null)

    T61 = fd.ops.broadcast_in_dim(T57, shape=V60, broadcast_dims=[0])
    T65 = fd.ops.broadcast_in_dim(T61, shape=V19, broadcast_dims=[0, 1])
    T66 = fd.ops.mul(T65, S38)
    T69 = fd.ops.mul(T66, T15)
    T70 = fd.ops.add(T35, T69)

    T73 = fd.ops.mul(T70, S13)
    T74 = fd.ops.mul(T73, T6)

    if dtype in PROMOTE_DTYPES:
        T70 = fd.ops.cast(T70, dtype=dtype)
        T74 = fd.ops.cast(T74, dtype=dtype)
        T32 = fd.ops.cast(T32, dtype=dtype)

    fd.add_output(T74)
    fd.add_output(T70)
    fd.add_output(T32)


def dropout_rmsnorm_bwd_iobytes(size, dtype):
    # Manual IOByte computation is required since nvFuser input/outputs differ from baseline outputs (output, grad_out).
    nvf_inp_out = {
        # Inputs
        "input1": (size, dtype),
        "input2": (size, dtype),
        "weights": (size[1], dtype),
        "rms": (size[0], torch.float),
        "grad_out": (size, dtype),
        "dropout_mask": (size, torch.bool),
        # Outputs
        "grad_in1": (size, dtype),
        "grad_in2": (size, dtype),
        "grad_weights": (size[1], dtype),
    }
    return compute_total_iobytes(nvf_inp_out)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
@pytest.mark.inner_persistent
def test_dropout_rmsnorm_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    input1 = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    input2 = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    dropout_p = 0.2
    dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)

    # Manually compute dropout for validation
    x = input2 + 1 / (1 - dropout_p) * dropout_mask * input1
    squared_mean = (x.to(torch.float) ** 2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + eps)

    with FusionDefinition() as fd:
        dropout_rmsnorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), dropout_p)

    if not disable_validation:
        eager_output = weights.to(torch.double) * (
            x.to(torch.double) / rms_eps.to(torch.double)
        )
        eager_output.backward(grads.to(torch.double))
        fd.validate(
            [input1, input2, dropout_mask, rms_eps, grads, weights],
            [input1.grad, input2.grad, weights.grad],
        )

    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            fd.execute,
            [input1, input2, dropout_mask, rms_eps, grads, weights],
        )


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
@pytest.mark.inner_persistent
def test_dropout_rmsnorm_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    dropout_p = 0.2
    input1 = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    input2 = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    fwd_fn = with_executor(executor, dropout_rmsnorm)
    fwd_inputs = [input1, input2, weights, dropout_p]
    outputs = fwd_fn(fwd_inputs)

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=dropout_rmsnorm_bwd_iobytes(size, dtype),
    )
