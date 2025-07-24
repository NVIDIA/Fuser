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
from .torch_ops import dropout_layernorm


def dropout_layernorm_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, dropout_p: float
) -> None:
    """
    Backward pass fusion definition for computing:
        output = layernorm (input2 + dropout (input1 p=dropout_p))

    Fusion inputs: inputs, dropout_mask, rms, grads, weights
    Fusion outputs: grad_input, grad_weights, grad_bias
    """
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Bool, is_cpu=False
    )  # mask
    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )  # mean
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )  # invstd
    T4 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # grads
    T5 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False
    )  # weights
    T6 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # input1

    T7 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )  # input2

    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
        T5 = fd.ops.cast(T5, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        T7 = fd.ops.cast(T7, dtype=DataType.Float)

    T9 = fd.ops.mul(T6, T1)
    S10 = fd.define_scalar(1 / (1 - dropout_p), dtype=DataType.Double)
    T11 = fd.ops.mul(T9, S10)
    T12 = fd.ops.add(T7, T11)

    V15 = fd.define_vector([T6.size(0), 1], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T2, shape=V15, broadcast_dims=[0])
    V19 = T6.shape()
    T20 = fd.ops.broadcast_in_dim(T16, shape=V19, broadcast_dims=[0, 1])
    T21 = fd.ops.sub(T12, T20)
    T25 = fd.ops.broadcast_in_dim(T3, shape=V19, broadcast_dims=[0, 1])
    T26 = fd.ops.mul(T21, T25)
    T30 = fd.ops.broadcast_in_dim(T5, shape=V19, broadcast_dims=[1])
    T35 = fd.ops.sum(T4, dims=[0], keepdim=False, dtype=DataType.Null)

    T37 = fd.ops.mul(T4, T30)
    T38 = fd.ops.mul(T4, T26)
    T39 = fd.ops.sum(T38, dims=[0], keepdim=False, dtype=DataType.Null)

    T41 = fd.ops.mul(T37, T25)
    T42 = fd.ops.mul(T37, T21)
    T43 = fd.ops.sum(T42, dims=[1], keepdim=False, dtype=DataType.Null)
    T47 = fd.ops.broadcast_in_dim(T43, shape=V15, broadcast_dims=[0])
    T48 = fd.ops.neg(T41)
    T49 = fd.ops.sum(T48, dims=[1], keepdim=False, dtype=DataType.Null)
    T53 = fd.ops.broadcast_in_dim(T49, shape=V15, broadcast_dims=[0])
    S54 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T55 = fd.ops.mul(S54, T47)
    S56 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T57 = fd.ops.pow(T3, S56)
    T58 = fd.ops.mul(T55, T57)
    T61 = fd.ops.sum(T53, dims=[1], keepdim=False, dtype=DataType.Null)
    T62 = fd.ops.sum(T58, dims=[1], keepdim=False, dtype=DataType.Null)
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

    if dtype in PROMOTE_DTYPES:
        T35 = fd.ops.cast(T35, dtype=dtype)
        T39 = fd.ops.cast(T39, dtype=dtype)
        T97 = fd.ops.cast(T97, dtype=dtype)
        T101 = fd.ops.cast(T101, dtype=dtype)

    fd.add_output(T101)
    fd.add_output(T97)
    fd.add_output(T39)
    fd.add_output(T35)


def dropout_layernorm_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOByte computation is required since nvFuser input/outputs differ from baseline outputs (output, grad_out).
    nvf_inp_out = {
        # Inputs
        "input1": (size, dtype),
        "input2": (size, dtype),
        "weights": (size[1], dtype),
        "mean": (size[0], torch.float),
        "invstd": (size[0], torch.float),
        "grad_out": (size, dtype),
        "dropout_mask": (size, torch.bool),
        # Outputs
        "grad_in1": (size, dtype),
        "grad_in2": (size, dtype),
        "grad_weights": (size[1], dtype),
        "grad_bias": (size[1], dtype),
    }
    return compute_total_iobytes(nvf_inp_out)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
def test_dropout_layernorm_bwd_nvf_benchmark(
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
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    dropout_p = 0.2
    dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)
    # Manually compute dropout for validation
    x = input2 + 1 / (1 - dropout_p) * dropout_mask * input1

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
            input1.shape[1:],
            weight=weights.to(torch.double),
            bias=bias.to(torch.double),
        )

        eager_output.backward(grads.to(torch.double))
        fd.validate(
            [dropout_mask, mean, invstd, grads, weights, input1, input2],
            [input1.grad, input2.grad, weights.grad, bias.grad],
        )
    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            fd.execute,
            [dropout_mask, mean, invstd, grads, weights, input1, input2],
        )


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dropout_layernorm_bwd_baseline_benchmark(
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
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, dropout_layernorm)
    fwd_inputs = [input1, input2, weights, bias, dropout_p]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=dropout_layernorm_bwd_iobytes(size, dtype),
    )
