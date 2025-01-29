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
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np
from .torch_ops import rmsnorm


def rmsnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T4 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    ) #input
    T5 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    ) #rms
    T6 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    ) #grad
    T7 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False) #weight

    S0 = fd.define_scalar(2.0, dtype=DataType.Double)

    if dtype in PROMOTE_DTYPES:
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        T7 = fd.ops.cast(T7, dtype=DataType.Float)

    T14 = fd.ops.broadcast_in_dim(T5, shape=T4.shape(), broadcast_dims=[0, 1]) #rms_bcast
    T15 = fd.ops.reciprocal(T14) # 1/rms
    T16 = fd.ops.mul(T4, T15) # inp/rms
    T20 = fd.ops.broadcast_in_dim(T7, shape=T4.shape(), broadcast_dims=[1]) #weight_bcast

    T23 = fd.ops.mul(T6, T16) #grad*inp/rms
    T24 = fd.ops.mul(T6, T20) #grad*weight
    T25 = fd.ops.sum(T23, dims=[0], keepdim=False, dtype=DataType.Null) #sum(grad*inp/rms)

    T28 = fd.ops.mul(T24, T15) #grad*weight/rms
    T29 = fd.ops.neg(T24) # -grad*weight
    T30 = fd.ops.mul(T29, T4) #-grad*weight*inp
    T32 = fd.ops.pow(T14, S0) #rms^2
    T33 = fd.ops.reciprocal(T32) #1/rms^2
    T34 = fd.ops.mul(T30, T33) #-grad*weight*inp/rms^2
    T35 = fd.ops.sum(T34, dims=[1], keepdim=False, dtype=DataType.Null) # sum(-grad*weight*inp/rms^2)
    V39 = fd.define_vector([T4.size(0), 1], dtype=DataType.Int)
    T41 = fd.ops.broadcast_in_dim(T35, shape=V39, broadcast_dims=[0]) #bcast sum
    T43 = fd.ops.mul(S0, T5) # 2*rms
    T44 = fd.ops.reciprocal(T43) #0.5/rms
    T45 = fd.ops.mul(T41, T44) # - 0.5  * sum(g*i*w/rms^3)
    S48 = fd.ops.reciprocal(T4.size(1)) #1/8192
    T49 = fd.ops.mul(T45, S48) # - 0.5 * S0/8192
    T50 = fd.ops.sum(T49, dims=[1], keepdim=False, dtype=DataType.Null) #-0.5*sum(S0)
    T54 = fd.ops.broadcast_in_dim(T50, shape=V39, broadcast_dims=[0]) #
    T58 = fd.ops.broadcast_in_dim(T54, shape=T4.shape(), broadcast_dims=[0, 1])
    T59 = fd.ops.mul(T58, S0) #-sum(S0)
    T62 = fd.ops.mul(T59, T4) #-inp*sum(S0)
    T63 = fd.ops.add(T28, T62) #g*w*rms - i*sum(S0)

    if dtype in PROMOTE_DTYPES:
        T63 = fd.ops.cast(T63, dtype=dtype)
        T25 = fd.ops.cast(T25, dtype=dtype)

    fd.add_output(T63)
    fd.add_output(T25)

def rmsnorm_thunder_prims_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T0 = fd.define_tensor(shape=[2048, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[8192], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T2 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T3 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T4 = fd.ops.cast(T0, dtype=DataType.BFloat16)
    T8 = fd.ops.broadcast_in_dim(T4, shape=[2048, 8192], broadcast_dims=[0, 1])
    T12 = fd.ops.broadcast_in_dim(T1, shape=[2048, 8192], broadcast_dims=[1])
    T13 = fd.ops.cast(T8, dtype=DataType.Float)
    T14 = fd.ops.cast(T2, dtype=DataType.Float)
    T15 = fd.ops.cast(T3, dtype=DataType.Float)
    T16 = fd.ops.cast(T12, dtype=DataType.Float)
    T17 = fd.ops.reciprocal(T13)
    T18 = fd.ops.mul(T14, T17)
    T19 = fd.ops.mul(T16, T15)
    T20 = fd.ops.reciprocal(T13)
    T21 = fd.ops.mul(T18, T20)
    T22 = fd.ops.neg(T19)
    T23 = fd.ops.mul(T22, T21)
    T24 = fd.ops.sum(T23, dims=[1], keepdim=False, dtype=DataType.Null)
    T25 = fd.ops.cast(T24, dtype=DataType.BFloat16)
    T29 = fd.ops.broadcast_in_dim(T25, shape=[2048, 1], broadcast_dims=[0])
    S30 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T31 = fd.ops.mul(S30, T0)
    T32 = fd.ops.cast(T29, dtype=DataType.Float)
    T33 = fd.ops.reciprocal(T31)
    T34 = fd.ops.mul(T32, T33)
    S35 = fd.define_scalar(8192.00, dtype=DataType.Double)
    S36 = fd.ops.reciprocal(S35)
    T37 = fd.ops.mul(T34, S36)
    T38 = fd.ops.sum(T37, dims=[1], keepdim=False, dtype=DataType.Null)
    T42 = fd.ops.broadcast_in_dim(T38, shape=[2048, 1], broadcast_dims=[0])
    T46 = fd.ops.broadcast_in_dim(T42, shape=[2048, 8192], broadcast_dims=[0, 1])
    S47 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T48 = fd.ops.pow(T14, S47)
    S49 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T50 = fd.ops.mul(T46, S49)
    T51 = fd.ops.reciprocal(T13)
    T52 = fd.ops.mul(T14, T51)
    T53 = fd.ops.mul(T50, T48)
    T54 = fd.ops.reciprocal(T13)
    T55 = fd.ops.mul(T19, T54)
    T56 = fd.ops.mul(T52, T15)
    T57 = fd.ops.add(T55, T53)
    T58 = fd.ops.sum(T56, dims=[0], keepdim=False, dtype=DataType.Null)
    T59 = fd.ops.cast(T57, dtype=DataType.BFloat16)
    T60 = fd.ops.cast(T58, dtype=DataType.BFloat16)
    fd.add_output(T59)
    fd.add_output(T60)
   

def rmsnorm_thunder_func_fusion(
  fd, dtype
):
    T0 = fd.define_tensor(shape=[8192], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0]) #weight
    T1 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) #grad
    T2 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) #inp
    T3 = fd.define_tensor(shape=[2048, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]) #1/rms
    T7 = fd.ops.broadcast_in_dim(T0, shape=[2048, 8192], broadcast_dims=[1]) #weight_bcast
    T8 = fd.ops.cast(T1, dtype=DataType.Float) #grad
    T9 = fd.ops.cast(T7, dtype=DataType.Float) #weight
    T10 = fd.ops.mul(T9, T8) #grad*weight (T24)
    T11 = fd.ops.cast(T2, dtype=DataType.Float) #inp
    T12 = fd.ops.mul(T11, T10) #grad*weight*inp
    T13 = fd.ops.sum(T12, dims=[1], keepdim=False, dtype=DataType.Null) #sum(grad*weight*input)
    T14 = fd.ops.cast(T13, dtype=DataType.BFloat16)
    T18 = fd.ops.broadcast_in_dim(T14, shape=[2048, 1], broadcast_dims=[0])
    T19 = fd.ops.cast(T18, dtype=DataType.Float)
    S20 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T21 = fd.ops.pow(T3, S20)
    S22 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T23 = fd.ops.mul(S22, T19)
    T24 = fd.ops.mul(T23, T21)
    S25 = fd.define_scalar(8192.00, dtype=DataType.Double)
    S26 = fd.ops.reciprocal(S25)
    T27 = fd.ops.mul(T24, S26)
    T28 = fd.ops.sum(T27, dims=[1], keepdim=False, dtype=DataType.Null)
    T29 = fd.ops.cast(T3, dtype=DataType.BFloat16)
    T33 = fd.ops.broadcast_in_dim(T28, shape=[2048, 1], broadcast_dims=[0])
    T37 = fd.ops.broadcast_in_dim(T29, shape=[2048, 8192], broadcast_dims=[0, 1])
    T41 = fd.ops.broadcast_in_dim(T33, shape=[2048, 8192], broadcast_dims=[0, 1])
    T42 = fd.ops.cast(T37, dtype=DataType.Float)
    T43 = fd.ops.mul(T11, T41)
    T44 = fd.ops.mul(T42, T10)
    T45 = fd.ops.mul(T11, T42)
    T46 = fd.ops.add(T44, T43)
    T47 = fd.ops.mul(T45, T8)
    T48 = fd.ops.add(T46, T43)
    T49 = fd.ops.sum(T47, dims=[0], keepdim=False, dtype=DataType.Null)
    T50 = fd.ops.cast(T48, dtype=DataType.BFloat16)
    T51 = fd.ops.cast(T49, dtype=DataType.BFloat16)
    fd.add_output(T50)
    fd.add_output(T51)
    
    
def rmsnorm_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOBytes computation since nvfuser input/outputs (in_tensor, grad_out, rms, weigts, grad_in, grad_weights) differ from baselines (out, grad_out)
    # Total IO bytes = in_tensor (size, dtype) + grad_out (size, dtype) + rms_eps(size[0], float) + weights (size[1], dtype) +
    #       grad_in (size, dtype) + grad_weights (size[1], dtype)
    return int(
        dtype.itemsize * (3 * np.prod(size) + 2 * size[1])
        + torch.float.itemsize * size[0]
    )


@pytest.mark.parametrize("size", [(2048, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("fusion", [rmsnorm_thunder_func_fusion, rmsnorm_bwd_fusion])
@pytest.mark.inner_outer_persistent
def test_rmsnorm_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    fusion,
    eps: float = 1e-5,
):
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    squared_mean = (inputs.to(torch.float) ** 2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + eps)

    with FusionDefinition() as fd:
        fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    bwd_inputs = [inputs, rms_eps, grads, weights]
    if fusion == rmsnorm_thunder_func_fusion:
      bwd_inputs = [weights, grads, inputs, 1 / rms_eps]
    if not disable_validation:
        eager_output = weights.to(torch.double) * (
            inputs.to(torch.double) / rms_eps.to(torch.double)
        )
        eager_output.backward(grads.to(torch.double))
        print(fusion)
        fd.validate(bwd_inputs, [inputs.grad, weights.grad])
        out = fd.execute(bwd_inputs, profile=True)
        prof = fd.profile()
        # breakpoint()
        print ()

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, bwd_inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, rmsnorm)
    fwd_inputs = [inputs, weights]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=rmsnorm_bwd_iobytes(size, dtype),
    )
