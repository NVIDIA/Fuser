# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import FusionDefinition, DataType
from .global_params import PROMOTE_DTYPES
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
import torch
from .core import run_benchmark, unary_bwd_torch, clear_dynamo_cache, with_executor
import numpy as np


def norm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    norm: str,
    num_dims: int,
    channels_last: bool,
    eps: float = 1e-5,
    momentum: float = 0.01,
) -> None:
    """
    Fusion definition for batch norm and instance norm forward in training mode.
    """
    batch_dim = 0
    channel_dim = 1 if not channels_last else num_dims - 1
    bcast_mask = [True if i != channel_dim else False for i in range(num_dims)]
    channels_only_bcast_mask = [
        True if i != channel_dim else False for i in range(num_dims)
    ]

    reduction_axes = [i for i in range(num_dims) if i != channel_dim]
    if norm == "instance_norm":
        reduction_axes.remove(batch_dim)
        bcast_mask[batch_dim] = False

    input = fd.define_tensor(
        shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False
    )
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    running_mean = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    running_var = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )

    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)
        bias = fd.ops.cast(bias, dtype=DataType.Float)

    var, mean = fd.ops.var_mean(input, dims=reduction_axes, correction=0, keepdim=False)

    eps = fd.define_scalar(eps, dtype=DataType.Double)
    var_eps = fd.ops.add(var, eps)
    invstd = fd.ops.rsqrt(var_eps)

    invstd_bcast = fd.ops.broadcast(invstd, bcast_mask)
    mean_bcast = fd.ops.broadcast(mean, bcast_mask)
    x_sub_mean = fd.ops.sub(input, mean_bcast)

    x_norm = fd.ops.mul(x_sub_mean, invstd_bcast)

    weight = fd.ops.broadcast(weight, channels_only_bcast_mask)
    x_scaled = fd.ops.mul(x_norm, weight)
    bias = fd.ops.broadcast(bias, channels_only_bcast_mask)
    output = fd.ops.add(x_scaled, bias)

    rev_momentum = fd.define_scalar(1 - momentum, dtype=DataType.Double)
    momentum = fd.define_scalar(momentum, dtype=DataType.Double)

    updated_mean = fd.ops.add(
        fd.ops.mul(momentum, mean), fd.ops.mul(rev_momentum, running_mean)
    )
    updated_var = fd.ops.add(
        fd.ops.mul(momentum, var), fd.ops.mul(rev_momentum, running_var)
    )

    if batch_dim not in reduction_axes:
        inverse_batch_size = fd.ops.reciprocal(input.size(0))
        updated_mean = fd.ops.mul(
            fd.ops.sum(updated_mean, batch_dim), inverse_batch_size
        )
        updated_var = fd.ops.mul(fd.ops.sum(updated_var, batch_dim), inverse_batch_size)

    if dtype in PROMOTE_DTYPES:
        output = fd.ops.cast(output, dtype=dtype)

    fd.add_output(output)
    fd.add_output(mean)
    fd.add_output(invstd)


def norm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    norm: str,
    num_dims: int,
    channels_last: bool,
    eps: float = 1e-5,
) -> None:
    """
    Fusion definition for batch norm and instance norm backward in training mode.
    """
    batch_dim = 0
    channel_dim = 1 if not channels_last else num_dims - 1
    bcast_mask = [True if i != channel_dim else False for i in range(num_dims)]
    channels_only_bcast_mask = [
        True if i != channel_dim else False for i in range(num_dims)
    ]

    reduction_axes = [i for i in range(num_dims) if i != channel_dim]
    if norm == "instance_norm":
        reduction_axes.remove(batch_dim)
        bcast_mask[batch_dim] = False

    num_spatial_dims = num_dims - len(reduction_axes)

    input = fd.define_tensor(
        shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False
    )
    grad = fd.define_tensor(
        shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False
    )
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    running_mean = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    running_var = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )

    mean = fd.define_tensor(
        shape=[-1] * num_spatial_dims,
        contiguity=[True] * num_spatial_dims,
        dtype=DataType.Float,
        is_cpu=False,
    )
    invstd = fd.define_tensor(
        shape=[-1] * num_spatial_dims,
        contiguity=[True] * num_spatial_dims,
        dtype=DataType.Float,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        grad = fd.ops.cast(grad, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)

    num_features = fd.define_scalar(1)
    for ax in reduction_axes:
        num_features *= input.size(ax)

    norm = fd.ops.reciprocal(num_features)

    mean = fd.ops.broadcast(mean, bcast_mask)

    grad_sum = fd.ops.sum(grad, dims=reduction_axes, keepdim=False)

    x_sub_mean = fd.ops.sub(input, mean)
    dot_p = fd.ops.sum(fd.ops.mul(grad, x_sub_mean), dims=reduction_axes, keepdim=False)

    grad_mean = fd.ops.broadcast(fd.ops.mul(grad_sum, norm), bcast_mask)
    proj_scale = fd.ops.mul(fd.ops.mul(dot_p, norm), fd.ops.mul(invstd, invstd))
    proj_scale = fd.ops.broadcast(proj_scale, bcast_mask)

    invstd_bcast = fd.ops.broadcast(invstd, bcast_mask)
    weight = fd.ops.broadcast(weight, channels_only_bcast_mask)
    grad_scale = fd.ops.mul(weight, invstd_bcast)
    proj = fd.ops.mul(proj_scale, x_sub_mean)

    grad_input = fd.ops.mul(fd.ops.sub(fd.ops.sub(grad, proj), grad_mean), grad_scale)
    grad_weight = fd.ops.mul(dot_p, invstd)
    grad_bias = grad_sum

    if batch_dim not in reduction_axes:
        # Weights and bias are channel-only
        grad_weight = fd.ops.sum(grad_weight, batch_dim, keepdim=False)
        grad_bias = fd.ops.sum(grad_bias, batch_dim, keepdim=False)

    if dtype in PROMOTE_DTYPES:
        grad_input = fd.ops.cast(grad_input, dtype=dtype)
        grad_weight = fd.ops.cast(grad_weight, dtype=dtype)
        grad_bias = fd.ops.cast(grad_bias, dtype=dtype)

    fd.add_output(grad_input)
    fd.add_output(grad_weight)
    fd.add_output(grad_bias)


def norm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    norm: str,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    """
    Common benchmark setup for batchnorm/instance forward call in training mode.
    """

    assert norm in ["batch_norm", "instance_norm"], NotImplementedError

    # Size is assumed to be in the order N, C, ...
    num_dims = len(size)

    at_inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype)
    bias = torch.randn(size[1], device="cuda", dtype=dtype)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    if channels_last:
        at_inputs = at_inputs.to(memory_format=torch.channels_last)
        inputs = at_inputs.clone().detach().permute((0, *range(2, num_dims), 1))
    else:
        inputs = at_inputs

    with FusionDefinition() as fd:
        norm_fwd_fusion(
            fd=fd,
            dtype=torch_dtype_to_nvfuser_dtype(dtype),
            norm=norm,
            num_dims=num_dims,
            channels_last=channels_last,
        )

    if not disable_validation:
        # PyTorch expects running mean and variance to be of same type as input.
        if norm == "batch_norm":
            eager_output = torch.nn.functional.batch_norm(
                at_inputs,
                running_mean.to(dtype),
                running_var.to(dtype),
                weight=weight,
                bias=bias,
                training=True,
            )
        elif norm == "instance_norm":
            eager_output = torch.nn.functional.instance_norm(
                at_inputs,
                running_mean.to(dtype),
                running_var.to(dtype),
                weight=weight,
                bias=bias,
            )
        if channels_last:
            eager_output = eager_output.permute((0, *range(2, num_dims), 1))

        batch_dim = 0
        channel_dim = 1 if not channels_last else num_dims - 1
        reduction_axes = [i for i in range(len(size)) if i != channel_dim]
        if norm == "instance_norm":
            reduction_axes.remove(batch_dim)

        mean = inputs.to(torch.float).mean(dim=reduction_axes)
        var = inputs.to(torch.float).var(dim=reduction_axes, unbiased=False)
        invstd = 1.0 / torch.sqrt(var + eps)

        fd.validate(
            [inputs, weight, bias, running_mean, running_var],
            [eager_output, mean, invstd],
        )

    if not disable_benchmarking:
        run_benchmark(
            benchmark, fd.execute, [inputs, weight, bias, running_mean, running_var]
        )


def norm_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    norm: str,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    """
    Common benchmark setup for batchnorm/instance forward call in training mode.
    """

    assert norm in ["batch_norm", "instance_norm"], NotImplementedError

    # Size is assumed to be in the order N, C, ...

    num_dims = len(size)

    at_inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    at_grads = torch.randn(size, device="cuda", dtype=dtype)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # CPP benchmarks assume mean and variance to be of type Float
    running_mean = torch.zeros(size[1], device="cuda", dtype=torch.float)
    running_var = torch.ones(size[1], device="cuda", dtype=torch.float)

    if channels_last:
        at_inputs = at_inputs.to(memory_format=torch.channels_last)
        at_inputs.retain_grad()
        at_grads = at_grads.to(memory_format=torch.channels_last)

        inputs = at_inputs.clone().detach().permute((0, *range(2, num_dims), 1))
        grads = at_grads.clone().detach().permute((0, *range(2, num_dims), 1))

    else:
        inputs = at_inputs
        grads = at_grads

    batch_dim = 0
    channel_dim = 1 if not channels_last else num_dims - 1
    reduction_axes = [i for i in range(len(size)) if i != channel_dim]
    if norm == "instance_norm":
        reduction_axes.remove(batch_dim)
    mean = inputs.to(torch.float).mean(dim=reduction_axes)
    var = inputs.to(torch.float).var(dim=reduction_axes, unbiased=False)
    invstd = 1.0 / torch.sqrt(var + eps)

    with FusionDefinition() as fd:
        norm_bwd_fusion(
            fd=fd,
            dtype=torch_dtype_to_nvfuser_dtype(dtype),
            norm=norm,
            num_dims=num_dims,
            channels_last=channels_last,
        )

    if not disable_validation:
        # PyTorch expects running mean and variance to be of same type as input.
        if norm == "batch_norm":
            eager_output = torch.nn.functional.batch_norm(
                at_inputs.to(torch.double),
                running_mean.to(torch.double),
                running_var.to(torch.double),
                weight=weight.to(torch.double),
                bias=bias.to(torch.double),
                training=True,
            )
        elif norm == "instance_norm":
            eager_output = torch.nn.functional.instance_norm(
                at_inputs.to(torch.double),
                running_mean.to(torch.double),
                running_var.to(torch.double),
                weight=weight.to(torch.double),
                bias=bias.to(torch.double),
            )

        eager_output.backward(at_grads.to(torch.double))

        if channels_last:
            eager_grad = at_inputs.grad.permute((0, *range(2, num_dims), 1))
        else:
            eager_grad = at_inputs.grad

        fd.validate(
            [inputs, grads, weight, running_mean, running_var, mean, invstd],
            [eager_grad, weight.grad, bias.grad],
        )

    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            fd.execute,
            [inputs, grads, weight, running_mean, running_var, mean, invstd],
        )


def batchnorm_fwd_fn(inputs: list):
    input, weight, bias, running_mean, running_var = inputs
    return torch.nn.functional.batch_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=True,
    )


def instancenorm_fwd_fn(inputs: list):
    input, weight, bias, running_mean, running_var = inputs
    return torch.nn.functional.instance_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
    )


def norm_fwd_iobytes(size: tuple, dtype: torch.dtype, norm: str):
    # Manual IOBytes computation required since nvFuser outputs (out, mean, invstd) differs from baselines (out)
    # size = [N, C, H, W]
    # Total IO bytes = in_tensor (size, dtype) + weight (size[1], dtype) + bias (size[1], dtype) +
    #           running_mean (size[1], float) + running_var (size[1], float) + output (size, dtype) +
    #           mean ([C]/[N, C] , float) + invstd ([C]/[N, C] , float)
    stat_size = (
        size[1] if norm == "batch_norm" else size[0] * size[1]
    )  # size of mean/invstd
    return int(
        dtype.itemsize * 2 * (np.prod(size) + size[1])
        + torch.float.itemsize * 2 * (size[1] + stat_size)
    )


def norm_bwd_iobytes(size: tuple, dtype: torch.dtype, norm: str):
    # Manual IOBytes computation since nvfuser input/outputs (in_tensor, grad_out, mean, invstd, weigts, grad_in, grad_weight, grad_bias) differ from baselines (out, grad_out)
    # size = [N, C, H, W]
    # Total IO bytes = in_tensor (size, dtype) + weight (size[1], dtype) +
    #           running_mean (size[1], float) + running_var (size[1], float) +
    #           mean ([C]/[N, C] , float) + invstd ([C]/[N, C] , float) + grad_out (size, dtype) +
    #           grad_in (size, dtype) + grad_weight (size[1], dtype) + grad_bias (size[1], dtype)
    stat_size = size[1] if norm == "batch_norm" else size[0] * size[1]
    return int(
        dtype.itemsize * 3 * (np.prod(size) + size[1])
        + torch.float.itemsize * 2 * (size[1] + stat_size)
    )


def norm_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    executor: str,
    norm: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    assert norm in ["batch_norm", "instance_norm"], NotImplementedError

    # Size is assumed to be in the order N, C, ...
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    running_mean = torch.zeros(size[1], device="cuda", dtype=dtype)
    running_var = torch.ones(size[1], device="cuda", dtype=dtype)
    if channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)

    norm_fwd_fn = batchnorm_fwd_fn if norm == "batch_norm" else instancenorm_fwd_fn

    benchmark_fn = with_executor(executor, norm_fwd_fn)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, weight, bias, running_mean, running_var],
        iobytes=norm_fwd_iobytes(size, dtype, norm),
    )


def norm_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    executor: str,
    norm: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    assert norm in ["batch_norm", "instance_norm"], NotImplementedError

    # Size is assumed to be in the order N, C, ...
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    running_mean = torch.zeros(size[1], device="cuda", dtype=dtype)
    running_var = torch.ones(size[1], device="cuda", dtype=dtype)
    grads = torch.randn(size, device="cuda", dtype=dtype)

    if channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)
        grads = grads.to(memory_format=torch.channels_last)

    norm_fwd_fn = batchnorm_fwd_fn if norm == "batch_norm" else instancenorm_fwd_fn

    fwd_fn = with_executor(executor, norm_fwd_fn)
    fwd_inputs = [inputs, weight, bias, running_mean, running_var]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=norm_bwd_iobytes(size, dtype, norm),
    )
