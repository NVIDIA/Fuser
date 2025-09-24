# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def reduction_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    reduction_axis: int,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.sum(T0, dims=[reduction_axis], keepdim=False)
    if dtype in PROMOTE_DTYPES:
        T2 = fd.ops.cast(T2, dtype=dtype)
    fd.add_output(T2)


def reduction_fwd_fn(inputs: list):  # in_tensor, reduction_axis
    return torch.sum(inputs[0], dim=inputs[1])


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
@pytest.mark.reduction
def test_reduction_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        reduction_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = torch.sum(inputs[0].to(torch.double), dim=reduction_axis)
        fd.validate(inputs, [eager_output.to(dtype)])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0, 1])
@pytest.mark.reduction
def test_reduction_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    input = torch.randn(size, device="cuda", dtype=dtype)

    benchmark_fn = with_executor(executor, reduction_fwd_fn)

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        benchmark_fn,
        [input, reduction_axis],
    )


@pytest.mark.parametrize("reduction_size", [i * 1024 for i in range(1, 33)])
@pytest.mark.parametrize("bf16_math", [True, False])
def test_reduction_nvf_dev_benchmark(
    benchmark,
    reduction_size: int,
    bf16_math: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    dtype = torch.bfloat16
    batch_size = 16384

    def reduction_fusion(
        fd: FusionDefinition, dtype: torch.dtype, bf16_math: bool, reduction_size: int
    ) -> None:
        T0 = fd.define_tensor(
            shape=[batch_size, reduction_size],
            contiguity=[True, True],
            dtype=dtype,
            is_cpu=False,
        )
        if not bf16_math:
            T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.max(T0, dims=[1], keepdim=False)
        if not bf16_math:
            T1 = fd.ops.cast(T1, dtype=DataType.BFloat16)
        fd.add_output(T1)

    def reduction_fwd_fn(inputs: list):
        return torch.max(inputs[0], dim=1)

    inputs = [torch.randn(batch_size, reduction_size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        reduction_fusion(
            fd, torch_dtype_to_nvfuser_dtype(dtype), bf16_math, reduction_size
        )

    if not disable_validation:
        fd.validate(inputs)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
