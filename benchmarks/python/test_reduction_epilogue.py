# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

# test the influence of epilogue on the performance of reduction.
# current reduction scheduler only allows epilogue to be fused with outer reduction without post reduction broadcast.
# So, in this test, only outer reduction is tested. [reduction_axis] is kept to allow the extension to inner reduction.


def reduction_epilogue_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    reduction_axis: int,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
    T2 = fd.ops.sum(T0, dims=[reduction_axis], keepdim=False)
    T3 = fd.ops.add(T2, T1)
    if dtype in PROMOTE_DTYPES:
        T3 = fd.ops.cast(T3, dtype=dtype)
    fd.add_output(T3)


def reduction_epilogue_fwd_fn(
    inputs: list,
):  # in_tensor, epilogue_tensor, reduction_axis
    return torch.sum(inputs[0], dim=inputs[2]) + inputs[1]


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0])
@pytest.mark.reduction
def test_reduction_epilogue_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    x = torch.randn(size, device="cuda", dtype=dtype)
    epilogue = torch.randn(size[reduction_axis - 1], device="cuda", dtype=dtype)
    with FusionDefinition() as fd:
        reduction_epilogue_fusion(
            fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis
        )

    if not disable_validation:
        eager_output = reduction_epilogue_fwd_fn(
            [x.to(torch.double), epilogue.to(torch.double), reduction_axis]
        )
        fd.validate([x, epilogue], [eager_output.to(dtype)])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [x, epilogue])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction_axis", [0])
@pytest.mark.reduction
def test_reduction_epilogue_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    x = torch.randn(size, device="cuda", dtype=dtype)
    epilogue = torch.randn(size[reduction_axis - 1], device="cuda", dtype=dtype)
    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation

    benchmark_fn = with_executor(executor, reduction_epilogue_fwd_fn)

    run_benchmark(
        benchmark,
        benchmark_fn,
        [x, epilogue, reduction_axis],
    )
