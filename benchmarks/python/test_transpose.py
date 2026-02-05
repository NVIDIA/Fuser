# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser_direct import FusionDefinition, DataType
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def transpose_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    is_copy_transpose: bool,
    axes: list,
    rank: int,
    num_inputs: int = 2,
):
    shape = [-1] * rank
    contiguity = [True] * rank
    T0 = fd.define_tensor(shape=shape, contiguity=contiguity, dtype=dtype, is_cpu=False)

    if num_inputs == 2:
        T1 = fd.define_tensor(
            shape=shape, contiguity=contiguity, dtype=dtype, is_cpu=False
        )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        if num_inputs == 2:
            T1 = fd.ops.cast(T1, dtype=DataType.Float)

    if num_inputs == 2:
        T4 = fd.ops.add(T0, T1)
    else:
        T4 = T0
    T5 = fd.ops.permute(T4, dims=axes)

    if dtype in PROMOTE_DTYPES:
        T5 = fd.ops.cast(T5, dtype=dtype)

    S6 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T7 = fd.ops.gt(T5, S6)
    T9 = fd.ops.where(T7, T5, S6)
    # add segmenter set to avoid presegment passes setting the output as a view of the input without any data movement. It leads to pointwise instead of transpose scheduler.
    # we can also expose OptimizationPassGuard to python frontend and disable presegmentation passes to enforce output to be contiguous and then transpose scheduler will be used.
    if is_copy_transpose:
        T10 = fd.ops.segment_set(T9)
        fd.add_output(T10)
    else:
        fd.add_output(T9)


# Without contiguous, transpose returns a view with swapped strides.
# contiguous() materializes a contiguous copy of the result.
# When compiled with thunder, contiguous version will use nvFuser's transpose scheduler, otherwise it will use the pointwise scheduler.
def transpose_fwd_fn(
    inputs: list,
):  # [input1, input2 (optional), dim0, dim1, is_copy_transpose, num_inputs]
    num_inputs = inputs[-1]
    is_copy_transpose = inputs[-2]
    if num_inputs == 2:
        data = inputs[0] + inputs[1]
        dim0, dim1 = inputs[2], inputs[3]
    else:
        data = inputs[0]
        dim0, dim1 = inputs[1], inputs[2]
    relu_transpose_result = torch.nn.functional.relu(torch.transpose(data, dim0, dim1))
    if is_copy_transpose:
        return relu_transpose_result.contiguous()
    else:
        return relu_transpose_result


def _generate_transpose_params():
    params = []
    for dims in (2, 3):
        sizes = generate_input_sizes(dims=dims)
        axes_list = [(0, 1)] if dims == 2 else [(0, 1), (0, 2), (1, 2)]
        for size in sizes:
            for axes in axes_list:
                params.append((size, axes, dims))
    return params


@pytest.mark.parametrize("size,axes,dims", _generate_transpose_params())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "is_copy_transpose",
    [True, False],
    ids=["copy_transpose", "view_transpose"],
)
@pytest.mark.parametrize(
    "num_inputs",
    [1, 2],
    ids=["1_input", "2_inputs"],
)
@pytest.mark.pointwise
def test_transpose_nvf_benchmark(
    benchmark,
    size: tuple,
    is_copy_transpose: bool,
    dtype: torch.dtype,
    axes: tuple,
    dims: int,
    num_inputs: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    inputs = [input1]
    if num_inputs == 2:
        input2 = torch.randn(size, device="cuda", dtype=dtype)
        inputs.append(input2)

    permute_axes = list(range(len(size)))
    permute_axes[axes[0]], permute_axes[axes[1]] = (
        permute_axes[axes[1]],
        permute_axes[axes[0]],
    )

    with FusionDefinition() as fd:
        transpose_fusion(
            fd,
            torch_dtype_to_nvfuser_dtype(dtype),
            is_copy_transpose,
            permute_axes,
            rank=dims,
            num_inputs=num_inputs,
        )

    if not disable_validation:
        if num_inputs == 2:
            eager_output = transpose_fwd_fn(
                [input1, input2, axes[0], axes[1], is_copy_transpose, num_inputs]
            )
        else:
            eager_output = transpose_fwd_fn(
                [input1, axes[0], axes[1], is_copy_transpose, num_inputs]
            )
        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size,axes,dims", _generate_transpose_params())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "is_copy_transpose",
    [True, False],
    ids=["copy_transpose", "view_transpose"],
)
@pytest.mark.parametrize(
    "num_inputs",
    [1, 2],
    ids=["1_input", "2_inputs"],
)
def test_transpose_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    is_copy_transpose: bool,
    axes: tuple,
    dims: int,
    num_inputs: int,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    input1 = torch.randn(size, device="cuda", dtype=dtype)

    benchmark_fn = with_executor(executor, transpose_fwd_fn)

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    if num_inputs == 2:
        input2 = torch.randn(size, device="cuda", dtype=dtype)
        run_benchmark(
            benchmark,
            benchmark_fn,
            [input1, input2, axes[0], axes[1], is_copy_transpose, num_inputs],
        )
    else:
        run_benchmark(
            benchmark,
            benchmark_fn,
            [input1, axes[0], axes[1], is_copy_transpose, num_inputs],
        )
