# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser_direct import FusionDefinition, DataType
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def transpose_fusion_input_smem(
    fd: FusionDefinition,
    dtype: DataType,
    is_copy_transpose: bool,
    axes: list,
    rank: int,
):
    """Single input: the transposed input is read through shared memory."""
    shape = [-1] * rank
    contiguity = [True] * rank
    T0 = fd.define_tensor(shape=shape, contiguity=contiguity, dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)

    T1 = fd.ops.permute(T0, dims=axes)

    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=dtype)

    S2 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T3 = fd.ops.gt(T1, S2)
    T4 = fd.ops.where(T3, T1, S2)
    # add segmenter set to avoid presegment passes setting the output as a
    # view of the input without any data movement. It leads to pointwise
    # instead of transpose scheduler.
    # we can also expose OptimizationPassGuard to python frontend and disable
    # presegmentation passes to enforce output to be contiguous and then
    # transpose scheduler will be used.
    if is_copy_transpose:
        T5 = fd.ops.segment_set(T4)
        fd.add_output(T5)
    else:
        fd.add_output(T4)


def transpose_fusion_output_smem(
    fd: FusionDefinition,
    dtype: DataType,
    is_copy_transpose: bool,
    axes: list,
    rank: int,
):
    """Two inputs: the transposed output is written through shared memory."""
    shape = [-1] * rank
    contiguity = [True] * rank
    T0 = fd.define_tensor(shape=shape, contiguity=contiguity, dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=shape, contiguity=contiguity, dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T2 = fd.ops.add(T0, T1)
    T3 = fd.ops.permute(T2, dims=axes)

    if dtype in PROMOTE_DTYPES:
        T3 = fd.ops.cast(T3, dtype=dtype)

    S4 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T5 = fd.ops.gt(T3, S4)
    T6 = fd.ops.where(T5, T3, S4)
    # add segmenter set to avoid presegment passes setting the output as a
    # view of the input without any data movement. It leads to pointwise
    # instead of transpose scheduler.
    # we can also expose OptimizationPassGuard to python frontend and disable
    # presegmentation passes to enforce output to be contiguous and then
    # transpose scheduler will be used.
    if is_copy_transpose:
        T7 = fd.ops.segment_set(T6)
        fd.add_output(T7)
    else:
        fd.add_output(T6)


# Without contiguous, transpose returns a view with swapped strides.
# contiguous() materializes a contiguous copy of the result.
# When compiled with thunder, contiguous version will use nvFuser's transpose scheduler, otherwise it will use the pointwise scheduler.
def transpose_fwd_fn(
    inputs: list,
):  # [input1, input2 (optional), axes, is_copy_transpose]
    is_copy_transpose = inputs[-1]
    axes = inputs[-2]
    if len(inputs) == 4:
        data = inputs[0] + inputs[1]
    else:
        data = inputs[0]
    relu_transpose_result = torch.nn.functional.relu(data.permute(axes))
    if is_copy_transpose:
        return relu_transpose_result.contiguous()
    else:
        return relu_transpose_result


def setup_input_smem(size, dtype, is_copy_transpose, axes, dims):
    """Single input: the transposed input is read through shared memory."""
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    nvfuser_inputs = [input1]

    with FusionDefinition() as fd:
        transpose_fusion_input_smem(
            fd,
            torch_dtype_to_nvfuser_dtype(dtype),
            is_copy_transpose,
            axes,
            rank=dims,
        )

    eager_inputs = [input1, axes, is_copy_transpose]
    return fd, nvfuser_inputs, eager_inputs


def setup_output_smem(size, dtype, is_copy_transpose, axes, dims):
    """Two inputs: the transposed output is written through shared memory."""
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    input2 = torch.randn(size, device="cuda", dtype=dtype)
    nvfuser_inputs = [input1, input2]

    with FusionDefinition() as fd:
        transpose_fusion_output_smem(
            fd,
            torch_dtype_to_nvfuser_dtype(dtype),
            is_copy_transpose,
            axes,
            rank=dims,
        )

    eager_inputs = [input1, input2, axes, is_copy_transpose]
    return fd, nvfuser_inputs, eager_inputs


def _generate_transpose_params():
    params = []
    for dims in (2, 3):
        sizes = generate_input_sizes(dims=dims)
        if dims == 2:
            axes_list = [(1, 0)]
        else:
            axes_list = [(1, 0, 2), (2, 1, 0), (0, 2, 1)]
        for size in sizes:
            for axes in axes_list:
                params.append((size, axes, dims))
    return params


@pytest.mark.parametrize("size,axes,dims", _generate_transpose_params())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "is_copy_transpose",
    [
        pytest.param(True, marks=pytest.mark.transpose, id="copy"),
        pytest.param(False, marks=pytest.mark.pointwise, id="view"),
    ],
)
@pytest.mark.parametrize(
    "setup_fn",
    [setup_input_smem, setup_output_smem],
    ids=["input_smem", "output_smem"],
)
def test_transpose_nvf_benchmark(
    benchmark,
    size: tuple,
    is_copy_transpose: bool,
    dtype: torch.dtype,
    axes: tuple,
    dims: int,
    setup_fn,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    fd, nvfuser_inputs, eager_inputs = setup_fn(
        size, dtype, is_copy_transpose, axes, dims
    )

    if not disable_validation:
        eager_output = transpose_fwd_fn(eager_inputs)
        fd.validate(nvfuser_inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, nvfuser_inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size,axes,dims", _generate_transpose_params())
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "is_copy_transpose",
    [True, False],
    ids=["copy", "view"],
)
@pytest.mark.parametrize(
    "setup_fn",
    [setup_input_smem, setup_output_smem],
    ids=["input_smem", "output_smem"],
)
def test_transpose_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    is_copy_transpose: bool,
    axes: tuple,
    dims: int,
    setup_fn,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    _, _, eager_inputs = setup_fn(size, dtype, is_copy_transpose, axes, dims)
    benchmark_fn = with_executor(executor, transpose_fwd_fn)
    run_benchmark(benchmark, benchmark_fn, eager_inputs)
