# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import itertools
from functools import partial, wraps

import torch
from torch.testing import make_tensor

from pytest_core import OpInfo, SampleInput, ErrorSample
from nvfuser import DataType


def cat_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shapes, dim
    cases = [
        ([(3,)], 0),  # single tensor provided
        # 1D
        ([(2,), (3,)], 0),
        ([(2,), (4,)], 0),
        ([(0,), (2,)], 0),
        ([(0,), (2,)], -1),
        ([(2, 3), (2, 4)], 1),
        ([(2, 3), (2, 4), (2, 5)], 1),
    ]

    for shapes, dim in cases:
        yield SampleInput([make_arg(s) for s in shapes], dim)


def cat_error_generator(op, dtype=torch.float32, requires_grad: bool = False, **kwargs):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shapes, dim, exception type, exception string
    empty_input_tensors = (
        ([], 0),
        RuntimeError,
        "Attempting to concatenate empty list of tensors",
    )
    positive_dim = (([(1,), (2,)], 1), RuntimeError, "Invalid dimension to cat")
    negative_dim = (([(2,), (2,)], -2), RuntimeError, "Invalid dimension to cat")
    # All tensors must have same number of dimension"
    ndims_mismatch = (
        ([(2,), (2, 3)], 0),
        RuntimeError,
        "Unexpected number of dimensions",
    )
    # All tensors must have same shape except for the cat dimension
    shape_mismatch = (([(2, 3), (4, 5)], 0), RuntimeError, "known_size == this_size")

    error_cases = [
        empty_input_tensors,
        positive_dim,
        negative_dim,
        ndims_mismatch,
        shape_mismatch,
    ]

    for case, ex_type, ex_str in error_cases:
        shapes, dim = case
        yield SampleInput([make_arg(s) for s in shapes], dim), ex_type, ex_str


# TODO Add small value, large value, and extremal-valued samples
def elementwise_unary_generator(
    op: OpInfo,
    dtype: torch.dtype,
    requires_grad: bool = False,
    *,
    supports_numbers: bool = True,
    **kwargs,
):
    low = None if op.domain.low is None else max(-9, op.domain.low)
    high = None if op.domain.high is None else min(9, op.domain.high)
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        low=low,
        high=high,
        requires_grad=requires_grad,
        **kwargs,
    )

    shapes = (
        # TODO: restore size zero cases
        # (0, 2, 1),
        # (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))

    # Noncontiguous inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape, noncontiguous=True))


def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)
        return op(torch.tensor(x)).item()

    return _fn


def define_tensor_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    yield SampleInput(symbolic_sizes=[-1], contiguity=[True])


def define_tensor_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    """
    "define_tensor",
    [](FusionDefinition& self,
        std::vector<int64_t>& sizes,
        std::vector<int64_t>& strides,
        PrimDataType dtype = DataType::Float,
        bool static_sizes = false,
        bool is_cpu = false) -> Tensor {
    ---
    "define_tensor",
    [](FusionDefinition& self,
        std::vector<int64_t>& symbolic_sizes,
        std::vector<std::optional<bool>>& contiguity,
        PrimDataType dtype = DataType::Float,
        bool is_cpu = false) -> Tensor {
    """

    MINIMUM_SYMBOLIC_SIZE = -1
    INT64_MAX = 9223372036854775807
    MAX_TENSOR_DIMS = 8

    check_size_contiguity_match = ErrorSample(
        {
            "symbolic_sizes": [-1, -1],
            "contiguity": [True, True, True],
            "dtype": DataType.Float,
        },
        "The size of contiguity must equal to the number of non-broadcasting IterDomains",
    )

    check_empty_tensor_size = ErrorSample(
        {"symbolic_sizes": [], "contiguity": []},
        "Empty tensor is unsupported.",
    )

    check_max_tensor_size = ErrorSample(
        {
            "symbolic_sizes": [-1 for _ in range(MAX_TENSOR_DIMS + 1)],
            "contiguity": [True for _ in range(MAX_TENSOR_DIMS + 1)],
        },
        "The specified tensor dimensionality exceeds the max tensor size for nvfuser.",
    )

    check_above_size_range = ErrorSample(
        {"symbolic_sizes": [INT64_MAX + 1], "contiguity": [True]},
        "define_tensor(): incompatible function arguments",
        TypeError,
    )

    check_below_size_range = ErrorSample(
        {"symbolic_sizes": [MINIMUM_SYMBOLIC_SIZE - 1], "contiguity": [True]},
        "The value -2 at index 0 was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1)",
    )

    check_contiguity_unknown_values = ErrorSample(
        {"symbolic_sizes": [10], "contiguity": [-1]},
        "define_tensor(): incompatible function arguments.",
        TypeError,
    )

    check_symbolic_sizes_unknown_dtypes = ErrorSample(
        {"symbolic_sizes": [10.0], "contiguity": [True]},
        "define_tensor(): incompatible function arguments.",
        TypeError,
    )

    # TODO: Fix empty and maximum tensor dimensionality error checks.
    # TODO: Add invalid argument checks for contiguity.
    error_cases = [
        check_size_contiguity_match,
        # check_empty_tensor_size,
        # check_max_tensor_size,
        check_above_size_range,
        check_below_size_range,
        # check_contiguity_unknown_values,
        check_symbolic_sizes_unknown_dtypes,
    ]

    input_tensor = make_tensor(
        (10, 10), device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    for es in error_cases:
        yield SampleInput(input_tensor, **es.kwargs), es.ex_type, es.ex_str


def take_along_axis_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    # a.shape, dim, b.shape
    cases = (
        ((4, 2, 3), 0, (8, 2, 3)),
        ((4, 2, 3), 1, (4, 1, 3)),
        ((4, 2, 3), 2, (4, 2, 5)),
        ((4,), 0, (8)),
        ((4,), 0, (1)),
        ((4, 1), 0, (3, 1)),
        ((4, 1), 1, (4, 5)),
    )

    for shape_a, dim, shape_b in cases:
        a = make_arg(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, b, dim)


def gather_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.gather(input: Tensor, dim: int, index: LongTensor)
    # * input and index tensors have same ndims.
    # * dim is within bounds

    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(
        make_tensor, device="cuda", dtype=torch.long, requires_grad=False
    )

    input_shape = (4, 2)
    valid_index_shape = (3, 1)
    invalid_index_shape = (5, 3)

    a = make_arg(input_shape)

    # dim, exception type, exception string
    positive_axis = (2, RuntimeError, "Tensor arguments have dimension")
    negative_axis = (-3, RuntimeError, "Tensor arguments have dimension")

    error_cases = [
        positive_axis,
        negative_axis,
    ]

    for dim, ex_type, ex_str in error_cases:
        b = make_index(valid_index_shape, low=0, high=10, dtype=torch.long)
        yield SampleInput(a, b, dim), ex_type, ex_str

    # TODO add index dtype check
    # b = make_index(valid_index_shape, low=0, high=input_shape[0], dtype=torch.float)
    # yield SampleInput(a, b, 0), RuntimeError, "index tensor can only be int or long dtype."

    # TODO add index out-of-bounds check
    # b = make_index(valid_index_shape, low=10, high=100, dtype=torch.long)
    # yield SampleInput(a, b, 0), RuntimeError, "out of bounds index value."

    # TODO add invalid index shape
    # b = make_index(invalid_index_shape, low=0, high=invalid_index_shape[0], dtype=torch.long)
    # index_dim_ex_str =" Expected dimension of index tensor to be smaller than input tensor except for specified axis"
    # yield SampleInput(a, b, 0), RuntimeError, index_dim_ex_str


def index_select_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(make_tensor, device="cuda", requires_grad=False)

    # a.shape, dim, b.shape
    cases = (
        ((4, 2, 3), 0, (8)),
        ((4, 2, 3), 1, (7)),
        ((4, 2, 3), 2, (2)),
        ((4,), 0, (8)),
        ((4,), 0, (1)),
        ((4, 1), 0, (3)),
        ((4, 1), 1, (5)),
        ((1, 0, 3), 0, (8)),
    )

    for shape_a, dim, shape_b in cases:
        for index_dtype in [torch.int, torch.long]:
            a = make_arg(shape_a)
            b = make_index(shape_b, low=0, high=shape_a[dim], dtype=index_dtype)
            yield SampleInput(a, b, dim)


def index_select_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    # torch.index_select(input: Tensor, dim: int, index: LongTensor)
    # * dim is within bounds
    # * index is a 1D vector
    # * index array can't have zero elements
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )
    make_index = partial(make_tensor, device="cuda", requires_grad=False)

    input_shape = (4, 2)
    index_shape = (8,)

    a = make_arg(input_shape)

    # dim, exception type, exception string
    positive_axis = (2, RuntimeError, "index_select on invalid axis")
    negative_axis = (-3, RuntimeError, "index_select on invalid axis")

    error_cases = [
        positive_axis,
        negative_axis,
    ]

    for dim, ex_type, ex_str in error_cases:
        b = make_index(index_shape, low=0, high=10, dtype=torch.long)
        yield SampleInput(a, b, dim), ex_type, ex_str

    # TODO add index dtype check
    # b = make_index(index_shape, low=0, high=input_shape[0], dtype=torch.float)
    # yield SampleInput(a, b, 0), RuntimeError, "index tensor can only be int or long dtype."

    # TODO add index out-of-bounds check
    # b = make_index(index_shape, low=10, high=100, dtype=torch.long)
    # yield SampleInput(a, b, 0), RuntimeError, "out of bounds index value."


# TODO: add stride testing
def slice_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape, start_indices, end_indices
    cases = (
        ((5, 7, 8), (1, 0, 3), (2, 6, 8)),
        ((3,), (1,), (2,)),
    )

    for shape, start_indices, end_indices in cases:
        a = make_arg(shape)
        yield SampleInput(a, start_indices=start_indices, end_indices=end_indices)


def slice_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor, device="cuda", dtype=dtype, requires_grad=requires_grad
    )

    # shape
    cases = ((10, 10), (5, 5))

    check_start_indices = ErrorSample(
        {"start_indices": [-1, -2], "end_indices": [5, 5], "strides": [7, 7]},
        "Slice operation start_indices must be greater-than-or-equal-to 0.",
    )

    check_end_indices = ErrorSample(
        {"start_indices": [3, 4], "end_indices": [1, 2], "strides": [1, 1]},
        "Slice operation end_indices must be greater-than-or-equal-to start_indices.",
    )

    check_strides = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [5, 5], "strides": [5, 5]},
        "nvFuser Limitation: All slice operation strides must be of size 1.",
    )

    check_tensor_dims = ErrorSample(
        {"start_indices": [0, 0, 0], "end_indices": [4, 4, 4], "strides": [1, 1, 1]},
        "Number of tensor dimensions does not match slice dimensions!",
    )

    check_slice_dims_start = ErrorSample(
        {"start_indices": [0, 0, 0], "end_indices": [4, 4], "strides": [1, 1]},
        "Slice start_indices and strides don't match!",
    )

    check_slice_dims_end = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [4, 4, 4], "strides": [1, 1]},
        "Slice indexing attribute dimensions don't match!",
    )

    check_slice_dims_stride = ErrorSample(
        {"start_indices": [0, 0], "end_indices": [4, 4], "strides": [1, 1, 1]},
        "Slice start_indices and strides don't match!",
    )

    error_cases = [
        check_start_indices,
        check_end_indices,
        check_strides,
        check_tensor_dims,
        check_slice_dims_start,
        check_slice_dims_end,
        check_slice_dims_stride,
    ]

    for shape, es in itertools.product(cases, error_cases):
        input_tensor = make_arg(shape)
        yield SampleInput(input_tensor, **es.kwargs), es.ex_type, es.ex_str


def reduction_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape, dim, keepdim, dtype
    cases = (
        ((4, 4), None, False, None),
        ((5,), None, True, None),
        ((5,), (0,), False, None),
        ((8, 1, 6), (1,), True, None),
        ((8, 7, 5, 1), (0, 1), True, None),
        ((8, 7, 5, 1), (1, 3), False, None),
    )

    for c in cases:
        shape, dim, keepdim, dtype = c
        yield (SampleInput(make_arg(shape), dim, keepdim, dtype=dtype))


def reduction_error_generator(
    op: OpInfo, dtype: torch.dtype, requires_grad: bool = False, **kwargs
):
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape
    cases = (
        (8, 1, 6),
        (8, 7, 5, 1),
    )

    # axes : List[int]
    # 1) all axis are int --- use float dtype
    # 2) all axes are unique --- duplicates
    # 3) after normalization, 0 <= axis[i] <= len(size)
    # 4) If empty tensor, then axis == 0

    int_dtype_axis = (
        lambda dims: float(dims),
        TypeError,
        "var_mean(): incompatible function arguments.",
    )
    duplicate_axis = (
        lambda dims: (0, 0, 0),
        RuntimeError,
        "Reduction axes are not unique",
    )
    lower_bound = (lambda dims: (-dims - 1,), RuntimeError, "Reduction on invalid axis")
    upper_bound = (lambda dims: (dims,), RuntimeError, "Reduction on invalid axis")
    # TODO Fix duplicate_axis, lower_bound, upper_bound
    error_cases = [int_dtype_axis]

    for shape, es in itertools.product(cases, error_cases):
        input_tensor = make_arg(shape)
        axis_fn, ex_type, ex_str = es
        yield SampleInput(input_tensor, axis_fn(len(shape))), ex_type, ex_str


def var_mean_generator(op: OpInfo, dtype: torch.dtype, requires_grad: bool = False):
    """torch.var_mean(input, dim=None, *, correction=1, keepdim=False)"""
    correction = (0, 1)
    samples = reduction_generator(op, dtype, requires_grad)
    for c, sample in itertools.product(correction, samples):
        a = sample.args[0]
        dim = (
            sample.args[1]
            if (len(sample.args) > 1 and sample.args[1])
            else tuple(range(a.ndim))
        )
        keepdim = sample.args[2] if len(sample.args) > 2 else False
        yield SampleInput(a, dim, correction=c, keepdim=keepdim)
