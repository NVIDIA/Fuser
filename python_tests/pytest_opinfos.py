# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import jax
from pytest_core import OpInfo, ReferenceType, Domain
from pytest_input_generators import (
    broadcast_error_generator,
    broadcast_in_dim_generator,
    broadcast_in_dim_error_generator,
    cat_generator,
    cat_error_generator,
    define_tensor_generator,
    define_tensor_error_generator,
    define_vector_generator,
    define_vector_error_generator,
    elementwise_unary_generator,
    _elementwise_unary_torch,
    gather_generator,
    index_select_generator,
    index_select_error_generator,
    pad_error_generator,
    permute_generator,
    permute_error_generator,
    reduction_error_generator,
    reshape_generator,
    reshape_error_generator,
    slice_generator,
    slice_error_generator,
    take_along_axis_generator,
    take_along_axis_error_generator,
    var_mean_generator,
)
from pytest_utils import float_complex_dtypes
from functools import partial
from typing import List

eps = 1e-2

opinfos = []

""" Start Fusion Input Operations """
fusion_input_ops = []

define_tensor_opinfo = OpInfo(
    lambda fd: fd.define_tensor,
    "define_tensor",
    sample_input_generator=define_tensor_generator,
    error_input_generator=define_tensor_error_generator,
    is_fusion_input_op=True,
)
fusion_input_ops.append(define_tensor_opinfo)

# NOTE: this operation only supports vectors of integers that represent
# tensor shapes and is not a general interface for defining vectors of
# data.  Vectors of data should be handled with a 1D `define_tensor`.
define_vector_opinfo = OpInfo(
    lambda fd: fd.define_vector,
    "define_vector",
    sample_input_generator=define_vector_generator,
    error_input_generator=define_vector_error_generator,
    is_fusion_input_op=True,
)
fusion_input_ops.append(define_vector_opinfo)

""" End Fusion Input Operations """

""" Start Unary-Float Operations """
elementwise_unary_ops = []

acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=Domain(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acos),
)
elementwise_unary_ops.append(acos_opinfo)

""" End Unary-Float Operations """

""" Start Normalization Operations """
normalization_ops = []

var_mean_opinfo = OpInfo(
    lambda fd: fd.ops.var_mean,
    "var_mean",
    dtypes=float_complex_dtypes,
    sample_input_generator=var_mean_generator,
    error_input_generator=reduction_error_generator,
    reference=torch.var_mean,
    symbolic_parameter_list=(True, False),
)
normalization_ops.append(var_mean_opinfo)

""" End Normalization Operations """

""" Start Shape Operations """

shape_ops = []

cat_opinfo = OpInfo(
    lambda fd: fd.ops.cat,
    "cat",
    sample_input_generator=cat_generator,
    error_input_generator=cat_error_generator,
    reference=torch.cat,
    symbolic_parameter_list=(True, False),
)
shape_ops.append(cat_opinfo)

broadcast_opinfo = OpInfo(
    lambda fd: fd.ops.broadcast,
    "broadcast",
    error_input_generator=broadcast_error_generator,
    symbolic_parameter_list=(True, False),
)
shape_ops.append(broadcast_opinfo)

broadcast_in_dim_opinfo = OpInfo(
    lambda fd: fd.ops.broadcast_in_dim,
    "broadcast_in_dim",
    sample_input_generator=broadcast_in_dim_generator,
    error_input_generator=broadcast_in_dim_error_generator,
    reference=jax.lax.broadcast_in_dim,
    reference_type=ReferenceType.Jax,
    symbolic_parameter_list=(True, False, False),
)
shape_ops.append(broadcast_in_dim_opinfo)


# translate between nvfuser and pytorch argument order for gather, take_along_dim, and index_select
def gather_wrapper(fn: callable, input: torch.Tensor, index: torch.Tensor, dim: int):
    return fn(input, dim, index)


gather_opinfo = OpInfo(
    lambda fd: fd.ops.gather,
    "gather",
    sample_input_generator=gather_generator,
    error_input_generator=take_along_axis_error_generator,
    reference=partial(gather_wrapper, torch.gather),
    symbolic_parameter_list=(True, True, False),
)
shape_ops.append(gather_opinfo)

index_select_opinfo = OpInfo(
    lambda fd: fd.ops.index_select,
    "index_select",
    sample_input_generator=index_select_generator,
    error_input_generator=index_select_error_generator,
    reference=partial(gather_wrapper, torch.index_select),
    symbolic_parameter_list=(True, True, False),
)
shape_ops.append(index_select_opinfo)

# NvFuser's API is significantly different than JAX.
# TODO: Change python frontend api to match JAX using a cpp wrapper function.
pad_opinfo = OpInfo(
    lambda fd: fd.ops.pad,
    "pad",
    error_input_generator=pad_error_generator,
    symbolic_parameter_list=(True, False, True),
)
shape_ops.append(pad_opinfo)


permute_opinfo = OpInfo(
    lambda fd: fd.ops.permute,
    "permute",
    sample_input_generator=permute_generator,
    error_input_generator=permute_error_generator,
    reference=torch.permute,
    symbolic_parameter_list=(True, False),
)
shape_ops.append(permute_opinfo)


# nvfuser expects input and output shapes while pytorch only requires the output shape.
def reshape_wrapper(
    fn: callable, input: torch.Tensor, input_shape: List[int], output_shape: List[int]
):
    return fn(input, output_shape)


reshape_opinfo = OpInfo(
    lambda fd: fd.ops.reshape,
    "reshape",
    sample_input_generator=reshape_generator,
    error_input_generator=reshape_error_generator,
    reference=partial(reshape_wrapper, torch.reshape),
    symbolic_parameter_list=(True, False, False),
)
shape_ops.append(reshape_opinfo)


slice_opinfo = OpInfo(
    lambda fd: fd.ops.slice,
    "slice",
    sample_input_generator=slice_generator,
    error_input_generator=slice_error_generator,
    reference=jax.lax.slice,
    reference_type=ReferenceType.Jax,
)
shape_ops.append(slice_opinfo)

take_along_axis_opinfo = OpInfo(
    lambda fd: fd.ops.take_along_axis,
    "take_along_dim",
    sample_input_generator=take_along_axis_generator,
    error_input_generator=take_along_axis_error_generator,
    reference=torch.take_along_dim,
    symbolic_parameter_list=(True, True, False),
)
shape_ops.append(take_along_axis_opinfo)

""" End Shape Operations """

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)
opinfos.extend(fusion_input_ops)
opinfos.extend(normalization_ops)
opinfos.extend(shape_ops)
