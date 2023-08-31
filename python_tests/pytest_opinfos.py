# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import math
import torch
import jax
from pytest_core import OpInfo, ReferenceType, Domain
from pytest_fusion_definitions import (
    api_test_fd_fn,
    tensor_input_fd_fn,
    tensor_api_test_fd_fn,
    vector_api_test_fd_fn,
)
from pytest_input_generators import (
    broadcast_error_generator,
    broadcast_in_dim_generator,
    broadcast_in_dim_error_generator,
    cat_generator,
    cat_error_generator,
    define_tensor_generator,
    define_tensor_error_generator,
    define_vector_constant_error_generator,
    define_vector_input_error_generator,
    elementwise_binary_generator,
    _elementwise_binary_torch,
    elementwise_unary_generator,
    _elementwise_unary_torch,
    full_error_generator,
    gather_generator,
    index_select_generator,
    index_select_error_generator,
    iota_error_generator,
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
    tensor_size_error_generator,
    var_mean_generator,
    vector_at_error_generator,
    where_error_generator,
)
from pytest_utils import (
    bool_int_dtypes,
    int_dtypes,
    full_precision_float_dtypes,
    int_float_dtypes,
    float_complex_dtypes,
    ArgumentType,
)
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
    fd_correctness_fn=tensor_input_fd_fn,
    fd_error_input_fn=tensor_input_fd_fn,
)
fusion_input_ops.append(define_tensor_opinfo)

# NOTE: "define_vector" only supports vectors of integers that represent
# tensor shapes and is not a general interface for defining vectors of
# data.  Vectors of data should be handled with a 1D `define_tensor`.
define_vector_constant_opinfo = OpInfo(
    lambda fd: fd.define_vector,
    "define_vector_constant",
    sample_input_generator=None,
    error_input_generator=define_vector_constant_error_generator,
    fd_error_input_fn=api_test_fd_fn,
)
fusion_input_ops.append(define_vector_constant_opinfo)

define_vector_input_opinfo = OpInfo(
    lambda fd: fd.define_vector,
    "define_vector_input",
    sample_input_generator=None,
    error_input_generator=define_vector_input_error_generator,
    fd_error_input_fn=api_test_fd_fn,
)
fusion_input_ops.append(define_vector_input_opinfo)

""" End Fusion Input Operations """

""" Start Unary-Float Operations """
unary_ops = []

abs_opinfo = OpInfo(
    lambda fd: fd.ops.abs,
    "abs",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.abs),
)
unary_ops.append(abs_opinfo)

acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=Domain(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acos),
)
unary_ops.append(acos_opinfo)

acosh_opinfo = OpInfo(
    lambda fd: fd.ops.acosh,
    "acosh",
    domain=Domain(-1, math.inf),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acosh),
)
unary_ops.append(acosh_opinfo)

asin_opinfo = OpInfo(
    lambda fd: fd.ops.asin,
    "asin",
    domain=Domain(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.asin),
)
unary_ops.append(asin_opinfo)

asinh_opinfo = OpInfo(
    lambda fd: fd.ops.asinh,
    "asinh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.asinh),
)
unary_ops.append(asinh_opinfo)

atan_opinfo = OpInfo(
    lambda fd: fd.ops.atan,
    "atan",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.atan),
)
unary_ops.append(atan_opinfo)

atanh_opinfo = OpInfo(
    lambda fd: fd.ops.atanh,
    "atanh",
    domain=Domain(-1 + eps, 1 + eps),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.atanh),
)
unary_ops.append(atanh_opinfo)

bitwise_not_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_not,
    "bitwise_not",
    dtypes=bool_int_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.bitwise_not),
)
unary_ops.append(bitwise_not_opinfo)

# TODO add nvfuser exception for int dtypes
ceil_opinfo = OpInfo(
    lambda fd: fd.ops.ceil,
    "ceil",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.ceil),
)
unary_ops.append(ceil_opinfo)

cos_opinfo = OpInfo(
    lambda fd: fd.ops.cos,
    "cos",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.cos),
)
unary_ops.append(cos_opinfo)

cosh_opinfo = OpInfo(
    lambda fd: fd.ops.cosh,
    "cosh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.cosh),
)
unary_ops.append(cosh_opinfo)

erf_opinfo = OpInfo(
    lambda fd: fd.ops.erf,
    "erf",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erf),
)
unary_ops.append(erf_opinfo)

erfc_opinfo = OpInfo(
    lambda fd: fd.ops.erfc,
    "erfc",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erfc),
)
unary_ops.append(erfc_opinfo)

erfcinv_opinfo = OpInfo(
    lambda fd: fd.ops.erfcinv,
    "erfcinv",
    dtypes=(
        torch.float32,
        torch.float64,
    ),
    domain=Domain(0.3, 0.7),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(lambda x: torch.erfinv(1 - x)),
)
unary_ops.append(erfcinv_opinfo)

erfinv_opinfo = OpInfo(
    lambda fd: fd.ops.erfinv,
    "erfinv",
    dtypes=int_float_dtypes,
    domain=Domain(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erfinv),
)
unary_ops.append(erfinv_opinfo)

exp_opinfo = OpInfo(
    lambda fd: fd.ops.exp,
    "exp",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.exp),
)
unary_ops.append(exp_opinfo)

exp2_opinfo = OpInfo(
    lambda fd: fd.ops.exp2,
    "exp2",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.exp2),
)
unary_ops.append(exp2_opinfo)

expm1_opinfo = OpInfo(
    lambda fd: fd.ops.expm1,
    "expm1",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.expm1),
)
unary_ops.append(expm1_opinfo)

# TODO add nvfuser exception for int dtypes
floor_opinfo = OpInfo(
    lambda fd: fd.ops.floor,
    "floor",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.floor),
)
unary_ops.append(floor_opinfo)

frac_opinfo = OpInfo(
    lambda fd: fd.ops.frac,
    "frac",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.frac),
)
unary_ops.append(frac_opinfo)

isfinite_opinfo = OpInfo(
    lambda fd: fd.ops.isfinite,
    "isfinite",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isfinite),
)
unary_ops.append(isfinite_opinfo)

isinf_opinfo = OpInfo(
    lambda fd: fd.ops.isinf,
    "isinf",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isinf),
)
unary_ops.append(isinf_opinfo)

isnan_opinfo = OpInfo(
    lambda fd: fd.ops.isnan,
    "isnan",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isnan),
)
unary_ops.append(isnan_opinfo)

# NOTE half-precision floating types are not automatically promoted to fp32
isneginf_opinfo = OpInfo(
    lambda fd: fd.ops.isneginf,
    "isneginf",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isneginf),
)
unary_ops.append(isneginf_opinfo)

# NOTE half-precision floating types are not automatically promoted to fp32
isposinf_opinfo = OpInfo(
    lambda fd: fd.ops.isposinf,
    "isposinf",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isposinf),
)
unary_ops.append(isposinf_opinfo)

isreal_opinfo = OpInfo(
    lambda fd: fd.ops.isreal,
    "isreal",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.isreal),
)
unary_ops.append(isreal_opinfo)

lgamma_opinfo = OpInfo(
    lambda fd: fd.ops.lgamma,
    "lgamma",
    dtypes=int_float_dtypes,
    domain=Domain(-1.0 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.lgamma),
)
unary_ops.append(lgamma_opinfo)

log_opinfo = OpInfo(
    lambda fd: fd.ops.log,
    "log",
    domain=Domain(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log),
)
unary_ops.append(log_opinfo)

log10_opinfo = OpInfo(
    lambda fd: fd.ops.log10,
    "log10",
    dtypes=int_float_dtypes,
    domain=Domain(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log10),
)
unary_ops.append(log10_opinfo)

log1p_opinfo = OpInfo(
    lambda fd: fd.ops.log1p,
    "log1p",
    dtypes=int_float_dtypes,
    domain=Domain(-1 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log1p),
)
unary_ops.append(log1p_opinfo)

log2_opinfo = OpInfo(
    lambda fd: fd.ops.log2,
    "log2",
    domain=Domain(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log2),
)
unary_ops.append(log2_opinfo)

neg_opinfo = OpInfo(
    lambda fd: fd.ops.neg,
    "neg",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.neg),
)
unary_ops.append(neg_opinfo)

reciprocal_opinfo = OpInfo(
    lambda fd: fd.ops.reciprocal,
    "reciprocal",
    domain=Domain(0 + eps, math.inf),
    sample_input_generator=partial(
        elementwise_unary_generator,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ),
    reference=_elementwise_unary_torch(torch.reciprocal),
)
unary_ops.append(reciprocal_opinfo)

# TODO add nvfuser exception for int dtypes
round_opinfo = OpInfo(
    lambda fd: fd.ops.round,
    "round",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.round),
)
unary_ops.append(round_opinfo)

rsqrt_opinfo = OpInfo(
    lambda fd: fd.ops.rsqrt,
    "rsqrt",
    domain=Domain(0 + eps, math.inf),
    sample_input_generator=partial(
        elementwise_unary_generator,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ),
    reference=_elementwise_unary_torch(torch.rsqrt),
)
unary_ops.append(rsqrt_opinfo)

sigmoid_opinfo = OpInfo(
    lambda fd: fd.ops.sigmoid,
    "sigmoid",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sigmoid),
)
unary_ops.append(sigmoid_opinfo)

signbit_opinfo = OpInfo(
    lambda fd: fd.ops.signbit,
    "signbit",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.signbit),
)
unary_ops.append(signbit_opinfo)

sin_opinfo = OpInfo(
    lambda fd: fd.ops.sin,
    "sin",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sin),
)
unary_ops.append(sin_opinfo)

sinh_opinfo = OpInfo(
    lambda fd: fd.ops.sinh,
    "sinh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sinh),
)
unary_ops.append(sinh_opinfo)

sqrt_opinfo = OpInfo(
    lambda fd: fd.ops.sqrt,
    "sqrt",
    domain=Domain(0, math.inf),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sqrt),
)
unary_ops.append(sqrt_opinfo)

tan_opinfo = OpInfo(
    lambda fd: fd.ops.tan,
    "tan",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.tan),
)
unary_ops.append(tan_opinfo)

tanh_opinfo = OpInfo(
    lambda fd: fd.ops.tanh,
    "tanh",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.tanh),
)
unary_ops.append(tanh_opinfo)

# TODO add nvfuser exception for int dtypes
trunc_opinfo = OpInfo(
    lambda fd: fd.ops.trunc,
    "trunc",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.trunc),
)
unary_ops.append(trunc_opinfo)

""" End Unary-Float Operations """

""" Start Binary Operations """

# atan2 --- promote int to float; allows fp16 and bf16
# nextafter, truediv  --- promote int to float; requires full-precision fp32, fp64
# ceildiv, div, fmod, mod, remainder, truediv --- except_zero
# add, mul, pow, sub
# bitwise_and, bitwise_or, bitwise_xor --- bool_int_only
# bitwise_left_shift, bitwise_right_shift, logical_right_shift --- int_only
# eq, ne, ge, gt, le, lt --- compare

# TODO Add "ceildiv" to python_frontend
# TODO Add support for python reference for "mod".
# TODO atan2 - complex dtypes are unsupported, but we fail when compiling kernel
# TODO logical_right_shift - domain of shift parameter is non-zero; Otherwise the result is undefined.


binary_ops = []

add_opinfo = OpInfo(
    lambda fd: fd.ops.add,
    "add",
    sample_input_generator=partial(
        elementwise_binary_generator, enable_extremal_value_testing=False
    ),
    reference=_elementwise_binary_torch(torch.add),
)
binary_ops.append(add_opinfo)

# TODO complex dtypes are unsupported, but we fail when compiling kernel
atan2_opinfo = OpInfo(
    lambda fd: fd.ops.atan2,
    "atan2",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.atan2),
)
binary_ops.append(atan2_opinfo)

bitwise_and_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_and,
    "bitwise_and",
    dtypes=bool_int_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.bitwise_and),
)
binary_ops.append(bitwise_and_opinfo)

bitwise_left_shift_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_left_shift,
    "bitwise_left_shift",
    dtypes=int_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.bitwise_left_shift),
)
binary_ops.append(bitwise_left_shift_opinfo)

bitwise_or_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_or,
    "bitwise_or",
    dtypes=bool_int_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.bitwise_or),
)
binary_ops.append(bitwise_or_opinfo)

bitwise_right_shift_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_right_shift,
    "bitwise_right_shift",
    dtypes=int_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.bitwise_right_shift),
)
binary_ops.append(bitwise_right_shift_opinfo)

bitwise_xor_opinfo = OpInfo(
    lambda fd: fd.ops.bitwise_xor,
    "bitwise_xor",
    dtypes=bool_int_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.bitwise_xor),
)
binary_ops.append(bitwise_xor_opinfo)

div_opinfo = OpInfo(
    lambda fd: fd.ops.div,
    "div",
    dtypes=float_complex_dtypes,
    sample_input_generator=partial(
        elementwise_binary_generator,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ),
    reference=_elementwise_binary_torch(torch.div),
)
binary_ops.append(div_opinfo)

eq_opinfo = OpInfo(
    lambda fd: fd.ops.eq,
    "eq",
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.eq),
)
binary_ops.append(eq_opinfo)

fmod_opinfo = OpInfo(
    lambda fd: fd.ops.fmod,
    "fmod",
    dtypes=int_float_dtypes,
    sample_input_generator=partial(elementwise_binary_generator, exclude_zero=True),
    reference=_elementwise_binary_torch(torch.fmod),
)
binary_ops.append(fmod_opinfo)

ge_opinfo = OpInfo(
    lambda fd: fd.ops.ge,
    "ge",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.ge),
)
binary_ops.append(ge_opinfo)

gt_opinfo = OpInfo(
    lambda fd: fd.ops.gt,
    "gt",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.gt),
)
binary_ops.append(gt_opinfo)

le_opinfo = OpInfo(
    lambda fd: fd.ops.le,
    "le",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.le),
)
binary_ops.append(le_opinfo)

# TODO domain of shift parameter greater than zero; Otherwise the result is undefined.
logical_right_shift_opinfo = OpInfo(
    lambda fd: fd.ops.logical_right_shift,
    "logical_right_shift",
    domain=Domain(0, None),
    dtypes=int_dtypes,
    sample_input_generator=partial(
        elementwise_binary_generator,
        enable_broadcast_testing=False,
        enable_extremal_value_testing=False,
        enable_large_value_testing=False,
        enable_small_value_testing=False,
    ),
    reference=jax.lax.shift_right_logical,
    reference_type=ReferenceType.Jax,
)
binary_ops.append(logical_right_shift_opinfo)

lt_opinfo = OpInfo(
    lambda fd: fd.ops.lt,
    "lt",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.lt),
)
binary_ops.append(lt_opinfo)

mod_opinfo = OpInfo(
    lambda fd: fd.ops.mod,
    "mod",
    dtypes=int_dtypes,
    sample_input_generator=partial(
        elementwise_binary_generator,
        exclude_zero=True,
    ),
    # Matlab rem (Remainder after Division) function
    # For more details, see https://www.mathworks.com/help/matlab/ref/rem.html
    reference=lambda a, b: a - b * torch.trunc(a / b).to(a.dtype),
)
binary_ops.append(mod_opinfo)

mul_opinfo = OpInfo(
    lambda fd: fd.ops.mul,
    "mul",
    sample_input_generator=partial(
        elementwise_binary_generator, enable_extremal_value_testing=False
    ),
    reference=_elementwise_binary_torch(torch.mul),
)
binary_ops.append(mul_opinfo)

ne_opinfo = OpInfo(
    lambda fd: fd.ops.ne,
    "ne",
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.ne),
)
binary_ops.append(ne_opinfo)

nextafter_opinfo = OpInfo(
    lambda fd: fd.ops.nextafter,
    "nextafter",
    dtypes=full_precision_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.nextafter),
)
binary_ops.append(nextafter_opinfo)

# complex dtypes --- AssertionError: Tensor-likes are not close!
pow_opinfo = OpInfo(
    lambda fd: fd.ops.pow,
    "pow",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_binary_generator,
    reference=_elementwise_binary_torch(torch.pow),
)
binary_ops.append(pow_opinfo)

remainder_opinfo = OpInfo(
    lambda fd: fd.ops.remainder,
    "remainder",
    dtypes=int_float_dtypes,
    sample_input_generator=partial(
        elementwise_binary_generator,
        exclude_zero=True,
        enable_extremal_value_testing=False,
    ),
    reference=_elementwise_binary_torch(torch.remainder),
)
binary_ops.append(remainder_opinfo)

sub_opinfo = OpInfo(
    lambda fd: fd.ops.sub,
    "sub",
    sample_input_generator=partial(
        elementwise_binary_generator, enable_extremal_value_testing=False
    ),
    reference=_elementwise_binary_torch(torch.sub),
)
binary_ops.append(sub_opinfo)

truediv_opinfo = OpInfo(
    lambda fd: fd.ops.truediv,
    "truediv",
    sample_input_generator=partial(
        elementwise_binary_generator,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ),
    reference=_elementwise_binary_torch(torch.true_divide),
)
binary_ops.append(truediv_opinfo)

# For int dtypes, nvfuser div op has the semantics of c++ / operator, so its reference is trunc_divide.
trunc_div_opinfo = OpInfo(
    lambda fd: fd.ops.div,
    "trunc_div",
    dtypes=int_dtypes,
    sample_input_generator=partial(
        elementwise_binary_generator,
        enable_small_value_testing=False,
        enable_extremal_value_testing=False,
        exclude_zero=True,
    ),
    reference=_elementwise_binary_torch(partial(torch.div, rounding_mode="trunc")),
)
binary_ops.append(trunc_div_opinfo)

""" End Binary Operations """

""" Start Ternary Operations """

ternary_ops = []

where_opinfo = OpInfo(
    lambda fd: fd.ops.where,
    "where",
    error_input_generator=where_error_generator,
)
ternary_ops.append(where_opinfo)

""" End Ternary Operations """

""" Start Dynamic Shape Enabling Operations """

dynamic_shapes_ops = []

# TODO: Add correctness testing as noted below
tensor_shape_opinfo = OpInfo(
    lambda fd: fd.ops.shape,
    "tensor_shape",
    # TODO: Check correctness once there are operators that can consume a Vector
    sample_input_generator=None,
    # NOTE: ops.shape will take any legal Tensor object where the creation of
    # Tensor inputs will check possible errors
    error_input_generator=None,
)
dynamic_shapes_ops.append(tensor_shape_opinfo)

# TODO: Add correctness testing as noted below
tensor_size_opinfo = OpInfo(
    lambda fd: fd.ops.size,
    "tensor_size",
    # TODO: Check correctness once there are operators that can consume a Vector
    sample_input_generator=None,
    error_input_generator=tensor_size_error_generator,
    fd_correctness_fn=None,
    fd_error_input_fn=tensor_api_test_fd_fn,
)
dynamic_shapes_ops.append(tensor_size_opinfo)

# TODO: Add correctness testing as noted below
vector_at_opinfo = OpInfo(
    lambda fd: fd.ops.at,
    "vector_at",
    # TODO: Check correctness once there are operators that can consume a Vector
    sample_input_generator=None,
    error_input_generator=vector_at_error_generator,
    fd_correctness_fn=None,
    fd_error_input_fn=vector_api_test_fd_fn,
)
dynamic_shapes_ops.append(vector_at_opinfo)


""" End Dynamic Shape Enabling Operations """

""" Start Normalization Operations """
normalization_ops = []

var_mean_opinfo = OpInfo(
    lambda fd: fd.ops.var_mean,
    "var_mean",
    dtypes=float_complex_dtypes,
    sample_input_generator=var_mean_generator,
    error_input_generator=reduction_error_generator,
    reference=torch.var_mean,
    symbolic_parameter_list=(ArgumentType.Symbolic, ArgumentType.Constant),
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
    symbolic_parameter_list=(ArgumentType.Symbolic, ArgumentType.Constant),
)
shape_ops.append(cat_opinfo)

broadcast_opinfo = OpInfo(
    lambda fd: fd.ops.broadcast,
    "broadcast",
    error_input_generator=broadcast_error_generator,
    symbolic_parameter_list=(ArgumentType.Symbolic, ArgumentType.Constant),
)
shape_ops.append(broadcast_opinfo)

# NOTE: The constant version of broadcast_in_dim opinfo tests the "shape"
# argument when a List of Constant Ints is used as an input.
# The symbolic parameter list lists the argument as "Constant" because
# otherwise an input is generated to attempt to supply the "shape" arg.
broadcast_in_dim_constant_opinfo = OpInfo(
    lambda fd: fd.ops.broadcast_in_dim,
    "broadcast_in_dim_constant",
    sample_input_generator=broadcast_in_dim_generator,
    error_input_generator=broadcast_in_dim_error_generator,
    reference=jax.lax.broadcast_in_dim,
    reference_type=ReferenceType.Jax,
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        # This argument is purposely Constant even though the positional
        # argument can also be symbolic.
        ArgumentType.Constant,
        ArgumentType.Constant,
    ),
)
shape_ops.append(broadcast_in_dim_constant_opinfo)


# NOTE: The symbolic version of broadcast_in_dim opinfo tests the "shape"
# argument with a Vector generated from another operation like ops.shape.
def broadcast_in_dim_sym_fn(fd, arg1, arg2, broadcast_dims):
    return fd.ops.broadcast_in_dim(arg1, arg2.shape(), broadcast_dims)


def jax_broadcast_in_dim_fn(arg1, arg2, broadcast_dims):
    return jax.lax.broadcast_in_dim(arg1, jax.numpy.shape(arg2), broadcast_dims)


broadcast_in_dim_symbolic_opinfo = OpInfo(
    lambda fd: partial(broadcast_in_dim_sym_fn, fd),
    "broadcast_in_dim_symbolic",
    sample_input_generator=broadcast_in_dim_generator,
    error_input_generator=broadcast_in_dim_error_generator,
    reference=jax_broadcast_in_dim_fn,
    reference_type=ReferenceType.Jax,
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Symbolic,
        ArgumentType.Constant,
    ),
)
shape_ops.append(broadcast_in_dim_symbolic_opinfo)


# translate between nvfuser and pytorch argument order for gather, take_along_dim, and index_select
def gather_wrapper(fn: callable, input: torch.Tensor, index: torch.Tensor, dim: int):
    return fn(input, dim, index)


gather_opinfo = OpInfo(
    lambda fd: fd.ops.gather,
    "gather",
    sample_input_generator=gather_generator,
    error_input_generator=take_along_axis_error_generator,
    reference=partial(gather_wrapper, torch.gather),
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Symbolic,
        ArgumentType.Constant,
    ),
)
shape_ops.append(gather_opinfo)

index_select_opinfo = OpInfo(
    lambda fd: fd.ops.index_select,
    "index_select",
    sample_input_generator=index_select_generator,
    error_input_generator=index_select_error_generator,
    reference=partial(gather_wrapper, torch.index_select),
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Symbolic,
        ArgumentType.Constant,
    ),
)
shape_ops.append(index_select_opinfo)

# NvFuser's API is significantly different than JAX.
# TODO: Change python frontend api to match JAX using a cpp wrapper function.
pad_opinfo = OpInfo(
    lambda fd: fd.ops.pad,
    "pad",
    error_input_generator=pad_error_generator,
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Constant,
        ArgumentType.Symbolic,
    ),
)
shape_ops.append(pad_opinfo)


permute_opinfo = OpInfo(
    lambda fd: fd.ops.permute,
    "permute",
    sample_input_generator=permute_generator,
    error_input_generator=permute_error_generator,
    reference=torch.permute,
    symbolic_parameter_list=(ArgumentType.Symbolic, ArgumentType.Constant),
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
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Constant,
        ArgumentType.Constant,
    ),
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
    symbolic_parameter_list=(
        ArgumentType.Symbolic,
        ArgumentType.Symbolic,
        ArgumentType.Constant,
    ),
)
shape_ops.append(take_along_axis_opinfo)

""" End Shape Operations """

""" Start Tensor Creation """
tensor_creation_ops = []

full_opinfo = OpInfo(
    lambda fd: fd.ops.full,
    "full",
    error_input_generator=full_error_generator,
    symbolic_parameter_list=(
        ArgumentType.Constant,
        ArgumentType.Symbolic,
        ArgumentType.Constant,
    ),
)
tensor_creation_ops.append(full_opinfo)

# Dynamic scalars are not checked at runtime, so we treat length, start, step as constants.
iota_opinfo = OpInfo(
    lambda fd: fd.ops.iota,
    "iota",
    dtypes=(torch.int64, torch.float64),
    error_input_generator=iota_error_generator,
    symbolic_parameter_list=(
        ArgumentType.ConstantScalar,
        ArgumentType.ConstantScalar,
        ArgumentType.ConstantScalar,
        ArgumentType.Constant,
    ),
)
tensor_creation_ops.append(iota_opinfo)

""" End Tensor Creation """

# Puts all opinfos into the "opinfos" list
opinfos.extend(unary_ops)
opinfos.extend(binary_ops)
opinfos.extend(ternary_ops)
opinfos.extend(fusion_input_ops)
opinfos.extend(dynamic_shapes_ops)
opinfos.extend(normalization_ops)
opinfos.extend(shape_ops)
opinfos.extend(tensor_creation_ops)
