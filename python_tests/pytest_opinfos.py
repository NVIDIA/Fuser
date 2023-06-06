import torch
import jax
import math
from pytest_core import (
    elementwise_unary_generator,
    _elementwise_unary_torch,
    OpInfo,
    ReferenceType,
    slice_sample_generator,
    slice_error_sample_generator,
)
from functools import partial
from make_tensor import int_float_dtypes

eps = 1e-2

opinfos = []

""" Start Unary-Float Operations """
elementwise_unary_ops = []

acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acos),
)
elementwise_unary_ops.append(acos_opinfo)

""" End Unary-Float Operations """

""" Start Shape Operations """

shape_ops = []

slice_opinfo = OpInfo(
    lambda fd: fd.ops.slice,
    "slice",
    sample_input_generator=slice_sample_generator,
    error_input_generator=slice_error_sample_generator,
    reference=jax.lax.slice,
    reference_type=ReferenceType.Jax,
)
shape_ops.append(slice_opinfo)

""" End Shape Operations """

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)
opinfos.extend(shape_ops)
