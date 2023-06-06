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

acosh_opinfo = OpInfo(
    lambda fd: fd.ops.acosh,
    "acosh",
    domain=(-1, math.inf),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acosh),
)
elementwise_unary_ops.append(acosh_opinfo)

asin_opinfo = OpInfo(
    lambda fd: fd.ops.asin,
    "asin",
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.asin),
)
elementwise_unary_ops.append(asin_opinfo)

asinh_opinfo = OpInfo(
    lambda fd: fd.ops.asinh,
    "asinh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.asinh),
)
elementwise_unary_ops.append(asinh_opinfo)

atan_opinfo = OpInfo(
    lambda fd: fd.ops.atan,
    "atan",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.atan),
)
elementwise_unary_ops.append(atan_opinfo)

atanh_opinfo = OpInfo(
    lambda fd: fd.ops.atanh,
    "atanh",
    domain=(-1 + eps, 1 + eps),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.atanh),
)
elementwise_unary_ops.append(atanh_opinfo)

cos_opinfo = OpInfo(
    lambda fd: fd.ops.cos,
    "cos",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.cos),
)
elementwise_unary_ops.append(cos_opinfo)

cosh_opinfo = OpInfo(
    lambda fd: fd.ops.cosh,
    "cosh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.cosh),
)
elementwise_unary_ops.append(cosh_opinfo)

erf_opinfo = OpInfo(
    lambda fd: fd.ops.erf,
    "erf",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erf),
)
elementwise_unary_ops.append(erf_opinfo)

erfc_opinfo = OpInfo(
    lambda fd: fd.ops.erfc,
    "erfc",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erfc),
)
elementwise_unary_ops.append(erfc_opinfo)

erfcinv_opinfo = OpInfo(
    lambda fd: fd.ops.erfcinv,
    "erfcinv",
    dtypes=int_float_dtypes,
    domain=(0.3, 0.7),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(lambda x: torch.erfinv(1 - x)),
)
elementwise_unary_ops.append(erfcinv_opinfo)

erfinv_opinfo = OpInfo(
    lambda fd: fd.ops.erfinv,
    "erfinv",
    dtypes=int_float_dtypes,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.erfinv),
)
elementwise_unary_ops.append(erfinv_opinfo)

exp_opinfo = OpInfo(
    lambda fd: fd.ops.exp,
    "exp",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.exp),
)
elementwise_unary_ops.append(exp_opinfo)

exp2_opinfo = OpInfo(
    lambda fd: fd.ops.exp2,
    "exp2",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.exp2),
)
elementwise_unary_ops.append(exp2_opinfo)

expm1_opinfo = OpInfo(
    lambda fd: fd.ops.expm1,
    "expm1",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.expm1),
)
elementwise_unary_ops.append(expm1_opinfo)

lgamma_opinfo = OpInfo(
    lambda fd: fd.ops.lgamma,
    "lgamma",
    dtypes=int_float_dtypes,
    domain=(-1.0 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.lgamma),
)
elementwise_unary_ops.append(lgamma_opinfo)

log_opinfo = OpInfo(
    lambda fd: fd.ops.log,
    "log",
    domain=(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log),
)
elementwise_unary_ops.append(log_opinfo)

log10_opinfo = OpInfo(
    lambda fd: fd.ops.log10,
    "log10",
    dtypes=int_float_dtypes,
    domain=(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log10),
)
elementwise_unary_ops.append(log10_opinfo)

log1p_opinfo = OpInfo(
    lambda fd: fd.ops.log1p,
    "log1p",
    dtypes=int_float_dtypes,
    domain=(-1 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log1p),
)
elementwise_unary_ops.append(log1p_opinfo)

log2_opinfo = OpInfo(
    lambda fd: fd.ops.log2,
    "log2",
    domain=(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.log2),
)
elementwise_unary_ops.append(log2_opinfo)

reciprocal_opinfo = OpInfo(
    lambda fd: fd.ops.reciprocal,
    "reciprocal",
    domain=(0 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.reciprocal),
)
elementwise_unary_ops.append(reciprocal_opinfo)

rsqrt_opinfo = OpInfo(
    lambda fd: fd.ops.rsqrt,
    "rqrt",
    domain=(0 + eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    reference=_elementwise_unary_torch(torch.rsqrt),
)
elementwise_unary_ops.append(rsqrt_opinfo)

sigmoid_opinfo = OpInfo(
    lambda fd: fd.ops.sigmoid,
    "sigmoid",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sigmoid),
)
elementwise_unary_ops.append(sigmoid_opinfo)

sin_opinfo = OpInfo(
    lambda fd: fd.ops.sin,
    "sin",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sin),
)
elementwise_unary_ops.append(sin_opinfo)

sinh_opinfo = OpInfo(
    lambda fd: fd.ops.sinh,
    "sinh",
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sinh),
)
elementwise_unary_ops.append(sinh_opinfo)

sqrt_opinfo = OpInfo(
    lambda fd: fd.ops.sqrt,
    "sqrt",
    domain=(0, math.inf),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.sqrt),
)
elementwise_unary_ops.append(sqrt_opinfo)

tan_opinfo = OpInfo(
    lambda fd: fd.ops.tan,
    "tan",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.tan),
)
elementwise_unary_ops.append(tan_opinfo)

tanh_opinfo = OpInfo(
    lambda fd: fd.ops.tanh,
    "tanh",
    dtypes=int_float_dtypes,
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.tanh),
)
elementwise_unary_ops.append(tanh_opinfo)

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
