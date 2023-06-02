import pytest
import numpy as np
import math
import torch
from torch.testing import assert_close

import inspect
import sys
import os
import re
import itertools
from functools import partial, wraps
from typing import List, Callable, Union
from collections import namedtuple
from copy import deepcopy

from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.jit_utils import RUN_CUDA

from make_tensor import make_tensor

# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser import (
        FusionCache,
        FusionDefinition,
        DataType,
        Tensor,
        version,
        compute_contiguity,
    )
    from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
except ImportError:
    pass

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM

all_dtypes = [
    torch.bool,
    # torch.uint8,
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

""" framework.py """


def _instantiate_opinfo_test_template(
    template: Callable, scope, *, opinfo, dtype: torch.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    test_name = "_".join((template.__name__, opinfo.name, str(dtype)))

    def test():
        return template(opinfo, dtype)

    test.__name__ = test_name
    test.__module__ = test.__module__
    return test


class ops:
    def __init__(self, opinfos, *, supported_dtypes=None, scope=None):
        self.opinfos = opinfos

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    def __call__(self, test_template):
        # NOTE Unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)
        for opinfo in self.opinfos:
            for dtype in sorted(opinfo._dtypes, key=lambda t: repr(t)):
                test = _instantiate_opinfo_test_template(
                    test_template,
                    self.scope,
                    opinfo=opinfo,
                    dtype=dtype,
                )
                # Adds the instantiated test to the requested scope
                self.scope[test.__name__] = test


def run_snippet(snippet, opinfo, dtype, *args, **kwargs):
    try:
        snippet(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()

        # Raises exceptions that occur with pytest, and returns debug information when
        # called otherwise
        # NOTE: PYTEST_CURRENT_TEST is set by pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise e
        return e, exc_info, snippet, opinfo, dtype, args, kwargs

    return None


""" opinfos.py """

Domain = namedtuple("Domain", "low high")


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"[SampleInput args={self.args} kwargs={self.kwargs}]"


# TODO: require use of generic Thunder dtypes (once they exist)
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op: Callable,
        name: str,
        *,
        dtypes=None,
        sample_input_generator,
        error_input_generator=None,
        torch_reference=None,
        numpy_reference=None,
        jax_reference=None,
        domain=(None, None),
    ):
        self.op = op
        self.name = name
        self._dtypes = dtypes if dtypes is not None else all_dtypes
        self.sample_input_generator = sample_input_generator
        self.error_input_generator = error_input_generator
        self.torch_reference = torch_reference
        self.numpy_reference = numpy_reference
        self.jax_reference = jax_reference
        self.domain = Domain(*domain)

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def sample_inputs(
        self, torch_dtype: torch.dtype, *, requires_grad: bool = False, **kwargs
    ):
        return self.sample_input_generator(self, torch_dtype, requires_grad, **kwargs)

    def error_inputs(self, **kwargs):
        return self.error_input_generator(self, **kwargs)

    def dtypes(self):
        return self._dtypes


# TODO Add small value, large value, and extremal-valued samples
def elementwise_unary_generator(
    op,
    dtype: torch.dtype,
    requires_grad: bool,
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
        # (),
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

    # Arbitrarily strided inputs
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        yield SampleInput(a)


def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)

        return op(torch.tensor(x)).item()

    return _fn


opinfos = []

elementwise_unary_ops = []
acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.acos),
)
elementwise_unary_ops.append(acos_opinfo)

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)


""" test_ops.py """


def unary_fusion_func(fd: FusionDefinition, operation, inputs):
    t0 = fd.from_pytorch(inputs[0])
    t1 = operation(fd)(t0)
    fd.add_output(t1)


def snippet_errors(nvf_op, sample, ex_type):
    ex = None
    try:
        with FusionDefinition() as fd:
            unary_fusion_func(fd, nvf_op, sample.args)
        fd.execute(*sample.args, **sample.kwargs)
    except Exception as e:
        ex = e

    assert ex is not None, f"Expected an exception"
    assert ex_type is type(
        ex
    ), f"Expected an exception with type {ex_type}, but found ex={ex}"


def snippet_torch_consistency(nvf_op, torch_op, sample):
    with FusionDefinition() as fd:
        unary_fusion_func(fd, nvf_op, sample.args)
    nvfuser_result = fd.execute(sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    assert_close(nvfuser_result[0], torch_result, equal_nan=True, atol=1e-3, rtol=0)


@ops(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op):
    for sample, ex_type in op.error_inputs(device):
        result = run_snippet(snippet_errors, op, None, op.op, sample, ex_type)
        if result is not None:
            return result


@ops(tuple(op for op in opinfos if op.torch_reference is not None))
def test_consistency(op, dtype: torch.dtype):
    for sample in op.sample_inputs(dtype):
        result = run_snippet(
            snippet_torch_consistency,
            op,
            dtype,
            op.op,
            op.torch_reference,
            sample,
        )
        if result is not None:
            return result
