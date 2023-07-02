# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import numpy as np

from torch.testing import assert_close
from pytest_framework import create_op_test
from pytest_core import ReferenceType, OpInfo, SampleInput
from pytest_opinfos import opinfos
from pytest_utils import ArgumentType
from typing import Callable, Optional

from nvfuser import FusionDefinition
from nvfuser.pytorch_utils import (
    python_scalar_to_nvfuser_dtype,
    torch_dtype_to_nvfuser_dtype,
)


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_tensor(a):
    return isinstance(a, torch.Tensor)


def parse_inputs_fusion_definition(fd: FusionDefinition, opinfo: OpInfo, *args):
    if len(args) == 0:
        return []

    nvf_args = []

    if opinfo.symbolic_parameter_list is None:
        opinfo.symbolic_parameter_list = [ArgumentType.Symbolic] * len(args)
    assert len(opinfo.symbolic_parameter_list) == len(args)

    for arg_type, a in zip(opinfo.symbolic_parameter_list, args):
        if arg_type == ArgumentType.Symbolic:
            if type(a) is torch.Tensor:
                nvf_args.append(fd.from_pytorch(a))
            elif type(a) is list and all(map(is_tensor, a)):
                nvf_args.append([fd.from_pytorch(inner_a) for inner_a in a])
            elif type(a) is list or type(a) is tuple:
                nvf_args.append(fd.define_vector(a))
            else:
                # For symbolic scalars, we do not define with constant value.
                # Otherwise, it becomes a constant and is not a fusion input.
                nvf_args.append(fd.define_scalar(python_scalar_to_nvfuser_dtype(a)))
        elif arg_type == ArgumentType.ConstantScalar:
            assert type(a) is not torch.Tensor
            nvf_args.append(fd.define_scalar(a))
        elif isinstance(a, torch.dtype):
            nvf_args.append(torch_dtype_to_nvfuser_dtype(a))
        else:
            assert type(a) is not torch.Tensor
            assert arg_type == ArgumentType.Constant
            nvf_args.append(a)
    return nvf_args


def parse_args_fusion_execution(opinfo: OpInfo, *args):
    if len(args) == 0:
        return []

    result = []
    for arg_type, a in zip(opinfo.symbolic_parameter_list, args):
        if arg_type == ArgumentType.Symbolic:
            if type(a) is list and all(map(is_tensor, a)):
                result.extend(a)
            else:
                result.append(a)
    return result


def opinfo_fusion_func(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    result = opinfo.op(fd)(*nvf_inputs, **kwargs)
    if type(result) is tuple:
        for a in result:
            fd.add_output(a)
    else:
        fd.add_output(result)


def input_fusion_func(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    this_inputs = opinfo.op(fd)(**kwargs)
    t1 = fd.ops.add(nvf_inputs[0], this_inputs)
    fd.add_output(t1)


def definition_op_in_schedule_error_test_fn(opinfo: OpInfo, sample: SampleInput):
    inputs = [
        torch.randn(8, 8, 8, device="cuda"),
    ]

    class SchedError(FusionDefinition):
        def definition(self):
            self.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            self.t1 = fd.ops.tanh(fd.t0)
            self.add_output(fd.t1)

        def schedule(self):
            nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *sample.args)
            opinfo.op(self)(*nvf_inputs, **sample.kwargs)

    exception = None
    try:
        fd = SchedError()
        fd.execute(parse_args_fusion_execution(opinfo, *sample.args))
    except Exception as e:
        exception = e

    assert exception is not None, "Expected an exception"
    exception_str = "Attempting to add to a completed definition!"
    assert exception_str in str(
        exception
    ), "Failed to find correct expection error message"


def errors_test_fn(
    fusion_func: Callable,
    nvf_op: OpInfo,
    sample: SampleInput,
    exception_type: Exception,
    exception_str: Optional[str],
):
    exception = None
    try:
        with FusionDefinition() as fd:
            fusion_func(fd, nvf_op, *sample.args, **sample.kwargs)
        fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))
    except Exception as e:
        exception = e

    assert exception is not None, "Expected an exception"
    assert exception_type is type(
        exception
    ), f"Expected an exception with type {exception_type}, but found exception={exception}"
    assert exception_str is None or exception_str in str(
        exception
    ), "Failed to find correct expection error message"


def torch_correctness_test_fn(
    fusion_func: Callable, nvf_op: OpInfo, sample: SampleInput
):
    with FusionDefinition() as fd:
        fusion_func(fd, nvf_op, *sample.args, **sample.kwargs)
    nvfuser_result = fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))
    torch_result = nvf_op.reference(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    if len(nvfuser_result) == 1:
        nvfuser_result = nvfuser_result[0]

    assert_close(nvfuser_result, torch_result, equal_nan=True, atol=1e-3, rtol=0)


def jax_correctness_test_fn(fusion_func: Callable, nvf_op: OpInfo, sample: SampleInput):
    with FusionDefinition() as fd:
        fusion_func(fd, nvf_op, *sample.args, **sample.kwargs)
    nvfuser_result = fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))

    jax_sample = sample.jax()
    jax_result = nvf_op.reference(*jax_sample.args, **jax_sample.kwargs)

    # NOTE: this strange unpacking is to handle NumPy's and JAX's sometimes odd
    #   number vs. array representation. In particular, NumPy can mimic
    #   Python numbers, but `asarray` doesn't understand this mimicry
    np_array = np.array(jax_result)
    if np_array.shape == ():
        jax_result = torch.tensor(np_array.item(), device="cuda")
    else:
        jax_result = torch.asarray(np_array, device="cuda")

    if len(nvfuser_result) == 1:
        nvfuser_result = nvfuser_result[0]

    # NOTE: dtype is not checked because jax will translate int64, float64, and complex128 to int32, float32 and complex64
    assert_close(
        nvfuser_result, jax_result, equal_nan=True, atol=1e-3, rtol=0, check_dtype=False
    )


def correctness_test_fn(
    reference_type: ReferenceType,
    is_fusion_input_op: bool,
    op: OpInfo,
    sample: SampleInput,
):
    fusion_func = input_fusion_func if is_fusion_input_op else opinfo_fusion_func
    if reference_type == ReferenceType.Pytorch:
        return torch_correctness_test_fn(fusion_func, op, sample)
    elif reference_type == ReferenceType.Jax:
        return jax_correctness_test_fn(fusion_func, op, sample)
    else:
        return None


@create_op_test(tuple(op for op in opinfos if op.reference is not None))
def test_correctness(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, dtype):
        result = correctness_test_fn(
            op.reference_type, op.is_fusion_input_op, op, sample
        )
        if result is not None:
            return result


# TODO Maybe only test a single dtype
@create_op_test(tuple(op for op in opinfos if op.sample_input_generator is not None))
def test_definition_op_in_schedule_error(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, torch.float32):
        result = definition_op_in_schedule_error_test_fn(
            op,
            sample,
        )
        if result is not None:
            return result


@create_op_test(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op: OpInfo, dtype: torch.dtype):
    fusion_func = input_fusion_func if op.is_fusion_input_op else opinfo_fusion_func
    for sample, ex_type, ex_regex in op.error_input_generator(op, dtype):
        result = errors_test_fn(
            fusion_func,
            op,
            sample,
            ex_type,
            ex_regex,
        )
        if result is not None:
            return result
