# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
import numpy as np

from pytest_fusion_definitions import default_fd_fn, parse_inputs_fusion_definition
from pytest_framework import create_op_test
from pytest_core import ReferenceType, OpInfo, SampleInput
from pytest_opinfos import opinfos
from pytest_utils import ArgumentType, is_tensor
from typing import Callable

from nvfuser import FusionDefinition


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def parse_args_fusion_execution(opinfo: OpInfo, *args):
    if len(args) == 0:
        return []

    if opinfo.symbolic_parameter_list is None:
        opinfo.symbolic_parameter_list = [ArgumentType.Symbolic] * len(args)
    assert len(opinfo.symbolic_parameter_list) == len(args)

    result = []
    for arg_type, a in zip(opinfo.symbolic_parameter_list, args):
        if arg_type == ArgumentType.Symbolic:
            if isinstance(a, list) and all(map(is_tensor, a)):
                result.extend(a)
            else:
                result.append(a)
    return result


# ****** Check an Operation's Results are Correct ******


def torch_correctness_test_fn(fd_fn: Callable, nvf_op: OpInfo, sample: SampleInput):
    with FusionDefinition() as fd:
        fd_fn(fd, nvf_op, *sample.args, **sample.kwargs)
    nvfuser_result = fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))
    torch_result = nvf_op.reference(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    if len(nvfuser_result) == 1:
        nvfuser_result = nvfuser_result[0]

    # TODO If dtype is fp16 or bf16, skip dtype check because nvfuser promotes to fp32 but does not return original dtype.
    torch.testing.assert_close(
        nvfuser_result, torch_result, equal_nan=True, atol=1e-3, rtol=0
    )


def jax_correctness_test_fn(fd_fn: Callable, nvf_op: OpInfo, sample: SampleInput):
    with FusionDefinition() as fd:
        fd_fn(fd, nvf_op, *sample.args, **sample.kwargs)
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
    torch.testing.assert_close(
        nvfuser_result, jax_result, equal_nan=True, atol=1e-3, rtol=0, check_dtype=False
    )


def python_correctness_test_fn(fd_fn: Callable, nvf_op: OpInfo, sample: SampleInput):
    # python reference function does not accept keyword arguments
    assert len(sample.kwargs) == 0

    with FusionDefinition() as fd:
        fd_fn(fd, nvf_op, *sample.args)
    nvfuser_result = fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))

    # expect only single result from function
    assert len(nvfuser_result) == 1

    # convert tensor arguments into flat, python lists
    python_sample = sample.python()

    # apply reference to python lists
    python_result = map(nvf_op.reference, *python_sample.args)

    # create pytorch tensor
    np_array = np.array(list(python_result))
    if np_array.shape == ():
        python_result = torch.tensor(
            np_array.item(), dtype=nvfuser_result[0].dtype, device="cuda"
        )
    else:
        python_result = torch.asarray(
            np_array, dtype=nvfuser_result[0].dtype, device="cuda"
        )

    # reshape flat output tensor into expected shape
    torch.testing.assert_close(
        nvfuser_result[0],
        python_result.reshape(nvfuser_result[0].shape),
        equal_nan=True,
        atol=1e-3,
        rtol=0,
    )


def correctness_test_fn(
    reference_type: ReferenceType,
    nvf_op: OpInfo,
    sample: SampleInput,
):
    _fd_fn = (
        nvf_op.fd_correctness_fn
        if nvf_op.fd_correctness_fn is not None
        else default_fd_fn
    )
    if reference_type == ReferenceType.Pytorch:
        return torch_correctness_test_fn(_fd_fn, nvf_op, sample)
    elif reference_type == ReferenceType.Jax:
        return jax_correctness_test_fn(_fd_fn, nvf_op, sample)
    elif reference_type == ReferenceType.Python:
        return python_correctness_test_fn(_fd_fn, nvf_op, sample)
    else:
        return None


@create_op_test(tuple(op for op in opinfos if op.reference is not None))
def test_correctness(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, dtype):
        result = correctness_test_fn(op.reference_type, op, sample)
        if result is not None:
            return result


# ****** Check a Definition Operation is not added to a Schedule ******


def definition_op_in_schedule_error_test_fn(opinfo: OpInfo, sample: SampleInput):
    class SchedError(FusionDefinition):
        def definition(self):
            # Create default fusion definition
            nvf_inputs = parse_inputs_fusion_definition(self, opinfo, *sample.args)
            result = opinfo.op(fd)(*nvf_inputs, **sample.kwargs)
            if isinstance(result, tuple):
                for a in result:
                    self.add_output(a)
            else:
                self.add_output(result)

        def schedule(self):
            # Attempt to add fusion operation during scheduling
            nvf_inputs = parse_inputs_fusion_definition(self, opinfo, *sample.args)
            opinfo.op(self)(*nvf_inputs, **sample.kwargs)

    fd = SchedError()
    nvfuser_result = fd.execute(parse_args_fusion_execution(opinfo, *sample.args))


# TODO Maybe only test a single dtype
@create_op_test(tuple(op for op in opinfos if op.sample_input_generator is not None))
def test_definition_op_in_schedule_error(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, dtype):
        with pytest.raises(
            RuntimeError, match=r"Attempting to add to a completed definition"
        ):
            definition_op_in_schedule_error_test_fn(op, sample)


# ****** Check that an Operation's API Gives Appropriate Input Errors ******


def errors_test_fn(
    nvf_op: OpInfo,
    sample: SampleInput,
):
    _fd_fn = (
        nvf_op.fd_error_input_fn
        if nvf_op.fd_error_input_fn is not None
        else default_fd_fn
    )
    with FusionDefinition() as fd:
        _fd_fn(fd, nvf_op, *sample.args, **sample.kwargs)
    fd.execute(parse_args_fusion_execution(nvf_op, *sample.args))


# A pair of parentheses () represents a capture group in regex.
# Escape parenthesis in regex string to match raw characters.
def _regex_escape_parenthesis(a: str) -> str:
    b = a.replace(r"(", r"\(")
    return b.replace(r")", r"\)")


@create_op_test(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op: OpInfo, dtype: torch.dtype):
    for sample, exception_type, exception_regex in op.error_input_generator(op, dtype):
        with pytest.raises(
            exception_type, match=_regex_escape_parenthesis(exception_regex)
        ):
            errors_test_fn(op, sample)
