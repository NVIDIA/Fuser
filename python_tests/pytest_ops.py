# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import numpy as np

from torch.testing import assert_close
from pytest_framework import ops, run_snippet
from pytest_core import ReferenceType, OpInfo, SampleInput
from pytest_opinfos import opinfos
from typing import Optional, Callable
from functools import partial

from nvfuser import FusionDefinition


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def snippet_definition_op_in_schedule_error(nvf_op, sample):
    inputs = [
        torch.randn(8, 8, 8, device="cuda"),
    ]

    class SchedError(FusionDefinition):
        def definition(self):
            self.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            self.t1 = fd.ops.tanh(fd.t0)
            self.add_output(fd.t1)

        def schedule(self):
            nvf_inputs = [fd.from_pytorch(x) for x in inputs if type(x) is torch.Tensor]
            nvf_op(self)(*nvf_inputs, **sample.kwargs)

    exception = None
    try:
        fd = SchedError()
        _ = fd.execute(inputs)
    except Exception as e:
        exception = e

    assert exception is not None, "Expected an exception"
    exception_str = "Attempting to add to a completed definition!"
    assert exception_str in str(
        exception
    ), "Failed to find correct expection error message"


def opinfo_fusion_func(fd: FusionDefinition, operation: Callable, inputs, **kwargs):
    nvf_inputs = [fd.from_pytorch(x) for x in inputs if type(x) is torch.Tensor]
    t1 = operation(fd)(*nvf_inputs, **kwargs)
    fd.add_output(t1)


def input_fusion_func(fd: FusionDefinition, operation: Callable, inputs, **kwargs):
    nvf_inputs = [fd.from_pytorch(x) for x in inputs if type(x) is torch.Tensor]
    this_inputs = operation(fd)(**kwargs)
    t1 = fd.ops.add(nvf_inputs[0], this_inputs)
    fd.add_output(t1)


def snippet_errors(
    fusion_func,
    nvf_op: Callable,
    sample: SampleInput,
    exception_type: Exception,
    exception_str: Optional[str],
):
    exception = None
    try:
        with FusionDefinition() as fd:
            fusion_func(fd, nvf_op, sample.args, **sample.kwargs)
        fd.execute(*sample.args)
        print(fd)
    except Exception as e:
        exception = e

    assert exception is not None, "Expected an exception"
    assert exception_type is type(
        exception
    ), f"Expected an exception with type {exception_type}, but found exception={exception}"
    assert exception_str is None or exception_str in str(
        exception
    ), "Failed to find correct expection error message"


def snippet_torch_consistency(
    fusion_func, nvf_op: Callable, torch_op, sample: SampleInput
):
    with FusionDefinition() as fd:
        fusion_func(fd, nvf_op, sample.args)
    nvfuser_result = fd.execute(sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    assert_close(nvfuser_result[0], torch_result, equal_nan=True, atol=1e-3, rtol=0)


def snippet_jax_consistency(fusion_func, nvf_op: Callable, jax_op, sample: SampleInput):
    with FusionDefinition() as fd:
        fusion_func(fd, nvf_op, sample.args, **sample.kwargs)
    nvfuser_result = fd.execute(sample.args, **sample.kwargs)

    jax_sample = sample.jax()
    jax_result = jax_op(*jax_sample.args, **jax_sample.kwargs)

    # NOTE: this strange unpacking is to handle NumPy's and JAX's sometimes odd
    #   number vs. array representation. In particular, NumPy can mimic
    #   Python numbers, but `asarray` doesn't understand this mimicry
    np_array = np.array(jax_result)
    if np_array.shape == ():
        jax_result = torch.tensor(np_array.item(), device=nvfuser_result[0].device)
    else:
        jax_result = torch.asarray(np_array, device=nvfuser_result[0].device)

    # NOTE: dtype is not checked because jax will translate int64, float64, and complex128 to int32, float32 and complex64
    assert_close(nvfuser_result[0], jax_result, equal_nan=True, check_dtype=False)


def snippet_consistency(reference_type: ReferenceType, is_fusion_input_op: bool):
    fusion_func = input_fusion_func if is_fusion_input_op else opinfo_fusion_func
    if reference_type == ReferenceType.Pytorch:
        return partial(snippet_torch_consistency, fusion_func)
    elif reference_type == ReferenceType.Jax:
        return partial(snippet_jax_consistency, fusion_func)
    else:
        return None


@ops(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op: OpInfo, dtype: torch.dtype):
    fusion_func = input_fusion_func if op.is_fusion_input_op else opinfo_fusion_func
    print(op.is_fusion_input_op)
    for sample, ex_type, ex_regex in op.error_inputs(dtype):
        result = run_snippet(
            partial(snippet_errors, fusion_func),
            op,
            dtype,
            op.op,
            sample,
            ex_type,
            ex_regex,
        )
        if result is not None:
            return result


@ops(tuple(op for op in opinfos if op.reference is not None))
def test_consistency(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_inputs(dtype):
        result = run_snippet(
            snippet_consistency(op.refernce_fn_type, op.is_fusion_input_op),
            op,
            dtype,
            op.op,
            op.reference,
            sample,
        )
        if result is not None:
            return result


# TODO Maybe only test a single dtype
@ops(tuple(op for op in opinfos if op.reference is not None))
def test_definition_op_in_schedule_error(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_inputs(torch.float32):
        result = run_snippet(
            snippet_definition_op_in_schedule_error,
            op,
            dtype,
            op.op,
            sample,
        )
        if result is not None:
            return result
