# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
import numpy as np
from copy import deepcopy

from opinfo_fusion_definitions import default_fd_fn, parse_inputs_fusion_definition
from opinfo_framework import create_op_test, atexit_serde_create_op_test
from opinfo_core import ReferenceType, OpInfo, SampleInput
from opinfos import opinfos
from nvfuser.testing.utils import (
    ArgumentType,
    is_tensor,
    requiresJAX,
    check_captured_python_definition,
    check_cpp_translation,
    debug_serde,
    basic_serde_check,
)
from typing import Callable

from nvfuser import DataType, FusionCache, FusionDefinition
from nvfuser.pytorch_utils import retry_on_oom_or_skip_test


def serde_check_ops(test_fn: Callable):
    """
    The pytest framework decorator only checks per-operator and per-dtype.
    It doesn't check every single sample input. We check more input variations
    in the pytest benchmarks than test_python_frontend.py. This is a timesaving
    measure.
    """

    def inner_fn(*args, **kwargs):
        # NOTE: For debug purposes, clear FusionCache before running first test
        # so the behavior is more deterministic (PR #1848).
        if debug_serde:
            FusionCache.reset()
            assert FusionCache.get().num_fusions() == 0

        # Populate FusionCache
        test_fn(*args, **kwargs)

        # Serialize and Deserialize FusionCache
        basic_serde_check()

        # Run test with repopulated FusionCache
        return test_fn(*args, **kwargs)

    return inner_fn


def parse_args_fusion_execution(opinfo: OpInfo, *args):
    if len(args) == 0:
        return []

    symbolic_parameter_list = (
        opinfo.symbolic_parameter_list
        if opinfo.symbolic_parameter_list is not None
        else [ArgumentType.Symbolic] * len(args)
    )

    assert len(symbolic_parameter_list) >= len(args)

    result = []
    for arg_type, a in zip(symbolic_parameter_list, args):
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
    inputs = parse_args_fusion_execution(nvf_op, *sample.args)
    inputs_cap = deepcopy(inputs)
    nvfuser_result = fd.execute(inputs)
    assert check_captured_python_definition(nvfuser_result, fd, inputs_cap)

    if nvf_op.is_clonable:
        assert check_cpp_translation(
            nvfuser_result, fd, inputs_cap, supports_segmentation=True
        )

    torch_result = nvf_op.reference(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    if len(nvfuser_result) == 1:
        nvfuser_result = nvfuser_result[0]

    # TODO If dtype is fp16 or bf16, skip dtype check because nvfuser promotes
    # to fp32 but does not return original dtype.
    # TODO Add specific dtype tolerances
    torch.testing.assert_close(
        nvfuser_result, torch_result, equal_nan=True, atol=1e-3, rtol=0
    )


@requiresJAX
def jax_correctness_test_fn(fd_fn: Callable, nvf_op: OpInfo, sample: SampleInput):
    with FusionDefinition() as fd:
        fd_fn(fd, nvf_op, *sample.args, **sample.kwargs)
    inputs = parse_args_fusion_execution(nvf_op, *sample.args)
    inputs_cap = deepcopy(inputs)
    nvfuser_result = fd.execute(inputs)
    assert check_captured_python_definition(nvfuser_result, fd, inputs_cap)

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
    inputs = parse_args_fusion_execution(nvf_op, *sample.args)
    inputs_cap = deepcopy(inputs)
    nvfuser_result = fd.execute(inputs)
    assert check_captured_python_definition(nvfuser_result, fd, inputs_cap)

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
    elif reference_type == ReferenceType.Numpy:
        pytest.xfail("Numpy feference functions are not supported.")
    else:
        pytest.xfail("Reference function is not defined for this correctness test.")


# Run serde check for each operation and dtype but not for each sample input.
# NOTE: Disabled serde_check_ops decorator to avoid CI timeout.
@retry_on_oom_or_skip_test
def serde_test_fn(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, dtype):
        result = correctness_test_fn(op.reference_type, op, sample)
        if result is not None:
            return result


@atexit_serde_create_op_test(
    tuple(op for op in opinfos if op.sample_input_generator is not None)
)
def test_correctness(op: OpInfo, dtype: torch.dtype):
    return serde_test_fn(op, dtype)


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
@retry_on_oom_or_skip_test
def test_definition_op_in_schedule_error(op: OpInfo, dtype: torch.dtype):
    for sample in op.sample_input_generator(op, dtype):
        # clear cache for each sample
        FusionCache.reset()
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


# A pair of parentheses ()/[] represents a capture group in regex.
# Escape parenthesis in regex string to match raw characters.
def _regex_escape_parenthesis(a: str) -> str:
    b = a.replace(r"[", r"\[").replace(r"]", r"\]")
    return b.replace(r"(", r"\(").replace(r")", r"\)")


@create_op_test(tuple(op for op in opinfos if op.error_input_generator is not None))
@retry_on_oom_or_skip_test
def test_errors(op: OpInfo, dtype: torch.dtype):
    for sample, exception_type, exception_regex in op.error_input_generator(op, dtype):
        with pytest.raises(
            exception_type, match=_regex_escape_parenthesis(exception_regex)
        ):
            errors_test_fn(op, sample)


@pytest.mark.skip("https://github.com/NVIDIA/Fuser/issues/3740")
def test_cat_qwen2_v2():
    def qwen2_cat_fusion_2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2048, 512],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        T3 = fd.define_tensor(
            shape=[1, 4, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T5 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T6 = fd.define_tensor(
            shape=[1, 28, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 2048, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
        T13 = fd.ops.cast(T12, dtype=DataType.Float)
        S14 = fd.define_scalar(0.00000, dtype=DataType.Double)
        S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
        S16 = fd.define_scalar(1, dtype=DataType.Int)
        S17 = fd.define_scalar(2048, dtype=DataType.Int)
        S18 = fd.define_scalar(512, dtype=DataType.Int)
        T20 = fd.ops.uniform(
            S14,
            S15,
            shape=[S16, S17, S18],
            rng_seed=S2,
            rng_offset=S1,
            dtype=DataType.BFloat16,
        )
        S21 = fd.define_scalar(4.00000, dtype=DataType.Double)
        T22 = fd.ops.mul(T13, S21)
        S23 = fd.define_scalar(0.900000, dtype=DataType.Double)
        T24 = fd.ops.lt(T20, S23)
        T25 = fd.ops.cast(T24, dtype=DataType.Float)
        T41 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T42 = fd.ops.mul(T22, T25)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.neg(T43)
        T50 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T66 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T67 = fd.ops.cast(T44, dtype=DataType.BFloat16)
        T73 = fd.ops.broadcast_in_dim(
            T5, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T89 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 28, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        S90 = fd.define_scalar(1.11111, dtype=DataType.Double)
        T91 = fd.ops.mul(T42, S90)
        T97 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T98 = fd.ops.cat([T67, T66], dim=-1, manual_padding=0)
        T104 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T105 = fd.ops.cast(T89, dtype=DataType.Float)
        T106 = fd.ops.cast(T97, dtype=DataType.Float)
        T107 = fd.ops.cast(T98, dtype=DataType.Float)
        T108 = fd.ops.cast(T104, dtype=DataType.Float)
        T109 = fd.ops.cast(T3, dtype=DataType.Float)
        T110 = fd.ops.neg(T105)
        T111 = fd.ops.cast(T7, dtype=DataType.Float)
        T112 = fd.ops.mul(T107, T106)
        T113 = fd.ops.mul(T109, T108)
        T129 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 28, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T130 = fd.ops.cast(T110, dtype=DataType.BFloat16)
        T131 = fd.ops.add(T111, T91)
        T137 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T138 = fd.ops.cat([T130, T129], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T145 = fd.ops.cast(T131, dtype=DataType.BFloat16)
        T146 = fd.ops.cast(T137, dtype=DataType.Float)
        T147 = fd.ops.cast(T138, dtype=DataType.Float)
        T148 = fd.ops.cast(T144, dtype=DataType.Float)
        T149 = fd.ops.cast(T6, dtype=DataType.Float)
        T155 = fd.ops.reshape(T145, new_shape=[1, 2048, 4, 128])
        T156 = fd.ops.add(T113, T112)
        T157 = fd.ops.mul(T147, T146)
        T158 = fd.ops.mul(T149, T148)
        T159 = fd.ops.permute(T155, dims=[0, 2, 1, 3])
        T160 = fd.ops.cast(T156, dtype=DataType.BFloat16)
        T167 = fd.ops.broadcast_in_dim(
            T159, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T174 = fd.ops.broadcast_in_dim(
            T160, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T181 = fd.ops.broadcast_in_dim(
            T167, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T188 = fd.ops.broadcast_in_dim(
            T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T189 = fd.ops.add(T158, T157)
        T195 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
        T201 = fd.ops.reshape(T188, new_shape=[1, 28, 2048, 128])
        T202 = fd.ops.cast(T189, dtype=DataType.BFloat16)
        T203 = fd.ops.stride_order(T195, stride_order=[3, 2, 1, 0])
        T204 = fd.ops.stride_order(T201, stride_order=[3, 2, 1, 0])
        T205 = fd.ops.stride_order(T202, stride_order=[3, 2, 1, 0])
        fd.add_output(T159)
        fd.add_output(T160)
        fd.add_output(T203)
        fd.add_output(T204)
        fd.add_output(T205)

    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device="cuda:0"),
        25546,
        1400552702872758,
        torch.randn(1048576, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 4, 2048, 128), (1048576, 128, 512, 1)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(7340032, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 28, 2048, 128), (7340032, 128, 3584, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 512), dtype=torch.bfloat16, device="cuda:0"
        ),
    ]

    with FusionDefinition() as fd:
        qwen2_cat_fusion_2(fd)

    fd.execute(inputs)
