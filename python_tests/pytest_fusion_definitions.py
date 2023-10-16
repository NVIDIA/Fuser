# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

from pytest_core import OpInfo
from pytest_utils import ArgumentType, is_tensor

from nvfuser import FusionDefinition
from nvfuser.pytorch_utils import (
    python_scalar_to_nvfuser_dtype,
    torch_dtype_to_nvfuser_dtype,
)


def parse_inputs_fusion_definition(fd: FusionDefinition, opinfo: OpInfo, *args):
    if len(args) == 0:
        return []

    nvf_args = []

    if opinfo.symbolic_parameter_list is None:
        opinfo.symbolic_parameter_list = [ArgumentType.Symbolic] * len(args)
    num_symbolic_parameters = len(opinfo.symbolic_parameter_list)
    assert num_symbolic_parameters == len(
        args
    ), f"{num_symbolic_parameters} vs {len(args)}"

    for arg_type, a in zip(opinfo.symbolic_parameter_list, args):
        if arg_type == ArgumentType.Symbolic:
            if isinstance(a, torch.Tensor):
                nvf_args.append(fd.from_pytorch(a))
            elif isinstance(a, list) and all(map(is_tensor, a)):
                nvf_args.append([fd.from_pytorch(inner_a) for inner_a in a])
            elif isinstance(a, list) or isinstance(a, tuple):
                nvf_args.append(fd.define_vector(a))
            else:
                # For symbolic scalars, we do not define with constant value.
                # Otherwise, it becomes a constant and is not a fusion input.
                nvf_args.append(fd.define_scalar(python_scalar_to_nvfuser_dtype(a)))
        elif arg_type == ArgumentType.ConstantScalar:
            assert not isinstance(a, torch.Tensor)
            nvf_args.append(fd.define_scalar(a))
        elif isinstance(a, torch.dtype):
            nvf_args.append(torch_dtype_to_nvfuser_dtype(a))
        else:
            assert not isinstance(a, torch.Tensor)
            assert arg_type == ArgumentType.Constant
            nvf_args.append(a)
    return nvf_args


# This function will purposely not generate a functional FusionDefintion as
# it lacks defining an output.  It is only meant to test the error checking
# of an operation.
def api_test_fd_fn(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    this_inputs = opinfo.op(fd)(**kwargs)


def default_fd_fn(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    result = opinfo.op(fd)(*nvf_inputs, **kwargs)
    if isinstance(result, tuple):
        for a in result:
            fd.add_output(a)
    else:
        fd.add_output(result)


def tensor_input_fd_fn(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    this_inputs = opinfo.op(fd)(**kwargs)
    t1 = fd.ops.add(nvf_inputs[0], this_inputs)
    fd.add_output(t1)


def tensor_api_test_fd_fn(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    out = opinfo.op(fd)(nvf_inputs[0], **kwargs)


def vector_api_test_fd_fn(fd: FusionDefinition, opinfo: OpInfo, *args, **kwargs):
    nvf_inputs = parse_inputs_fusion_definition(fd, opinfo, *args)
    v0 = nvf_inputs[0].shape()
    out = opinfo.op(fd)(v0, **kwargs)
