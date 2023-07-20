# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from pytest_utils import all_dtypes, ArgumentType
from typing import Callable, Optional
import torch
import jax.numpy as jnp
from enum import Enum
from dataclasses import dataclass


class ReferenceType(Enum):
    Pytorch = 0
    Jax = 1
    Numpy = 2
    Python = 3


@dataclass
class ErrorSample:
    kwargs: dict
    ex_str: str
    ex_type: Exception = RuntimeError


@dataclass
class Domain:
    low: int
    high: int


_torch_to_jax_dtype_map = {
    torch.bool: jnp.bool_,
    torch.uint8: jnp.uint8,
    torch.int8: jnp.int8,
    torch.int16: jnp.int16,
    torch.int32: jnp.int32,
    torch.int64: jnp.int64,
    torch.bfloat16: jnp.bfloat16,
    torch.float16: jnp.float16,
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
    torch.complex64: jnp.complex64,
    torch.complex128: jnp.complex128,
}


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

    def jax(self):
        def to_jax(t):
            if isinstance(t, torch.Tensor):
                return jnp.array(t.cpu().numpy())
            if isinstance(t, torch.dtype):
                return _torch_to_jax_dtype_map[t]
            return t

        # Note: We assume arguments have flat hierarchy.
        # TODO Add support for kwargs
        args = map(to_jax, self.args)
        return SampleInput(*args, *self.kwargs.values())


@dataclass
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    op: Callable

    name: str

    # Set of valid inputs for this operation
    domain: Domain = Domain(None, None)

    # Set of valid dtypes for this operation
    dtypes: tuple = all_dtypes

    # Generates valid inputs
    sample_input_generator: Callable = None

    # Generates error inputs
    error_input_generator: Callable = None

    # Function of FusionDefintion operations for valid inputs
    fd_correctness_fn: Callable = None

    # Function of FusionDefintion operations for error inputs
    fd_error_input_fn: Callable = None

    # Reference function for operation
    reference: Callable = None

    # Designate which framework defines the reference
    reference_type: ReferenceType = ReferenceType.Pytorch

    # Nvfuser requires reduction axes to be constant values.
    # symbolic_parameter_list specifies whether an operation's parameters are symbolic.
    # All keyword arguments are considered constant.
    # If symbolic_parameter_list is None, then we assume all parameters to be symbolic.
    symbolic_parameter_list: Optional[list[ArgumentType]] = None
