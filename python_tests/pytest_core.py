# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from pytest_utils import all_dtypes
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

        # Thunder uses tree_map here. We assume arguments have flat hierarchy.
        # TODO add support for kwargs
        args = map(to_jax, self.args)
        return SampleInput(*args, *self.kwargs.values())


class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op: Callable,
        name: str,
        *,
        dtypes=None,
        sample_input_generator=None,
        error_input_generator=None,
        reference=None,
        reference_type=ReferenceType.Pytorch,
        domain=(None, None),
        is_fusion_input_op: bool = False,
        symbolic_parameter_list: Optional[list[bool]] = None,
    ):
        self.op = op
        self.name = name
        self._dtypes = dtypes if dtypes is not None else all_dtypes
        self.sample_input_generator = sample_input_generator
        self.error_input_generator = error_input_generator
        self.reference = reference
        self.refernce_fn_type = reference_type
        self.domain = Domain(*domain)
        self.is_fusion_input_op = is_fusion_input_op
        # Nvfuser requires reduction axes to be constant values.
        # symbolic_parameter_list specifies whether an operation's parameters are symbolic.
        # All keyword arguments are considered constant.
        # If symbolic_parameter_list is None, then we assume all parameters to be symbolic.
        self.symbolic_parameter_list = symbolic_parameter_list

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def sample_inputs(
        self, torch_dtype: torch.dtype, *, requires_grad: bool = False, **kwargs
    ):
        return self.sample_input_generator(self, torch_dtype, requires_grad, **kwargs)

    def error_inputs(
        self, torch_dtype: torch.dtype, *, requires_grad: bool = False, **kwargs
    ):
        return self.error_input_generator(self, torch_dtype, requires_grad, **kwargs)

    def dtypes(self):
        return self._dtypes
