# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import jax.numpy as jnp
from torch.testing import make_tensor
from typing import Optional

from enum import Enum, auto


class ArgumentType(Enum):
    # a symbolic value requires an input argument during kernel execution
    Symbolic = auto()
    # scalar with constant value
    ConstantScalar = auto()
    # python number - int, float, complex, bool
    Constant = auto()


bool_dtypes = (torch.bool,)

int_dtypes = (
    torch.int32,
    torch.int64,
)

half_precision_float_dtypes = (
    torch.bfloat16,
    torch.float16,
)

full_precision_float_dtypes = (
    torch.float32,
    torch.float64,
)

complex_dtypes = (
    torch.complex64,
    torch.complex128,
)

# Half-precision float dtypes bf16, fp16 are skipped because nvfuser upcasts those dtypes to fp32
# but does not return the original type.
bool_int_dtypes = bool_dtypes + int_dtypes
float_dtypes = half_precision_float_dtypes + full_precision_float_dtypes
int_float_dtypes = int_dtypes + full_precision_float_dtypes
float_complex_dtypes = full_precision_float_dtypes + complex_dtypes
all_dtypes_except_reduced = int_dtypes + full_precision_float_dtypes + complex_dtypes
all_dtypes_except_bool = all_dtypes_except_reduced + half_precision_float_dtypes
all_dtypes = all_dtypes_except_bool + bool_dtypes

map_dtype_to_str = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}

torch_to_jax_dtype_map = {
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

torch_to_python_dtype_map = {
    torch.bool: bool,
    torch.uint8: int,
    torch.int8: int,
    torch.int16: int,
    torch.int32: int,
    torch.int64: int,
    torch.bfloat16: float,
    torch.float16: float,
    torch.float32: float,
    torch.float64: float,
    torch.complex64: complex,
    torch.complex128: complex,
}


def make_tensor_like(a):
    # type: (torch.Tensor) -> torch.Tensor
    """Returns a tensor with the same properties as the given tensor.

    Args:
        a (torch.Tensor): The tensor to copy properties from.

    Returns:
        torch.Tensor: A tensor with the same properties as :attr:`a`.
    """
    return torch.testing.make_tensor(
        a.shape, device=a.device, dtype=a.dtype, requires_grad=a.requires_grad
    )


def make_number(
    dtype: torch.dtype, low: Optional[float] = None, high: Optional[float] = None
):
    """Returns a random number with desired dtype

    Args:
        dtype (torch.dtype): Desired dtype for number.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range.

    Returns:
        (Scalar): The scalar number with specified dtype.
    """
    return make_tensor([1], device="cpu", dtype=dtype, low=low, high=high).item()


def find_nonmatching_dtype(dtype: torch.dtype):
    if dtype in int_float_dtypes:
        return torch.complex128
    elif dtype in complex_dtypes:
        return torch.double
    elif dtype is torch.bool:
        return torch.float32
    return None


def is_complex_dtype(dtype: torch.dtype):
    return dtype in complex_dtypes


def is_floating_dtype(dtype: torch.dtype):
    return dtype in float_dtypes


def is_integer_dtype(dtype: torch.dtype):
    return dtype in int_dtypes


def is_tensor(a):
    return isinstance(a, torch.Tensor)
