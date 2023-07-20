# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
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


# uint8, int8, int16, bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32
# but does not return the original type.
all_dtypes = (
    torch.bool,
    # torch.uint8,
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.bfloat16,
    # torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

# bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32 but does not return the original type.
int_float_dtypes = (
    torch.int32,
    torch.int64,
    # torch.bfloat16,
    # torch.float16,
    torch.float32,
    torch.float64,
)

float_complex_dtypes = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

int_dtypes = (torch.int32, torch.int64)
float_dtypes = (torch.float32, torch.float64, torch.bfloat16, torch.float16)
complex_dtypes = (torch.complex64, torch.complex128)

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
